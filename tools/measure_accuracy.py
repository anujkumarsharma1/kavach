"""Measures whether hiding strings in the weights, and disarming them, changes
ImageNet ResNet18 predictions. Uses the Imagenette 160px validation split
(3,925 ImageNet images of 10 classes, https://github.com/fastai/imagenette).

Usage: python tools/measure_accuracy.py /path/to/imagenette2-160/val
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

from kavach_ref import INT_VIEW, scan_named
from make_samples import EICAR, MYSTERY, embed

WNID_TO_IMAGENET = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
    "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701,
}


def predictions(model, loader):
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x).argmax(1))
    return torch.cat(out)


def main(val_dir):
    torch.set_num_threads(8)
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights).eval()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = torchvision.datasets.ImageFolder(val_dir, tf)
    labels = torch.tensor([WNID_TO_IMAGENET[ds.classes[t]] for t in ds.targets])
    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=4)

    clean_sd = {k: v.clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(2026)
    big = [k for k, v in clean_sd.items() if v.is_floating_point() and v.numel() >= 100_000]
    tampered = dict(clean_sd)
    for payload, name in zip([EICAR, EICAR, MYSTERY, MYSTERY, MYSTERY], rng.choice(big, size=5, replace=False)):
        offset = int(rng.integers(0, tampered[name].numel() - len(payload) * 8))
        tampered[name] = embed(tampered[name], payload, offset)
    found = scan_named(list(tampered.items()))
    flagged = sorted({f["layer"] for f in found["findings"]})

    disarmed = dict(tampered)
    for name in flagged:
        ints = disarmed[name].contiguous().view(INT_VIEW[disarmed[name].dtype]).clone()
        noise = torch.from_numpy(np.random.default_rng(42).integers(0, 2, ints.numel())).view(ints.shape).to(ints.dtype)
        disarmed[name] = ((ints & ~1) | noise).view(torch.float32)
    rescan = scan_named(list(disarmed.items()))

    results = {}
    preds = {}
    for label, sd in [("clean", clean_sd), ("tampered", tampered), ("disarmed", disarmed)]:
        model.load_state_dict(sd)
        preds[label] = predictions(model, loader)
        results[f"{label}_top1"] = round(float((preds[label] == labels).float().mean()) * 100, 2)
    results["tampered_same_prediction_as_clean"] = int((preds["tampered"] == preds["clean"]).sum())
    results["disarmed_same_prediction_as_clean"] = int((preds["disarmed"] == preds["clean"]).sum())
    results["images"] = len(ds)
    results["payloads_planted"] = 5
    results["findings_after_tamper"] = len(found["findings"])
    results["findings_after_disarm"] = len(rescan["findings"])
    results["max_weight_change_tampered"] = float(max((tampered[k] - clean_sd[k]).abs().max() for k in flagged))
    results["torch"] = torch.__version__
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
