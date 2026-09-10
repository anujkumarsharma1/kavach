"""
Two demo models behind one interface. This file is the ONLY place that
knows the difference between them -- stego.py, scan.py, disarm.py, and
multi_tamper() in tamper.py never reference a model type, a layer name,
or an architecture. That's not a coincidence, it's the actual proof of
the "cross-architecture" claim: the same four files run unmodified
against whichever model you load here.

IMPORTANT -- read before using the chest X-ray option:
- torchxrayvision is a THIRD dependency beyond what you already have
  (`pip install torchxrayvision`, plus `scikit-image` for image loading,
  which torchxrayvision pulls in as its own dependency). It downloads
  ~30MB of pretrained weights the first time you load the model, cached
  to ~/.torchxrayvision after that.
- The library's own documentation states plainly: NOT FOR MEDICAL USE.
  We are using it as a real, independently-trained clinical-imaging
  architecture to prove detection is architecture-agnostic -- not to
  make any diagnostic claim. Say this explicitly in the pitch if you use
  this option; do not let a judge assume otherwise.
- I verified the API below against the library's current README and
  PyPI page (fetched today), and it matches your original doc's own
  earlier note on this. I could NOT execute-test this specific loading
  and inference code myself -- torch isn't installable in my sandbox
  without pulling a multi-GB CUDA build, the exact trap your original
  doc already warned about. Run the self-test at the bottom of this
  file FIRST, on its own, before wiring it into the live demo.
- Hard timebox: 30 minutes to get `python models_zoo.py --check-xray`
  printing real pathology scores. If it's fighting you past that,
  abandon it and present with ResNet18 only -- you already have a
  complete, working, tested demo without this.
"""
import glob

import numpy as np
import torch
import torchvision
from PIL import Image


# ---------------------------------------------------------------------------
# Option 1: ImageNet ResNet18 (torchvision) -- your original demo model.
# ---------------------------------------------------------------------------

def load_resnet18():
    weights_meta = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights_meta)
    model.eval()
    return model, weights_meta


def predict_resnet18(model, weights_meta, image_paths):
    transforms = weights_meta.transforms()
    categories = weights_meta.meta["categories"]
    out = {}
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        batch = transforms(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(batch)[0], dim=0)
        idx = int(torch.argmax(probs))
        out[path] = f"{categories[idx]} ({float(probs[idx]):.1%})"
    return out


def sample_images_resnet18():
    return sorted(
        glob.glob("sample_images/*.jpg")
        + glob.glob("sample_images/*.jpeg")
        + glob.glob("sample_images/*.png")
    )


# ---------------------------------------------------------------------------
# Option 2: Chest X-ray DenseNet121 (torchxrayvision). NOT FOR MEDICAL USE.
# pip install torchxrayvision scikit-image
# ---------------------------------------------------------------------------

def load_chest_xray():
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model, None


def predict_chest_xray(model, _unused, image_paths, top_k=3):
    import skimage.io
    import torchxrayvision as xrv

    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    )
    out = {}
    for path in image_paths:
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = img[None, ...]
        img = transform(img)
        img_t = torch.from_numpy(img)
        with torch.no_grad():
            scores = model(img_t[None, ...])[0].numpy()
        top_idx = np.argsort(scores)[::-1][:top_k]
        out[path] = ", ".join(f"{model.pathologies[i]} {scores[i]:.0%}" for i in top_idx)
    return out


def sample_images_chest_xray():
    return sorted(
        glob.glob("chest_sample_images/*.jpg")
        + glob.glob("chest_sample_images/*.jpeg")
        + glob.glob("chest_sample_images/*.png")
    )


MODEL_ZOO = {
    "ImageNet ResNet18 (torchvision)": {
        "load": load_resnet18,
        "predict": predict_resnet18,
        "sample_images": sample_images_resnet18,
    },
    "Chest X-ray DenseNet121 (torchxrayvision) \u2014 NOT FOR MEDICAL USE": {
        "load": load_chest_xray,
        "predict": predict_chest_xray,
        "sample_images": sample_images_chest_xray,
    },
}


if __name__ == "__main__":
    import sys

    if "--check-xray" in sys.argv:
        print("Loading torchxrayvision DenseNet121 (downloads ~30MB on first run)...")
        model, _ = load_chest_xray()
        print(f"Loaded. {sum(p.numel() for p in model.parameters()):,} parameters, "
              f"{len(list(model.named_parameters()))} named tensors, "
              f"{len(model.pathologies)} pathology outputs.")
        images = sample_images_chest_xray()
        if not images:
            print(
                "No images in chest_sample_images/ yet. Get one with:\n"
                "  git clone --depth 1 https://github.com/mlmed/torchxrayvision.git\n"
                "  cp torchxrayvision/tests/*.png chest_sample_images/\n"
                "(see the plan doc for details)"
            )
        else:
            preds = predict_chest_xray(model, None, images)
            for path, result in preds.items():
                print(f"{path}: {result}")
        print("If the above printed real pathology names and percentages, you're good to wire this into app.py.")
    else:
        print("Usage: python models_zoo.py --check-xray")
        print("Run this standalone check BEFORE touching app.py's model selector.")
