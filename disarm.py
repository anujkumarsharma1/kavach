"""
Phase 4 — Disarm: destroy any hidden payload by randomizing every LSB
in the flagged layer . 
"""
import numpy as np
import torch
import torchvision

from payload import PAYLOAD
from scan import scan_model


def disarm_layer(weights: np.ndarray, rng_seed: int = 42) -> np.ndarray:
    flat = weights.flatten()
    as_int = flat.view(np.uint32).copy()
    rng = np.random.default_rng(rng_seed)
    random_bits = rng.integers(0, 2, size=as_int.size, dtype=np.uint32)
    as_int = (as_int & ~np.uint32(1)) | random_bits
    return as_int.view(np.float32).reshape(weights.shape)


if __name__ == "__main__":
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load("tampered_model.pt", weights_only=True))
    model.eval()

    disarmed_w = disarm_layer(model.fc.weight.detach().numpy())
    model.fc.weight.data = torch.from_numpy(disarmed_w.copy())

    torch.save(model.state_dict(), "disarmed_model.pt")
    print("Saved disarmed_model.pt.")

    findings = scan_model(model, payload_len_bytes=len(PAYLOAD))
    if findings:
        print("WARNING: payload still detectable after disarm — something's wrong.")
        for f in findings:
            print(f)
    else:
        print("Re-scan after disarm: CLEAN. Payload destroyed, extraction now fails.")
