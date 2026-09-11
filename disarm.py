"""
Phase 4 — Disarm: destroy any hidden payload by randomizing every LSB in
every layer the scan flagged (not just one hardcoded layer — with two
embed sites now live, hardcoding one was a real bug, not a hypothetical).
"""
import numpy as np
import torch



from scan import scan_model
from stego import get_param_by_name


def disarm_layer(weights: np.ndarray, rng_seed: int = 42) -> np.ndarray:
    flat = weights.flatten()
    as_int = flat.view(np.uint32).copy()
    rng = np.random.default_rng(rng_seed)
    random_bits = rng.integers(0, 2, size=as_int.size, dtype=np.uint32)
    as_int = (as_int & ~np.uint32(1)) | random_bits
    return as_int.view(np.float32).reshape(weights.shape)


def disarm_model(model, findings):
    """Disarm every distinct layer named in `findings` (scan_model's output)."""
    done = set()
    for f in findings:
        layer_name = f["layer"]
        if layer_name in done:
            continue
        param = get_param_by_name(model, layer_name)
        disarmed_w = disarm_layer(param.detach().numpy())
        param.data = torch.from_numpy(disarmed_w.copy())
        done.add(layer_name)
    return model


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "tampered_model.pt"

    # Loaded as the whole model object -- disarm.py works on whatever
    # architecture the file actually contains, same as scan.py.
    model = torch.load(model_path, weights_only=False)
    model.eval()

    findings, _ = scan_model(model)
    disarm_model(model, findings)

    torch.save(model, "disarmed_model.pt")
    print(f"Disarmed {len({f['layer'] for f in findings})} flagged layer(s).")
    print("Saved disarmed_model.pt.")

    reverify, _ = scan_model(model)
    if reverify:
        print("WARNING: payload still detectable after disarm — something's wrong.")
        for f in reverify:
            print(f)
    else:
        print("Re-scan after disarm: CLEAN. Both payloads destroyed, extraction now fails.")