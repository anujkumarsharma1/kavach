"""
Phase 2b -- Build the tampered model.

CHANGED again from the first update: this now plants payloads at MANY
locations across MANY different layers in one pass (multi_tamper), not
just one hit in fc.weight. Two things this proves that a single hit
doesn't:

1. It's a richer live demo -- "found 18 hits across the model" reads as
   a real scan, not a canned single-file check.
2. It works on layers with completely different shapes and sizes without
   ever being told which layer to target -- it just asks each layer "are
   you big enough to hold this payload?" and picks from whichever say
   yes. That's the actual mechanism behind "architecture-agnostic," not
   a slide claim.

multi_tamper() takes a model and never references a layer by name --
this is what makes it work unmodified on a completely different
architecture (see models_zoo.py for the chest X-ray swap).

Verified (numpy-only, no torch needed for the core logic): ran across
10 random seeds against a mock model with 20+ layers of mixed sizes --
every planted payload was found by scan_model with zero collisions and
zero misses, every time. See chat for the transcript.
"""
import numpy as np
import torch

from payload import MYSTERY_PAYLOAD, NUM_EICAR_LOCATIONS, NUM_MYSTERY_LOCATIONS, PAYLOAD
from stego import embed_payload, extract_payload


def multi_tamper(model, payload_plan, rng_seed=1337):
    """Embed each (payload, count) pair in `payload_plan` at `count`
    distinct layers, chosen only by which parameters are large enough to
    hold the payload -- never by name. Guaranteed disjoint: no layer is
    ever used for more than one payload, across the whole plan, so
    nothing can collide and corrupt another payload's bits.

    Returns a list of (layer_name, offset, payload_bytes) for every
    location actually used -- print or log this, you'll want it for the
    "here's exactly where we planted it" moment in the demo.
    """
    rng = np.random.default_rng(rng_seed)
    all_params = list(model.named_parameters())
    used_names = set()
    locations = []

    for payload, count in payload_plan:
        min_size = len(payload) * 8 + 64  # payload bits + margin
        candidates = [
            (name, p) for name, p in all_params
            if name not in used_names and p.numel() >= min_size
        ]
        n = min(count, len(candidates))
        if n < count:
            print(
                f"WARNING: only {n} layer(s) large enough for a {len(payload)}-byte "
                f"payload (wanted {count}). Using {n}."
            )
        if n == 0:
            continue
        chosen_idx = rng.choice(len(candidates), size=n, replace=False)
        for idx in chosen_idx:
            name, p = candidates[idx]
            w = p.detach().numpy()
            max_start = w.size - len(payload) * 8
            offset = int(rng.integers(0, max_start + 1))
            tampered = embed_payload(w, payload, start=offset)
            p.data = torch.from_numpy(tampered.copy())
            used_names.add(name)
            locations.append((name, offset, payload))

    return locations


if __name__ == "__main__":
    import argparse

    from models_zoo import load_chest_xray, load_resnet18

    parser = argparse.ArgumentParser(description="Plant hidden payloads into a demo model.")
    parser.add_argument(
        "--model", choices=["resnet18", "chest-xray"], default="resnet18",
        help="Which architecture to tamper (default: resnet18). chest-xray requires "
             "`pip install torchxrayvision scikit-image` -- see models_zoo.py.",
    )
    args = parser.parse_args()

    model, _ = load_chest_xray() if args.model == "chest-xray" else load_resnet18()
    model.eval()

    locations = multi_tamper(
        model,
        [(PAYLOAD, NUM_EICAR_LOCATIONS), (MYSTERY_PAYLOAD, NUM_MYSTERY_LOCATIONS)],
    )

    # Saved as the whole model object (not just a state_dict) so scan.py and
    # disarm.py never need to be told which architecture produced this file --
    # the file itself carries it.
    torch.save(model, "tampered_model.pt")
    print(f"Planted {len(locations)} payloads across {len(locations)} distinct layers:")
    for name, offset, payload in locations:
        print(f"  {name}  offset={offset}  ({len(payload)} bytes, {'EICAR' if payload == PAYLOAD else 'mystery'})")
    print("Saved tampered_model.pt.")

    # Self-check every single planted location before you trust the scanner
    # to find them -- if embedding itself is broken, better to know now.
    params = dict(model.named_parameters())
    for name, offset, payload in locations:
        w = params[name].detach().numpy()
        recovered = extract_payload(w, len(payload), start=offset)
        assert recovered == payload, f"Round-trip FAILED at {name}:{offset} -- stop and debug before demoing."
    print(f"Self-check OK: all {len(locations)} locations round-trip cleanly.")