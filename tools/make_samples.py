"""Generates the benign, infected, dangerous and malformed test files plus the
Python reference results the web tests compare against.

Usage: python tools/make_samples.py
Writes web/tests/fixtures/* and web/public/samples/*.
"""
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from kavach_ref import INT_VIEW, load_named, scan_named

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "web" / "tests" / "fixtures"
PUB = ROOT / "web" / "public" / "samples"
EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
MYSTERY = b"youdontknowme-UNKNOWN-STRUCTURED-DEMO-PAYLOAD-XYZ"


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.fc = torch.nn.Linear(512, 64)
        self.head = torch.nn.Linear(64, 10)


def embed(t: torch.Tensor, payload: bytes, offset: int) -> torch.Tensor:
    ints = t.detach().contiguous().view(INT_VIEW[t.dtype]).flatten().clone()
    bits = torch.from_numpy(np.unpackbits(np.frombuffer(payload, np.uint8)).astype(np.int64))
    seg = ints[offset:offset + bits.numel()].to(torch.int64)
    ints[offset:offset + bits.numel()] = ((seg & ~1) | bits).to(ints.dtype)
    return ints.view(t.dtype).view(t.shape)


def infect(sd, rng):
    """Plants EICAR in one tensor and the mystery payload in another, at
    offsets that are not multiples of 8."""
    big = [k for k, v in sd.items() if v.is_floating_point() and v.numel() >= 1200]
    out = dict(sd)
    for payload, name in zip([EICAR, MYSTERY], rng.choice(big, size=2, replace=False)):
        limit = out[name].numel() - len(payload) * 8
        offset = int(rng.integers(0, limit // 8)) * 8 + int(rng.integers(1, 8))
        offset = min(offset, limit)
        out[name] = embed(out[name], payload, offset)
    return out


class Dangerous:
    """Serialises as a call to os.system. torch.save only writes the opcodes;
    nothing here is ever loaded."""
    def __reduce__(self):
        return (os.system, ("echo kavach-test",))


def cast(sd, dtype):
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}


def save_variant(path: Path, kind: str, sd, model=None):
    if kind == "flat":
        torch.save(sd, path)
    elif kind == "nested":
        torch.save({"epoch": 3, "model": {"state_dict": sd}, "optimizer": {"lr": 0.01},
                    "extra": [sd["fc.weight"].t()]}, path)
    elif kind == "whole":
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        model = model.to(next(v for v in sd.values() if v.is_floating_point()).dtype)
        torch.save(model, path)
    elif kind == "safetensors":
        save_file({k: v.contiguous() for k, v in sd.items()}, str(path))


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    base = TinyNet().state_dict()
    expected = {}
    cases = []
    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        clean = cast(base, dtype)
        dirty = infect(clean, rng)
        for kind in ["flat", "nested", "whole", "safetensors"]:
            if kind == "whole" and dtype != torch.float32:
                continue
            ext = "safetensors" if kind == "safetensors" else "pt"
            for label, sd in [("clean", clean), ("infected", dirty)]:
                name = f"{label}-{kind}-{dtype_name}.{ext}"
                save_variant(FIX / name, kind, sd, TinyNet())
                cases.append(name)
    for name in cases:
        expected[name] = scan_named(load_named(str(FIX / name)))
        expected[name]["dangerous"] = []

    dirty32 = infect(cast(base, torch.float32), np.random.default_rng(11))
    torch.save({"state_dict": dirty32, "hook": Dangerous()}, FIX / "dangerous.pt")
    ref = scan_named(load_named_from_obj({"state_dict": dirty32}))
    ref["dangerous"] = ["posix.system" if os.name == "posix" else "nt.system"]
    expected["dangerous.pt"] = ref

    raw = (FIX / "clean-flat-fp32.pt").read_bytes()
    (FIX / "malformed-truncated.pt").write_bytes(raw[: len(raw) // 2])
    (FIX / "malformed-empty.pt").write_bytes(b"")
    (FIX / "malformed-random.pt").write_bytes(np.random.default_rng(3).bytes(4096))
    st = (FIX / "clean-safetensors-fp32.safetensors").read_bytes()
    (FIX / "malformed-header.safetensors").write_bytes((10**9).to_bytes(8, "little") + st[8:])
    (FIX / "malformed-legacy.pt").write_bytes(pickle.dumps({"a": 1}, protocol=2))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("archive/data.pkl", pickle.dumps({"a": 1}, protocol=2))
    (FIX / "malformed-deflated.pt").write_bytes(buf.getvalue())

    (FIX / "expected.json").write_text(json.dumps(expected, indent=1))
    for src, dst in [("clean-flat-fp32.pt", "clean-model.pt"),
                     ("infected-nested-fp32.pt", "infected-checkpoint.pt"),
                     ("infected-safetensors-bf16.safetensors", "infected-bf16.safetensors"),
                     ("dangerous.pt", "dangerous-pickle.pt")]:
        (PUB / dst).write_bytes((FIX / src).read_bytes())
    print(f"wrote {len(expected)} reference results to {FIX}")


def load_named_from_obj(obj):
    from kavach_ref import flatten
    return list(flatten(obj))


if __name__ == "__main__":
    main()
