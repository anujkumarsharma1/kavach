"""
Phase 2 — Core trick: hide arbitrary bytes in the LSBs of float32 weights,
and read them back out.
"""
import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def embed_payload(weights: np.ndarray, payload: bytes, start: int = 0) -> np.ndarray:
    """Return a COPY of `weights` with `payload` hidden in the LSB of each float32,
    starting at float-index `start` (lets tamper.py plant payloads at any offset,
    not just index 0)."""
    flat = weights.flatten()
    as_int = flat.view(np.uint32).copy()
    bits = bytes_to_bits(payload)
    end = start + len(bits)
    if end > as_int.size:
        raise ValueError(f"Payload needs {len(bits)} weight slots at offset {start}, this layer only has {as_int.size}.")
    as_int[start:end] = (as_int[start:end] & ~np.uint32(1)) | bits.astype(np.uint32)
    return as_int.view(np.float32).reshape(weights.shape)


def extract_bits(weights: np.ndarray, num_bits: int, start: int = 0) -> np.ndarray:
    """Read back `num_bits` LSBs starting at float-index `start` in the
    flattened array (lets scan.py probe every bit-phase alignment, not
    just index 0)."""
    flat = weights.flatten()
    as_int = flat.view(np.uint32)
    return (as_int[start:start + num_bits] & np.uint32(1)).astype(np.uint8)


def extract_payload(weights: np.ndarray, num_bytes: int, start: int = 0) -> bytes:
    return bits_to_bytes(extract_bits(weights, num_bytes * 8, start=start))


def get_param_by_name(model, name: str):
    """
    Walk a dotted parameter name like 'layer4.1.conv2.weight' — the exact
    names scan_model reports via named_parameters() — down to the actual
    nn.Parameter. Lets tamper.py and disarm.py address whatever layer a
    scan flagged, generically, instead of hardcoding one layer.
    """
    module = model
    parts = name.split(".")
    for part in parts[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return getattr(module, parts[-1])


if __name__ == "__main__":
    # Self-test: round-trip a string through fake "weights" before you
    # trust this on the real model.
    test_weights = np.random.default_rng(0).standard_normal(10000).astype(np.float32)
    secret = b"HELLO KAVACH"
    tampered = embed_payload(test_weights, secret)
    recovered = extract_payload(tampered, len(secret))
    assert recovered == secret, f"Round-trip FAILED: got {recovered!r}"
    max_shift = np.abs(tampered - test_weights).max()
    print(f"Round-trip OK. Max value shift: {max_shift:.2e} (should be tiny, ~1e-7 scale).")