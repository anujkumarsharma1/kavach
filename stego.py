"""
Phase 2 — Core trick: hide arbitrary bytes in the LSBs of float32 weights,
and read them back out.
"""
import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def embed_payload(weights: np.ndarray, payload: bytes) -> np.ndarray:
    """Return a COPY of `weights` with `payload` hidden in the LSB of each float32."""
    flat = weights.flatten()
    as_int = flat.view(np.uint32).copy()
    bits = bytes_to_bits(payload)
    if len(bits) > as_int.size:
        raise ValueError(f"Payload needs {len(bits)} weight slots, this layer only has {as_int.size}.")
    as_int[: len(bits)] = (as_int[: len(bits)] & ~np.uint32(1)) | bits.astype(np.uint32)
    return as_int.view(np.float32).reshape(weights.shape)


def extract_bits(weights: np.ndarray, num_bits: int) -> np.ndarray:
    """Read back `num_bits` LSBs from the start of the flattened array."""
    flat = weights.flatten()
    as_int = flat.view(np.uint32)
    return (as_int[:num_bits] & np.uint32(1)).astype(np.uint8)


def extract_payload(weights: np.ndarray, num_bytes: int) -> bytes:
    return bits_to_bytes(extract_bits(weights, num_bytes * 8))


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