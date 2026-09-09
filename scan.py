"""
Phase 3 — Scan every layer for hidden payloads: detection, extraction,
validation, and localization in one pass.
"""
import numpy as np
import torch
import torchvision

from stego import bits_to_bytes, extract_bits

KNOWN_SIGNATURES = [
    b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE",
]


def printable_ratio(data: bytes) -> float:
    return sum(1 for b in data if 32 <= b <= 126) / max(len(data), 1)


def scan_layer(weights: np.ndarray, window_bits: int, stride: int = None, threshold: float = 0.70):
    stride = stride or window_bits
    flat = weights.flatten()
    total_bits = flat.size  # 1 bit slot per value
    hits = []
    for start in range(0, max(total_bits - window_bits, 0), stride):
        bits = extract_bits(flat[start : start + window_bits], window_bits)
        data = bits_to_bytes(bits)
        ratio = printable_ratio(data)
        if ratio >= threshold:
            hits.append({"offset": start, "printable_ratio": ratio, "decoded": data})
    return hits


def scan_model(model, payload_len_bytes: int, margin_bits: int = 32):
    # Window sized close to the payload length so the signal isn't diluted
    # by surrounding noise — see the printable-ratio numbers above.
    window_bits = payload_len_bytes * 8 + margin_bits
    report = []
    for name, param in model.named_parameters():
        w = param.detach().numpy()
        hits = scan_layer(w, window_bits)
        for hit in hits:
            matched_sig = next((sig for sig in KNOWN_SIGNATURES if sig in hit["decoded"]), None)
            report.append(
                {
                    "layer": name,
                    "offset": hit["offset"],
                    "printable_ratio": hit["printable_ratio"],
                    "matched_signature": matched_sig.decode() if matched_sig else None,
                    "decoded_preview": hit["decoded"][:80],
                }
            )
    return report


if __name__ == "__main__":
    import sys

    from payload import PAYLOAD

    model_path = sys.argv[1] if len(sys.argv) > 1 else "tampered_model.pt"
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    findings = scan_model(model, payload_len_bytes=len(PAYLOAD))

    if not findings:
        print(f"CLEAN — no suspicious regions found in {model_path}.")
    for f in findings:
        print(f"TAMPERED — layer '{f['layer']}', bit offset {f['offset']}")
        print(f"  printable ratio: {f['printable_ratio']:.1%} (baseline noise ~37-43%)")
        print(f"  matched signature: {f['matched_signature']}")
        print(f"  decoded bytes: {f['decoded_preview']}")