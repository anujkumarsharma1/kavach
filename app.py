import glob

import streamlit as st
import torch
import torchvision
from PIL import Image

from disarm import disarm_model
from payload import PAYLOAD
from scan import scan_model
from tamper import tamper_model

st.set_page_config(page_title="KAVACH", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
code, pre, .stCodeBlock, .stCode {
    font-family: 'IBM Plex Mono', monospace !important;
}

.kavach-verdict {
    padding: 1.0rem 1.3rem;
    border-radius: 8px;
    border: 1px solid;
    margin: 0.4rem 0 1.2rem 0;
    font-size: 1.05rem;
    line-height: 1.5;
}
.kavach-verdict.clean {
    background: #EAF6EF;
    border-color: #2E7D4F;
    color: #1E4E31;
}
.kavach-verdict.tampered {
    background: #FCEBEA;
    border-color: #C0392B;
    color: #7A231A;
}

.kavach-finding {
    background: #FFFFFF;
    border: 1px solid #E3E7ED;
    border-left: 4px solid #C0392B;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    color: #12181F;
    line-height: 1.55;
}
.kavach-finding.unknown {
    border-left-color: #B8860B;
}

.kavach-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    margin-bottom: 0.45rem;
}
.kavach-tag.known {
    background: #FCEBEA;
    color: #7A231A;
}
.kavach-tag.unknown {
    background: #FCF3D9;
    color: #7A5B0A;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ KAVACH — Antivirus for AI Models")
st.caption("Detects, extracts, and disarms steganographic payloads hidden in neural network weights.")

with st.expander("How this works, in 20 seconds"):
    st.markdown(
        """
1. **Load** a real, pretrained ImageNet model — genuine trained weights, genuine floating-point noise.
2. **Tamper** — two payloads get hidden in the least-significant bits of two different layers. Predictions barely move (~1e-7 value shift).
3. **Scan** — every layer is swept for statistically anomalous LSB regions; hidden bytes are extracted and validated, not just flagged.
4. **Disarm** — every flagged layer gets its LSBs randomized, destroying the payload for good, at the same negligible cost to accuracy.
"""
    )

if "clean_model" not in st.session_state:
    st.session_state.clean_model = None
    st.session_state.weights_meta = None
    st.session_state.tampered_model = None
    st.session_state.disarmed_model = None
    st.session_state.findings = None
    st.session_state.reverify = None
    st.session_state.predictions = {}

if st.sidebar.button("Reset demo"):
    st.session_state.clear()
    st.rerun()

st.sidebar.caption("Kavach — Precision Care Challenge")


def build_model():
    return torchvision.models.resnet18(weights=None)


def load_sample_images():
    return sorted(
        glob.glob("sample_images/*.jpg") + glob.glob("sample_images/*.jpeg") + glob.glob("sample_images/*.png")
    )


def predict_all(model, weights_meta, image_paths):
    transforms = weights_meta.transforms()
    categories = weights_meta.meta["categories"]
    out = {}
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        batch = transforms(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(batch)[0], dim=0)
        idx = int(torch.argmax(probs))
        out[path] = (categories[idx], float(probs[idx]))
    return out


col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("1. Load Baseline", use_container_width=True):
        weights_meta = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights_meta)
        model.eval()
        st.session_state.clean_model = model
        st.session_state.weights_meta = weights_meta
        images = load_sample_images()
        if images:
            st.session_state.predictions["clean"] = predict_all(model, weights_meta, images)

with col2:
    if st.button("2. Tamper", use_container_width=True, disabled=st.session_state.clean_model is None):
        tampered_model = build_model()
        tampered_model.load_state_dict(st.session_state.clean_model.state_dict())
        tamper_model(tampered_model)
        tampered_model.eval()
        st.session_state.tampered_model = tampered_model
        images = load_sample_images()
        if images:
            st.session_state.predictions["tampered"] = predict_all(
                tampered_model, st.session_state.weights_meta, images
            )

with col3:
    if st.button("3. Scan", use_container_width=True, disabled=st.session_state.tampered_model is None):
        st.session_state.findings = scan_model(st.session_state.tampered_model, payload_len_bytes=len(PAYLOAD))

with col4:
    if st.button("4. Disarm", use_container_width=True, disabled=not st.session_state.findings):
        disarmed_model = build_model()
        disarmed_model.load_state_dict(st.session_state.tampered_model.state_dict())
        disarm_model(disarmed_model, st.session_state.findings)
        disarmed_model.eval()
        st.session_state.disarmed_model = disarmed_model
        st.session_state.reverify = scan_model(disarmed_model, payload_len_bytes=len(PAYLOAD))
        images = load_sample_images()
        if images:
            st.session_state.predictions["disarmed"] = predict_all(
                disarmed_model, st.session_state.weights_meta, images
            )

st.divider()

if st.session_state.findings is not None:
    st.subheader("Scan Report")
    if not st.session_state.findings:
        st.markdown(
            '<div class="kavach-verdict clean">✅ <b>CLEAN</b> — no suspicious regions found.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="kavach-verdict tampered">🚨 <b>TAMPERED</b> — '
            f"{len(st.session_state.findings)} hidden region(s) found and extracted.</div>",
            unsafe_allow_html=True,
        )

    for f in st.session_state.findings:
        known = f["matched_signature"] is not None
        tag_class = "known" if known else "unknown"
        tag_label = "KNOWN SIGNATURE MATCH" if known else "UNKNOWN — FLAGGED BY ANOMALY, NO SIGNATURE"
        sig_line = f["matched_signature"] or "None — extracted on statistical grounds alone"
        st.markdown(
            f"""<div class="kavach-finding {tag_class}">
<span class="kavach-tag {tag_class}">{tag_label}</span><br>
<b>Layer:</b> <code>{f['layer']}</code> &nbsp;·&nbsp; <b>bit offset:</b> {f['offset']}<br>
<b>Printable ratio:</b> {f['printable_ratio']:.1%} (natural weight noise sits ~37–43%)<br>
<b>Matched signature:</b> {sig_line}
</div>""",
            unsafe_allow_html=True,
        )
        st.code(f["decoded_preview"])

if st.session_state.reverify is not None:
    st.subheader("Post-Disarm Re-Verification")
    if not st.session_state.reverify:
        st.markdown(
            '<div class="kavach-verdict clean">✅ Extraction now <b>FAILS</b> on every layer — '
            "payloads destroyed, model is clean.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"{len(st.session_state.reverify)} region(s) still detectable — disarm did not fully work.")

if st.session_state.predictions:
    st.subheader("Does the model still work? (prediction agreement)")
    for stage_name, preds in st.session_state.predictions.items():
        st.markdown(f"**{stage_name}**")
        for path, (cls, conf) in preds.items():
            st.write(f"- {path}: {cls} ({conf:.1%})")