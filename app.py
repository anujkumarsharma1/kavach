import glob

import streamlit as st
import torch
import torchvision
from PIL import Image

from disarm import disarm_layer
from payload import PAYLOAD
from scan import scan_model
from stego import embed_payload

st.set_page_config(page_title="KAVACH", layout="wide")
st.title("🛡️ KAVACH — Antivirus for AI Models")
st.caption("Detects, extracts, and disarms steganographic payloads hidden in neural network weights.")

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
        clean_w = st.session_state.clean_model.fc.weight.detach().numpy()
        tampered_w = embed_payload(clean_w, PAYLOAD)
        tampered_model = build_model()
        tampered_model.load_state_dict(st.session_state.clean_model.state_dict())
        tampered_model.fc.weight.data = torch.from_numpy(tampered_w.copy())
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
        tampered_w = st.session_state.tampered_model.fc.weight.detach().numpy()
        disarmed_w = disarm_layer(tampered_w)
        disarmed_model = build_model()
        disarmed_model.load_state_dict(st.session_state.tampered_model.state_dict())
        disarmed_model.fc.weight.data = torch.from_numpy(disarmed_w.copy())
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
        st.success("CLEAN — no suspicious regions found.")
    for f in st.session_state.findings:
        st.error(f"TAMPERED — layer `{f['layer']}`, bit offset {f['offset']}")
        st.write(f"Printable ratio: **{f['printable_ratio']:.1%}** (natural weight noise sits around 37-43%)")
        st.write(f"Matched signature: **{f['matched_signature']}**")
        st.code(f["decoded_preview"])

if st.session_state.reverify is not None:
    st.subheader("Post-Disarm Re-Verification")
    if not st.session_state.reverify:
        st.success("Extraction now FAILS — payload destroyed, model is clean.")
    else:
        st.warning("Payload still detectable — disarm did not fully work.")

if st.session_state.predictions:
    st.subheader("Does the model still work? (prediction agreement)")
    for stage_name, preds in st.session_state.predictions.items():
        st.write(f"**{stage_name}**")
        for path, (cls, conf) in preds.items():
            st.write(f"- {path}: {cls} ({conf:.1%})")