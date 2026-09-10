import copy
import os

import pandas as pd
import streamlit as st
import torch

from disarm import disarm_layer
from models_zoo import MODEL_ZOO
from payload import MYSTERY_PAYLOAD, NUM_EICAR_LOCATIONS, NUM_MYSTERY_LOCATIONS, PAYLOAD
from scan import STRIDE_BYTES, THRESHOLD, WINDOW_BYTES, scan_model
from tamper import multi_tamper

st.set_page_config(page_title="KAVACH", page_icon="\U0001F6E1\uFE0F", layout="wide")

st.title("\U0001F6E1\uFE0F KAVACH \u2014 Antivirus for AI Models")
st.caption(
    "Detects, extracts, and disarms steganographic payloads hidden in neural network "
    "weights \u2014 across every parameter tensor, at any bit offset, on any architecture, "
    "without needing to know the payload's length in advance."
)

with st.expander("How this differs from pickle scanners like Fickling / ModelScan"):
    st.markdown(
        "Tools like **Fickling** (Trail of Bits) and **ModelScan** (Protect AI) catch "
        "*code-execution* attacks \u2014 a malicious pickle opcode that runs the moment the "
        "file loads. They inspect the deserialization instructions, not the weight "
        "**values**.\n\n"
        "KAVACH targets a different, complementary surface: data smuggled inside the "
        "numeric values of the weights themselves. A model tampered this way executes "
        "zero unsafe code on load and passes a pickle-opcode scan cleanly \u2014 the payload "
        "is invisible to that entire tool category. KAVACH is built for that specific "
        "blind spot, not to replace those tools."
    )

if "clean_model" not in st.session_state:
    st.session_state.clean_model = None
    st.session_state.model_choice = None
    st.session_state.weights_meta = None
    st.session_state.tampered_model = None
    st.session_state.disarmed_model = None
    st.session_state.findings = None
    st.session_state.layer_summary = None
    st.session_state.reverify = None
    st.session_state.predictions = {}

if st.sidebar.button("Reset demo"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Demo model",
    list(MODEL_ZOO.keys()),
    disabled=st.session_state.clean_model is not None,
    help="Locked once you click Load Baseline \u2014 hit Reset demo to switch.",
)
if "NOT FOR MEDICAL USE" in model_choice:
    st.sidebar.caption(
        "\u26a0\ufe0f Used here to prove architecture-agnostic detection on a real "
        "clinical-imaging model. Not a diagnostic tool, not validated for medical use."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Detection parameters**\n\n"
    f"- Window: {WINDOW_BYTES} bytes\n"
    f"- Stride: {STRIDE_BYTES} bytes (overlapping)\n"
    f"- Bit-phase alignments checked: 8\n"
    f"- Printable-ratio threshold: {THRESHOLD:.0%}\n\n"
    f"None of these are told the true payload length, offset, or which "
    f"layer(s) \u2014 all of that is recovered, not assumed."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("1. Load Baseline", use_container_width=True):
        zoo = MODEL_ZOO[model_choice]
        with st.spinner("Loading model..."):
            model, weights_meta = zoo["load"]()
        st.session_state.clean_model = model
        st.session_state.model_choice = model_choice
        st.session_state.weights_meta = weights_meta
        images = zoo["sample_images"]()
        if images:
            st.session_state.predictions["clean"] = zoo["predict"](model, weights_meta, images)

with col2:
    if st.button("2. Tamper", use_container_width=True, disabled=st.session_state.clean_model is None):
        zoo = MODEL_ZOO[st.session_state.model_choice]
        tampered_model = copy.deepcopy(st.session_state.clean_model)
        multi_tamper(
            tampered_model,
            [(PAYLOAD, NUM_EICAR_LOCATIONS), (MYSTERY_PAYLOAD, NUM_MYSTERY_LOCATIONS)],
        )
        st.session_state.tampered_model = tampered_model
        images = zoo["sample_images"]()
        if images:
            st.session_state.predictions["tampered"] = zoo["predict"](
                tampered_model, st.session_state.weights_meta, images
            )

with col3:
    if st.button("3. Scan", use_container_width=True, disabled=st.session_state.tampered_model is None):
        with st.spinner("Scanning every layer across 8 bit-phase alignments..."):
            findings, layer_summary = scan_model(st.session_state.tampered_model)
        st.session_state.findings = findings
        st.session_state.layer_summary = layer_summary

with col4:
    if st.button("4. Disarm", use_container_width=True, disabled=not st.session_state.findings):
        zoo = MODEL_ZOO[st.session_state.model_choice]
        flagged_layers = sorted({f["layer"] for f in st.session_state.findings})
        disarmed_model = copy.deepcopy(st.session_state.tampered_model)
        params = dict(disarmed_model.named_parameters())
        for layer_name in flagged_layers:
            disarmed_w = disarm_layer(params[layer_name].detach().numpy())
            params[layer_name].data = torch.from_numpy(disarmed_w.copy())
        st.session_state.disarmed_model = disarmed_model
        reverify, _ = scan_model(disarmed_model)
        st.session_state.reverify = reverify
        images = zoo["sample_images"]()
        if images:
            st.session_state.predictions["disarmed"] = zoo["predict"](
                disarmed_model, st.session_state.weights_meta, images
            )

st.divider()

if st.session_state.findings is not None:
    st.subheader("Scan Report")

    n_layers = len(st.session_state.layer_summary or {})
    n_flagged = len(st.session_state.findings)
    n_critical = sum(1 for f in st.session_state.findings if f["severity"] == "CRITICAL")
    n_suspicious = n_flagged - n_critical
    top_ratio = max([f["printable_ratio"] for f in st.session_state.findings], default=0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Parameter tensors scanned", n_layers)
    m2.metric("CRITICAL (signature matched)", n_critical)
    m3.metric("SUSPICIOUS (unknown, structured)", n_suspicious)
    m4.metric("Highest printable ratio", f"{top_ratio:.0%}" if n_flagged else "\u2014")

    if not st.session_state.findings:
        st.success("CLEAN \u2014 no suspicious regions found across any layer, any bit-phase.")
    else:
        df = pd.DataFrame(
            [
                {
                    "Layer": f["layer"],
                    "Severity": f["severity"],
                    "Bits": f"[{f['bit_start']}:{f['bit_end']}]",
                    "Ratio": f"{f['printable_ratio']:.0%}",
                    "Signature": f["matched_signature"] or "unknown \u2014 flagged on structure",
                }
                for f in st.session_state.findings
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        crit_example = next((f for f in st.session_state.findings if f["severity"] == "CRITICAL"), None)
        susp_example = next((f for f in st.session_state.findings if f["severity"] == "SUSPICIOUS"), None)
        with st.expander("See the actual extracted bytes (one CRITICAL + one SUSPICIOUS example)"):
            if crit_example:
                st.write(f"**CRITICAL** \u2014 `{crit_example['layer']}`, matched `{crit_example['matched_signature']}`")
                st.code(crit_example["decoded_preview"])
            if susp_example:
                st.write(f"**SUSPICIOUS** \u2014 `{susp_example['layer']}`, no signature match")
                st.code(susp_example["decoded_preview"])

    if st.session_state.layer_summary:
        st.write(
            "**Peak printable ratio by layer** (top 15 \u2014 the planted layers spike, "
            "everything else is the noise floor)"
        )
        top_layers = dict(
            sorted(st.session_state.layer_summary.items(), key=lambda kv: kv[1], reverse=True)[:15]
        )
        st.bar_chart(pd.Series(top_layers, name="peak printable ratio"))

if st.session_state.reverify is not None:
    st.subheader("Post-Disarm Re-Verification")
    if not st.session_state.reverify:
        n_disarmed = len(st.session_state.findings or [])
        st.success(
            f"Extraction now FAILS across every layer and bit-phase \u2014 all "
            f"{n_disarmed} payload(s) destroyed, model is clean."
        )
    else:
        st.warning(f"{len(st.session_state.reverify)} payload(s) still detectable after disarm \u2014 something's wrong.")
        for f in st.session_state.reverify:
            st.write(f)

if st.session_state.predictions:
    st.subheader("Does the model still work? (prediction agreement)")
    cols = st.columns(len(st.session_state.predictions))
    for col, (stage_name, preds) in zip(cols, st.session_state.predictions.items()):
        with col:
            st.write(f"**{stage_name.capitalize()}**")
            for path, result in preds.items():
                st.write(f"- {os.path.basename(path)}: {result}")