import copy
import os

import pandas as pd
import streamlit as st
import torch

from disarm import disarm_layer
from modelscan_check import run_modelscan
from models_zoo import MODEL_ZOO
from payload import MYSTERY_PAYLOAD, NUM_EICAR_LOCATIONS, NUM_MYSTERY_LOCATIONS, PAYLOAD
from scan import STRIDE_BYTES, THRESHOLD, WINDOW_BYTES, scan_model
from tamper import multi_tamper
from tensor_access import get_array, iter_named_tensors, set_tensor

UPLOAD_OPTION = "Upload your own file"

st.set_page_config(page_title="KAVACH", page_icon="\U0001F6E1️", layout="wide")

st.markdown(
    """
<style>
:root {
    --peach-50:  #FFF9F5;
    --peach-100: #FFF1E6;
    --peach-200: #FFE1CC;
    --peach-300: #FFC9A3;
    --peach-500: #F0925A;
    --peach-600: #E8763C;
    --peach-700: #C4592A;
    --ink:       #3A2A20;
    --ink-soft:  #7A6656;
    --white:     #FFFFFF;
    --green:     #2E7D4F;
    --green-bg:  #EAF6EF;
    --red:       #C0392B;
    --red-bg:    #FCEBEA;
    --amber:     #A9660A;
    --amber-bg:  #FCF3D9;
}

html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

/* ---- Hero banner ---- */
.kavach-hero {
    background: linear-gradient(135deg, var(--peach-100) 0%, var(--peach-50) 60%, var(--white) 100%);
    border: 1px solid var(--peach-200);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
}
.kavach-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2.1rem;
    color: var(--peach-700);
    letter-spacing: -0.02em;
}
.kavach-hero p {
    margin: 0;
    color: var(--ink-soft);
    font-size: 1.02rem;
    line-height: 1.5;
}

/* ---- Section labels ---- */
.kavach-section-label {
    display: inline-block;
    background: var(--peach-100);
    color: var(--peach-700);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* ---- Step cards ---- */
.kavach-step {
    background: var(--white);
    border: 1px solid var(--peach-200);
    border-radius: 14px;
    padding: 0.9rem 1rem 0.4rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 3px rgba(232, 118, 60, 0.08);
}
.kavach-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--peach-600);
    color: var(--white);
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.5rem;
}
.kavach-step-title {
    font-weight: 600;
    color: var(--ink);
    font-size: 0.95rem;
}
.kavach-step-desc {
    color: var(--ink-soft);
    font-size: 0.8rem;
    margin: 0.25rem 0 0.7rem 0;
    min-height: 2.4em;
}

/* ---- Buttons ---- */
div.stButton > button {
    border-radius: 10px !important;
    border: 1px solid var(--peach-300) !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"] {
    background: var(--peach-600) !important;
    border-color: var(--peach-600) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--peach-700) !important;
    border-color: var(--peach-700) !important;
}

/* ---- Verdict banners ---- */
.kavach-verdict {
    padding: 1.0rem 1.3rem;
    border-radius: 10px;
    border: 1px solid;
    margin: 0.4rem 0 1.2rem 0;
    font-size: 1.02rem;
    line-height: 1.5;
}
.kavach-verdict.clean {
    background: var(--green-bg);
    border-color: var(--green);
    color: #1E4E31;
}
.kavach-verdict.tampered {
    background: var(--red-bg);
    border-color: var(--red);
    color: #7A231A;
}
.kavach-verdict.warn {
    background: var(--amber-bg);
    border-color: var(--amber);
    color: #6B4A05;
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--peach-50);
    border: 1px solid var(--peach-200);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: var(--ink-soft);
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--peach-700) !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--peach-600) !important;
}

/* ---- Sidebar caption ---- */
section[data-testid="stSidebar"] .kavach-side-note {
    background: var(--peach-100);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    font-size: 0.78rem;
    color: var(--peach-700);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="kavach-hero">
<h1>\U0001F6E1️ KAVACH — Antivirus for AI Models</h1>
<p>Detects, extracts, and disarms steganographic payloads hidden in neural network
weights — across every parameter tensor, at any bit offset, on any architecture,
without needing to know the payload's length in advance.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("How this differs from pickle scanners like Fickling / ModelScan"):
    st.markdown(
        "Tools like **Fickling** (Trail of Bits) and **ModelScan** (Protect AI) catch "
        "*code-execution* attacks — a malicious pickle opcode that runs the moment the "
        "file loads. They inspect the deserialization instructions, not the weight "
        "**values**.\n\n"
        "KAVACH targets a different, complementary surface: data smuggled inside the "
        "numeric values of the weights themselves. A model tampered this way executes "
        "zero unsafe code on load and passes a pickle-opcode scan cleanly — the payload "
        "is invisible to that entire tool category. KAVACH is built for that specific "
        "blind spot, not to replace those tools."
    )


# ---------------------------------------------------------------------------
# Shared rendering helpers -- used by both the built-in demo flow (ResNet18 /
# chest X-ray) and the upload flow, so both paths show results identically
# and there's exactly one place to get this display logic right.
# ---------------------------------------------------------------------------

def render_modelscan_comparison(ms_clean, ms_raw, n_kavach_hits):
    if ms_raw is None:
        return
    st.markdown('<span class="kavach-section-label">Third-Party Comparison</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ms_label = "Not available" if ms_clean is None else ("Clean" if ms_clean else "Flagged")
        st.metric("ModelScan (code-execution check)", ms_label)
    with c2:
        st.metric("KAVACH (weight-value check)", f"{n_kavach_hits} hit(s)")
    if ms_clean is None:
        st.caption(
            "ModelScan couldn't be run or couldn't recognize this file for "
            "comparison -- see raw output below."
        )
    with st.expander("Raw ModelScan output"):
        st.code(ms_raw)
    st.write("")


def render_scan_report(findings, layer_summary, show_chart=True):
    n_layers = len(layer_summary or {})
    n_flagged = len(findings)
    n_critical = sum(1 for f in findings if f["severity"] == "CRITICAL")
    n_suspicious = n_flagged - n_critical
    top_ratio = max([f["printable_ratio"] for f in findings], default=0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Parameter tensors scanned", n_layers)
    m2.metric("CRITICAL (signature matched)", n_critical)
    m3.metric("SUSPICIOUS (unknown, structured)", n_suspicious)
    m4.metric("Highest printable ratio", f"{top_ratio:.0%}" if n_flagged else "—")

    if not findings:
        st.markdown(
            '<div class="kavach-verdict clean">✅ <b>CLEAN</b> — no suspicious '
            "regions found across any layer, any bit-phase.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="kavach-verdict tampered">\U0001F6A8 <b>TAMPERED</b> — '
            f"{n_flagged} hidden region(s) found and extracted.</div>",
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(
            [
                {
                    "Layer": f["layer"],
                    "Severity": f["severity"],
                    "Bits": f"[{f['bit_start']}:{f['bit_end']}]",
                    "Ratio": f"{f['printable_ratio']:.0%}",
                    "Signature": f["matched_signature"] or "unknown — flagged on structure",
                }
                for f in findings
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        crit_example = next((f for f in findings if f["severity"] == "CRITICAL"), None)
        susp_example = next((f for f in findings if f["severity"] == "SUSPICIOUS"), None)
        with st.expander("See the actual extracted bytes (one CRITICAL + one SUSPICIOUS example)"):
            if crit_example:
                st.write(f"**CRITICAL** — `{crit_example['layer']}`, matched `{crit_example['matched_signature']}`")
                st.code(crit_example["decoded_preview"])
            if susp_example:
                st.write(f"**SUSPICIOUS** — `{susp_example['layer']}`, no signature match")
                st.code(susp_example["decoded_preview"])

    if show_chart and layer_summary:
        st.write(
            "**Peak printable ratio by layer** (top 15 — the planted layers spike, "
            "everything else is the noise floor)"
        )
        top_layers = dict(sorted(layer_summary.items(), key=lambda kv: kv[1], reverse=True)[:15])
        st.bar_chart(pd.Series(top_layers, name="peak printable ratio"), color="#E8763C")


def render_reverify(reverify, n_planted):
    if not reverify:
        st.markdown(
            f'<div class="kavach-verdict clean">✅ Extraction now <b>FAILS</b> across '
            f"every layer and bit-phase — all {n_planted} payload(s) destroyed, "
            "model is clean.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="kavach-verdict warn">⚠️ {len(reverify)} '
            "payload(s) still detectable after disarm — something's wrong.</div>",
            unsafe_allow_html=True,
        )
        for f in reverify:
            st.write(f)


def disarm_flagged(state_or_model, findings):
    """Disarm every distinct layer named in `findings`, in place, via the
    generic tensor_access helpers -- works on an nn.Module or a plain
    state_dict identically."""
    flagged_layers = sorted({f["layer"] for f in findings})
    params = dict(iter_named_tensors(state_or_model))
    for layer_name in flagged_layers:
        disarmed_w = disarm_layer(get_array(params[layer_name]))
        set_tensor(state_or_model, layer_name, disarmed_w)


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
    st.session_state.modelscan_clean = None
    st.session_state.modelscan_raw = None
    # -- Upload flow (Feature A) --
    st.session_state.upload_state_dict = None
    st.session_state.upload_error = None
    st.session_state.upload_findings = None
    st.session_state.upload_layer_summary = None
    st.session_state.upload_modelscan_clean = None
    st.session_state.upload_modelscan_raw = None
    st.session_state.upload_tampered_state_dict = None
    st.session_state.upload_tamper_findings = None
    st.session_state.upload_tamper_layer_summary = None
    st.session_state.upload_tamper_modelscan_clean = None
    st.session_state.upload_tamper_modelscan_raw = None
    st.session_state.upload_reverify = None

st.sidebar.markdown('<span class="kavach-section-label">Controls</span>', unsafe_allow_html=True)
if st.sidebar.button("\U0001F504 Reset demo", use_container_width=True):
    st.session_state.clear()
    st.rerun()

something_loaded = st.session_state.clean_model is not None or st.session_state.upload_state_dict is not None

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="kavach-section-label">Demo Model</span>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox(
    "Choose an architecture",
    list(MODEL_ZOO.keys()) + [UPLOAD_OPTION],
    disabled=something_loaded,
    help="Locked once you load something — hit Reset demo to switch.",
    label_visibility="collapsed",
)
if "NOT FOR MEDICAL USE" in model_choice:
    st.sidebar.markdown(
        '<div class="kavach-side-note">⚠️ Used here to prove architecture-agnostic '
        "detection on a real clinical-imaging model. Not a diagnostic tool, not validated "
        "for medical use.</div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="kavach-section-label">Detection Parameters</span>', unsafe_allow_html=True)
st.sidebar.markdown(
    f"- Window: **{WINDOW_BYTES} bytes**\n"
    f"- Stride: **{STRIDE_BYTES} bytes** (overlapping)\n"
    f"- Bit-phase alignments checked: **8**\n"
    f"- Printable-ratio threshold: **{THRESHOLD:.0%}**\n\n"
    f"None of these are told the true payload length, offset, or which "
    f"layer(s) — all of that is recovered, not assumed."
)

# ===========================================================================
# Upload flow (Feature A) -- entirely separate from the 4-button demo
# workflow below, since its shape is different: scan happens immediately on
# load, tampering is an explicit opt-in second step on the judge's own file.
# ===========================================================================
if model_choice == UPLOAD_OPTION:
    st.markdown('<span class="kavach-section-label">Upload</span>', unsafe_allow_html=True)
    st.caption(
        "Upload a **state_dict** — i.e. a file saved with "
        "`torch.save(model.state_dict(), 'file.pt')` — not a full pickled model object. "
        "It's loaded with `torch.load(..., weights_only=True)`, which restricts "
        "unpickling to a safe allowlist (tensors, basic containers) and refuses "
        "anything else outright. If a file can't even be loaded safely, that alone is "
        "a red flag — no unsafe retry is offered."
    )
    uploaded_file = st.file_uploader("Upload a .pt / .pth state_dict", type=["pt", "pth"])

    if (
        uploaded_file is not None
        and st.session_state.upload_state_dict is None
        and st.session_state.upload_error is None
    ):
        try:
            loaded = torch.load(uploaded_file, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"Loaded object is a {type(loaded).__name__}, not a state_dict. "
                    "Upload a state_dict (torch.save(model.state_dict(), ...)), not a "
                    "full pickled model object."
                )
            st.session_state.upload_state_dict = loaded
            with st.spinner("Scanning immediately, before any tampering..."):
                findings, layer_summary = scan_model(loaded)
            st.session_state.upload_findings = findings
            st.session_state.upload_layer_summary = layer_summary
            with st.spinner("Running ModelScan (independent pickle-opcode check) for comparison..."):
                ms_clean, _, ms_raw = run_modelscan(loaded)
            st.session_state.upload_modelscan_clean = ms_clean
            st.session_state.upload_modelscan_raw = ms_raw
        except Exception as e:
            st.session_state.upload_error = f"{type(e).__name__}: {e}"

    if st.session_state.upload_error:
        st.markdown(
            f'<div class="kavach-verdict tampered">\U0001F6AB <b>Refused to load</b> — '
            f"this file could not be safely loaded.</div>",
            unsafe_allow_html=True,
        )
        st.code(st.session_state.upload_error)
        st.caption(
            "This is a safe refusal, not a crash — a genuinely unsafe file (or anything "
            "outside `weights_only=True`'s allowlist) is exactly what should happen here. "
            "\"This file couldn't even be loaded safely\" is itself a useful red flag, "
            "without needing our detector at all."
        )

    elif st.session_state.upload_state_dict is not None:
        n_tensors = len(st.session_state.upload_state_dict)
        st.markdown(
            f'<div class="kavach-verdict clean">✅ Loaded safely — {n_tensors} tensor(s) '
            "in this state_dict.</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Immediate Scan Report (before any tampering)")
        render_modelscan_comparison(
            st.session_state.upload_modelscan_clean,
            st.session_state.upload_modelscan_raw,
            len(st.session_state.upload_findings),
        )
        render_scan_report(st.session_state.upload_findings, st.session_state.upload_layer_summary)

        st.divider()
        st.markdown('<span class="kavach-section-label">Live Proof</span>', unsafe_allow_html=True)
        st.caption(
            "This proves detection on a file neither of us has seen before, live — "
            "not a canned demo model."
        )
        if st.button(
            "Tamper this file and scan again", type="primary",
            disabled=st.session_state.upload_tampered_state_dict is not None,
        ):
            tampered_sd = copy.deepcopy(st.session_state.upload_state_dict)
            multi_tamper(
                tampered_sd,
                [(PAYLOAD, NUM_EICAR_LOCATIONS), (MYSTERY_PAYLOAD, NUM_MYSTERY_LOCATIONS)],
            )
            st.session_state.upload_tampered_state_dict = tampered_sd
            with st.spinner("Scanning the tampered upload..."):
                findings2, layer_summary2 = scan_model(tampered_sd)
            st.session_state.upload_tamper_findings = findings2
            st.session_state.upload_tamper_layer_summary = layer_summary2
            with st.spinner("Running ModelScan on the tampered upload..."):
                ms_clean2, _, ms_raw2 = run_modelscan(tampered_sd)
            st.session_state.upload_tamper_modelscan_clean = ms_clean2
            st.session_state.upload_tamper_modelscan_raw = ms_raw2
            st.rerun()

        if st.session_state.upload_tampered_state_dict is not None:
            st.subheader("Scan Report — After Tampering")
            render_modelscan_comparison(
                st.session_state.upload_tamper_modelscan_clean,
                st.session_state.upload_tamper_modelscan_raw,
                len(st.session_state.upload_tamper_findings),
            )
            render_scan_report(st.session_state.upload_tamper_findings, st.session_state.upload_tamper_layer_summary)

            if st.button(
                "Disarm", type="primary",
                disabled=not st.session_state.upload_tamper_findings or st.session_state.upload_reverify is not None,
            ):
                disarmed_sd = copy.deepcopy(st.session_state.upload_tampered_state_dict)
                disarm_flagged(disarmed_sd, st.session_state.upload_tamper_findings)
                reverify, _ = scan_model(disarmed_sd)
                st.session_state.upload_reverify = reverify
                st.rerun()

            if st.session_state.upload_reverify is not None:
                st.subheader("Post-Disarm Re-Verification")
                render_reverify(st.session_state.upload_reverify, len(st.session_state.upload_tamper_findings))

        st.info(
            "\U0001F4CB Prediction comparison isn't available for an uploaded file — we "
            "don't know its architecture, only its numbers. Detection and disarm don't "
            "need to know either, which is the whole point."
        )

# ===========================================================================
# Built-in demo flow (ResNet18 / chest X-ray) -- unchanged behavior.
# ===========================================================================
else:
    st.markdown('<span class="kavach-section-label">Workflow</span>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    step_meta = [
        ("1", "Load Baseline", "Load the clean, pretrained model for the selected architecture."),
        ("2", "Tamper", "Hide EICAR + a mystery payload across ~5 layers via LSB steganography."),
        ("3", "Scan", "Sweep every layer, every bit-phase, for statistically anomalous regions."),
        ("4", "Disarm", "Randomize the LSBs of every flagged layer to destroy the payloads."),
    ]

    with col1:
        st.markdown(
            f'<div class="kavach-step"><span class="kavach-step-num">{step_meta[0][0]}</span>'
            f'<span class="kavach-step-title">{step_meta[0][1]}</span>'
            f'<div class="kavach-step-desc">{step_meta[0][2]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Load Baseline", use_container_width=True, type="primary"):
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
        st.markdown(
            f'<div class="kavach-step"><span class="kavach-step-num">{step_meta[1][0]}</span>'
            f'<span class="kavach-step-title">{step_meta[1][1]}</span>'
            f'<div class="kavach-step-desc">{step_meta[1][2]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Tamper", use_container_width=True, type="primary",
            disabled=st.session_state.clean_model is None,
        ):
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
        st.markdown(
            f'<div class="kavach-step"><span class="kavach-step-num">{step_meta[2][0]}</span>'
            f'<span class="kavach-step-title">{step_meta[2][1]}</span>'
            f'<div class="kavach-step-desc">{step_meta[2][2]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Scan", use_container_width=True, type="primary",
            disabled=st.session_state.tampered_model is None,
        ):
            with st.spinner("Scanning every layer across 8 bit-phase alignments..."):
                findings, layer_summary = scan_model(st.session_state.tampered_model)
            st.session_state.findings = findings
            st.session_state.layer_summary = layer_summary
            with st.spinner("Running ModelScan (independent pickle-opcode check) for comparison..."):
                ms_clean, _, ms_raw = run_modelscan(st.session_state.tampered_model)
            st.session_state.modelscan_clean = ms_clean
            st.session_state.modelscan_raw = ms_raw

    with col4:
        st.markdown(
            f'<div class="kavach-step"><span class="kavach-step-num">{step_meta[3][0]}</span>'
            f'<span class="kavach-step-title">{step_meta[3][1]}</span>'
            f'<div class="kavach-step-desc">{step_meta[3][2]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Disarm", use_container_width=True, type="primary",
            disabled=not st.session_state.findings,
        ):
            zoo = MODEL_ZOO[st.session_state.model_choice]
            disarmed_model = copy.deepcopy(st.session_state.tampered_model)
            disarm_flagged(disarmed_model, st.session_state.findings)
            st.session_state.disarmed_model = disarmed_model
            reverify, _ = scan_model(disarmed_model)
            st.session_state.reverify = reverify
            images = zoo["sample_images"]()
            if images:
                st.session_state.predictions["disarmed"] = zoo["predict"](
                    disarmed_model, st.session_state.weights_meta, images
                )

    st.write("")

    has_results = (
        st.session_state.findings is not None
        or st.session_state.reverify is not None
        or st.session_state.predictions
    )

    if not has_results:
        st.markdown(
            '<div class="kavach-verdict warn">Run through the workflow above — '
            "results will appear here, organized by stage.</div>",
            unsafe_allow_html=True,
        )
    else:
        tab_labels = []
        if st.session_state.findings is not None:
            tab_labels.append("\U0001F4CA Scan Report")
        if st.session_state.reverify is not None:
            tab_labels.append("\U0001F9F9 Re-Verification")
        if st.session_state.predictions:
            tab_labels.append("\U0001F5BC️ Predictions")

        tabs = st.tabs(tab_labels)
        tab_idx = 0

        if st.session_state.findings is not None:
            with tabs[tab_idx]:
                render_modelscan_comparison(
                    st.session_state.modelscan_clean,
                    st.session_state.modelscan_raw,
                    len(st.session_state.findings),
                )
                render_scan_report(st.session_state.findings, st.session_state.layer_summary)
            tab_idx += 1

        if st.session_state.reverify is not None:
            with tabs[tab_idx]:
                render_reverify(st.session_state.reverify, len(st.session_state.findings or []))
            tab_idx += 1

        if st.session_state.predictions:
            with tabs[tab_idx]:
                st.caption("Does the model still work? Predictions across each stage should stay in agreement.")
                cols = st.columns(len(st.session_state.predictions))
                for col, (stage_name, preds) in zip(cols, st.session_state.predictions.items()):
                    with col:
                        st.markdown(f"**{stage_name.capitalize()}**")
                        for path, result in preds.items():
                            st.write(f"- {os.path.basename(path)}: {result}")
