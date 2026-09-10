# KAVACH — Three Stretch Features: Implementation Brief

This is a spec for a coding agent (Claude Code) to implement against the
existing KAVACH codebase: `payload.py`, `stego.py`, `tamper.py`, `scan.py`,
`disarm.py`, `models_zoo.py`, `app.py`. Everything below is additive —
none of it should change the existing ResNet18 / chest-X-ray demo flow,
which is already built and tested.

**Test each feature in isolation before moving to the next one.** They
touch different parts of the codebase but Feature A changes a shared
foundation the others don't need — build it first.

---

## Feature A: Judges upload their own `.pt` file

### The problem this solves
Right now KAVACH tampers its own demo model, then finds its own
tampering. A judge can reasonably ask "is this rigged to work on your
model specifically?" Letting them upload any file of their own and
watching it get scanned live removes that doubt.

### The one non-negotiable safety rule
**Never load an uploaded file with `torch.load(path, weights_only=False)`
or plain `torch.load(path)` on older PyTorch defaults.** This tool exists
because loading untrusted model files can execute arbitrary code — using
an unsafe load on a judge's upload would be KAVACH getting owned by the
exact attack class it's built to catch. That's not a hypothetical risk,
it's a plot hole in the pitch if it happens.

Use `torch.load(uploaded_file, map_location="cpu", weights_only=True)`.
This restricts unpickling to a safe allowlist (tensors, basic
containers) and will simply raise an exception on anything it can't
safely load — including a genuinely malicious file, which is itself a
useful result to show ("this file couldn't even be loaded safely — that
alone is a red flag we didn't need our detector for").

If `weights_only=True` load fails: show the error, stop. Do not retry
with `weights_only=False`. Do not add an "advanced/unsafe mode" toggle.

### What kind of file this actually accepts
`weights_only=True` will successfully load a plain `state_dict()` —
i.e. whatever `torch.save(model.state_dict(), "file.pt")` produces: an
`OrderedDict` mapping parameter names to tensors. It will generally
**not** load a full pickled model object (`torch.save(model, ...)`)
unless the uploader's custom classes happen to be on the safe allowlist,
which most won't be. That's fine — tell the user in the UI: "Upload a
state_dict (`torch.save(model.state_dict(), ...)`), not a full pickled
model object." This is a real, explainable constraint, not a bug to
work around.

### Core change: make scan/tamper/disarm accept a raw state_dict, not just an `nn.Module`

Right now `scan_model`, `multi_tamper`, and the disarm logic in `app.py`
all call `model.named_parameters()` and mutate via `.data =`. A loaded
`state_dict` is a plain `dict[str, Tensor]` — no `.named_parameters()`,
and "mutating" means replacing the dict entry, not setting `.data`.

Add a new small file, `tensor_access.py`, with three functions, and use
them everywhere `scan.py`, `tamper.py`, and `app.py`'s disarm step
currently touch parameters directly:

```python
import numpy as np
import torch

def iter_named_tensors(model_or_state_dict):
    """(name, tensor) pairs for either an nn.Module or a plain
    name->tensor mapping (e.g. an uploaded state_dict)."""
    if hasattr(model_or_state_dict, "named_parameters"):
        yield from model_or_state_dict.named_parameters()
    else:
        yield from model_or_state_dict.items()

def get_array(tensor_like):
    if hasattr(tensor_like, "detach"):
        return tensor_like.detach().numpy()
    return np.asarray(tensor_like)

def set_tensor(container, name, new_array):
    new_tensor = torch.from_numpy(new_array.copy())
    if hasattr(container, "named_parameters"):
        dict(container.named_parameters())[name].data = new_tensor
    else:
        container[name] = new_tensor
```

Update `scan_model` (in `scan.py`) to iterate via `iter_named_tensors`
and read via `get_array` instead of calling `.named_parameters()` /
`.detach().numpy()` directly. Update `multi_tamper` (in `tamper.py`)
the same way, writing back via `set_tensor` instead of `p.data = ...`.
Update `app.py`'s Disarm button handler the same way.

**This is a pure generalization — run the full existing test suite
(the ResNet18 and chest-X-ray flows) after this change and confirm
nothing broke before writing any new UI.**

### UI changes (`app.py`)

Add a third option to the model picker: `"Upload your own file"`. When
selected:
- Replace "Load Baseline" with `st.file_uploader("Upload a .pt state_dict", type=["pt", "pth"])`.
- On successful safe load, store the raw state_dict in session state
  (there is no "model" object here, just tensors — that's fine, every
  downstream function now accepts that).
- **Scan immediately, before any tampering**, and show the result. Most
  uploaded files should come back CLEAN — that's the point: proving no
  false positives on real files judges actually trust.
- Then offer a clearly-labeled second action: "Tamper this file and
  scan again" — runs `multi_tamper` on their actual uploaded tensors,
  rescans, shows the hits. This is the moment that proves
  architecture-agnostic detection on a file neither of you has seen
  before, live.
- Disarm works the same as today, using the generalized `tensor_access`
  helpers.
- **Skip the prediction-agreement panel for this mode** and say why:
  "Prediction comparison isn't available for an uploaded file — we
  don't know its architecture, only its numbers. Detection and disarm
  don't need to know either, which is the whole point."

### Acceptance checklist for this feature
- [ ] A plain ResNet18 `state_dict()` saved with `torch.save(model.state_dict(), "x.pt")`, uploaded fresh, scans CLEAN
- [ ] The same file, tampered via the new "tamper this file" button, is fully detected
- [ ] A deliberately malicious pickle (e.g. one Fickling's own test fixtures, or a simple `__reduce__` PoC like the one used to test ModelScan) fails to load with a clear error, and nothing executes
- [ ] Existing ResNet18 / chest-X-ray demo buttons still work exactly as before

---

## Feature B: Detection-rate benchmark

### The goal
Replace "it worked when we tried it" with a real number: "detected
100% of planted payloads across N independent trials, 0 false
positives." Run this **offline, before the presentation**, to get a
citable statistic for the deck — it does not need to run live in front
of judges (20 full trials will take real wall-clock time; don't gamble
dead air on it).

### Design
New standalone script, `benchmark.py`. Reuses `multi_tamper`,
`scan_model`, `disarm_layer` completely unchanged — this is genuinely
straightforward, no new detection logic.

The one thing worth doing properly: don't reuse the exact same two
payloads every trial. Define a small pool of different lengths/content
so the reported number actually supports "works across payload types,"
not just "works for EICAR specifically":

```python
PAYLOAD_POOL = [
    PAYLOAD,                                    # 69 bytes, known signature
    MYSTERY_PAYLOAD,                            # ~66 bytes, unknown signature
    b"SHORT-TAG-42",                            # ~12 bytes -- short
    b"X" * 40 + b"-MID-LENGTH-UNKNOWN-MARKER",   # ~65 bytes -- different content
    b"Y" * 150 + b"-LONG-PAYLOAD-END-MARKER",    # ~170 bytes -- long
]
```

Each trial: pick a random subset/mix from the pool (varying count and
which payloads), a fresh `rng_seed`, run
`multi_tamper(model, [(payload, count), ...], rng_seed=seed)`, scan,
check every planted `(layer, offset, payload)` was actually found (not
just that *some* findings exist — verify count and, for known
signatures, that the match is correct), disarm, rescan and confirm
zero findings remain. Record pass/fail and the raw counts per trial.

```python
def run_benchmark(model_loader, num_trials=20, base_seed=0):
    results = []
    for trial in range(num_trials):
        seed = base_seed + trial
        rng = np.random.default_rng(seed)
        plan = [(p, int(rng.integers(2, 6))) for p in rng.choice(PAYLOAD_POOL, size=rng.integers(2, 4), replace=False)]
        model, _ = model_loader()
        locations = multi_tamper(model, plan, rng_seed=seed)
        findings, _ = scan_model(model)
        detected_ok = len(findings) == len(locations)  # tighten this: also match layer names
        flagged = sorted({f["layer"] for f in findings})
        for name in flagged:
            # disarm using tensor_access helpers
            ...
        reverify, _ = scan_model(model)
        clean_after = len(reverify) == 0
        results.append({"trial": trial, "planted": len(locations), "found": len(findings), "clean_after_disarm": clean_after})
    return results
```

Print a summary at the end: total planted vs. total found across all
trials, trials with a perfect match, trials fully clean after disarm.
Also run a **separate clean-only loop** (no tampering at all, just load
+ scan) across a similar number of trials to report a false-positive
rate alongside the detection rate — both numbers matter for the claim.

### Acceptance checklist
- [ ] Script runs against `models_zoo.load_resnet18` end to end
- [ ] Prints a final line usable verbatim on a slide, e.g. `"Detected
      100% of planted payloads across 20 trials (312/312), 0 false
      positives across 20 clean-model scans, 20/20 fully disarmed."`
- [ ] If any trial fails, the script says exactly which trial and why
      (don't silently average over a real miss)
- [ ] (Optional, only if the above is solid and time remains) also run
      it against `models_zoo.load_chest_xray` for a second citable
      cross-architecture number

---

## Feature C: Live ModelScan comparison inside the app

### The goal
Right now the "ModelScan says clean, KAVACH finds it" contrast is a
manual step (alt-tab to a terminal). Wire it into the Scan button so
both results appear on the same screen.

### Before writing code: verify ModelScan's actual CLI output yourself
Run `modelscan --help` and a real `modelscan -p <file>` in this
environment first. Check specifically whether there's a machine-readable
output flag (JSON or similar) — if there is, use it instead of scraping
text output, it'll be far less fragile. Don't assume a flag name without
checking; I have not verified this specific flag myself.

### Design sketch
```python
import os
import shutil
import subprocess
import tempfile

def run_modelscan(model_or_state_dict):
    if shutil.which("modelscan") is None:
        return None, "modelscan not installed -- `pip install modelscan` to enable this comparison."
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(model_or_state_dict, tmp.name)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["modelscan", "-p", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        raw = result.stdout
    finally:
        os.unlink(tmp_path)
    clean = "No issues found" in raw   # replace with JSON parsing if a flag exists -- see above
    return clean, raw
```

Call this from the Scan button handler, on the *same* tampered
object/state_dict that just went through `scan_model`. Display side by
side:

```python
c1, c2 = st.columns(2)
with c1:
    st.metric("ModelScan (code-execution check)", "Clean" if ms_clean else "Flagged")
with c2:
    st.metric("KAVACH (weight-value check)", f"{n_flagged} hit(s)")
with st.expander("Raw ModelScan output"):
    st.code(ms_raw_output)
```

Showing the raw CLI output in the expander matters for credibility —
it's visibly a real third-party tool's real output, not something
built to agree with you.

### Edge case to handle explicitly
`torch.save(model_or_state_dict, tmp.name)` needs to work whether given
an `nn.Module` (our own demo models) or a plain state_dict (Feature A's
upload path) — `torch.save` accepts either natively, no change needed
here, but test both paths.

### Acceptance checklist
- [ ] Works against the ResNet18 tampered demo model
- [ ] Works against an uploaded-and-tampered file (Feature A), if built
- [ ] Shows a clear, honest message (not a crash) if `modelscan` isn't
      installed in the demo environment
- [ ] The temp file is always cleaned up, including if the subprocess
      call raises

---

## Suggested build order

1. `tensor_access.py` + generalize `scan_model` / `multi_tamper` /
   `app.py` disarm to use it. **Re-run the existing ResNet18 and
   chest-X-ray flows and confirm they still pass before continuing.**
2. Feature C (ModelScan comparison) — independent, no dependency on
   Feature A, gives a working UI improvement fastest.
3. Feature A (upload) — depends on step 1.
4. Feature B (benchmark) — fully independent, can be done any time,
   including in parallel with the others. Run it once you're happy
   with the numbers and keep the output for the deck.

## Guardrails — do not touch

- `stego.py`'s bit-level embed/extract logic, `scan.py`'s
  `WINDOW_BYTES` / `STRIDE_BYTES` / `THRESHOLD` constants, and the
  8-bit-phase scan loop are already tested and tuned (see the false-
  positive testing that set `THRESHOLD = 0.75`). Don't adjust these
  while building any of the above.
- Don't change the existing ResNet18 / chest-X-ray button flow or
  remove the "NOT FOR MEDICAL USE" labeling.
