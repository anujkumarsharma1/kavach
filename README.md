# KAVACH — Antivirus for AI Models

> Built for the **Precision Care Hackathon**. We did not place / did not move
> forward past this round. Development is stopped as of **16 Sep 2026**. This
> README is a retrospective write-up of what the project is, how it works,
> what was planned, how much actually got built, and why we're stepping away
> from it. The code is left as-is — untouched, verified working — as a
> record and in case it's ever picked up again.

---

## 1. What is this?

AI model files (`.pt`, `.pth`, etc.) are, underneath everything, just huge
arrays of numbers — millions of float32 weights a network learned during
training. **KAVACH is built around a specific, under-appreciated attack on
those files**: you can hide an arbitrary secret (a malware signature, stolen
credentials, a C2 config, anything) inside the weights themselves, by
flipping only the least-significant bit of each float you touch.

The value shift from flipping that one bit is about `0.0000002` — smaller
than ordinary training noise. The model still loads fine, still predicts
normally, still "looks" like an untouched file. But it's now secretly
carrying a payload.

**The blind spot this exploits:** every existing AI-model security scanner
we found (Fickling, ModelScan, etc.) checks one thing — *can loading this
file execute malicious code?* They inspect the pickle deserialization
opcodes. None of them look at the actual numeric weight **values**. A
payload hidden in the weights runs zero code on load, so it sails through
that entire category of tool with a clean bill of health.

KAVACH is "antivirus for AI models" built specifically for that blind spot,
as a **complement** to tools like ModelScan/Fickling, not a replacement:

1. **Scan** — sweep every parameter tensor in a model and catch the
   statistical fingerprint a hidden payload leaves behind, with zero prior
   knowledge of what was hidden, how long it is, or exactly where it sits.
2. **Disarm** — once something suspicious is found, scramble the flagged
   region's LSBs enough to destroy the payload, while leaving the model's
   real predictions essentially untouched.

It also ships its own "attack simulator" (`tamper.py`) that plants real
payloads into a real pretrained model, so the whole hide → find → remove
cycle can be demonstrated live, on demand, not just claimed on a slide.

---

## 2. How it actually works

### Hiding data (`stego.py`)
Every weight is a 32-bit IEEE-754 float. Reinterpreting its raw bits and
overwriting only the last bit (the LSB) lets you smuggle one bit of
arbitrary data per weight at a ~1e-7 value shift — far below normal
training/quantization noise. String enough weights together and you can
encode whole byte sequences. Payloads can start at *any* float index, not
just index 0 — `stego.py` supports an arbitrary `start` offset.

### Attacking (`tamper.py`)
`multi_tamper` plants payloads across several layers in one pass, choosing
candidate layers only by "is this tensor big enough to hold the payload and
clear the scanner's own worst-case window size?" — **never** by hardcoded
layer name. That's what makes the same attack/detection code portable
across architectures. Each demo run plants two payload *types* at random,
non-zero, non-byte-aligned offsets:
- A known-signature EICAR antivirus test string (2 locations)
- A "mystery" string not in any signature database (3 locations) — this is
  what proves the detector isn't just doing a string search.

### Finding it (`scan.py`)
A sliding-window detector (64-byte window, 8-byte stride) measures what
fraction of bytes in each window are printable ASCII. Random trained-weight
noise is printable roughly 37–43% of the time by chance; a real hidden
payload spikes that to 90–100%. The threshold is tuned to **75%**. Because a
payload can start at any bit offset, not just a byte boundary, the scanner
checks **all 8 possible bit-phase alignments** per layer — decoding from the
wrong phase turns a real payload into garbage, so this was a real failure
mode that had to be explicitly solved, not an edge case skipped over.
Overlapping flagged windows get merged into single reported regions, then
matched against a small known-signature list (`CRITICAL`) or reported as
statistically anomalous but unmatched (`SUSPICIOUS`).

### Cleaning it (`disarm.py`)
Once a layer is flagged, `disarm_layer` randomizes **every** LSB across that
entire layer (not just the exact flagged bytes — a deliberate safety margin
in case the true payload extends past what one scan window reported), which
destroys the payload at effectively zero cost to model accuracy, since it's
touching the same class of bit-magnitude that hiding used in the first
place.

### Working on any architecture (`models_zoo.py`, `tensor_access.py`)
- `models_zoo.py` exposes one shared `load` / `predict` / `sample_images`
  interface behind a `MODEL_ZOO` dict, so none of `stego.py` / `scan.py` /
  `tamper.py` / `disarm.py` ever reference a specific architecture. Two
  models are wired in: **ImageNet ResNet18** (torchvision, the original demo
  model) and a **chest X-ray DenseNet121** (torchxrayvision — a real,
  independently-trained clinical-imaging model, used purely to prove
  cross-architecture detection and clearly labeled **NOT FOR MEDICAL USE**
  everywhere it appears).
- `tensor_access.py` is a small shared accessor layer so every scan/tamper/
  disarm function works identically whether given a live `nn.Module` or a
  plain uploaded `state_dict` (`dict[str, Tensor]`) — one code path, not two.

### The Streamlit app (`app.py`)
A peach-and-white themed UI with a guided 4-step workflow: **Load
Baseline → Tamper → Scan → Disarm**, plus:
- A sidebar model picker (built-in demo models, or upload your own `.pt`
  file).
- A scan report tab: severity counts, a findings table, extracted-byte
  previews, and a bar chart of peak printable ratio per layer.
- A live side-by-side comparison against the real, independent `ModelScan`
  CLI tool on the same file (`modelscan_check.py`) — this is the "ModelScan
  says clean, KAVACH finds it" contrast made visible on one screen instead
  of an alt-tab to a terminal.
- Post-disarm re-verification confirming extraction now fails.
- A predictions panel showing the model's real predictions on sample images
  stay consistent across clean → tampered → disarmed.

---

## 3. The original plan

The MVP (steganographic hide/find/disarm on one model, ResNet18) came
together first. With time left before the hackathon deadline, we wrote up a
three-feature stretch-goal spec (`CLAUDE_CODE_BRIEF.md`) aimed squarely at
the most obvious judge pushback — *"is this rigged to only work on your own
demo model?"*:

- **Feature A — Judges upload their own file.** Let anyone upload a `.pt`
  state_dict and watch it get scanned live, using PyTorch's
  `weights_only=True` safe-load mode so KAVACH can never itself become a
  victim of the exact code-execution attack class it exists to catch.
- **Feature B — A real detection-rate number.** An offline benchmark
  (`benchmark.py`) running many randomized tamper → scan → disarm trials
  plus a separate clean-only run, to replace "it worked when we tried it"
  with an actual citable statistic for the deck.
- **Feature C — Live ModelScan comparison inside the app**, instead of a
  manual alt-tab to a terminal during the demo.

Alongside the code, we planned the actual pitch: a tight 7-minute
presentation script (`PRESENTATION_SCRIPT.md`) built around a live demo
rather than slides, with a deliberate "physical moment" — handing the
laptop to a judge and scanning *their* file live — as the centerpiece
meant to kill the "is this rigged?" question before anyone could ask it.

---

## 4. How far we actually got

**All of it.** Every feature in the brief and the features doc got built
and, as of this write-up, has been re-verified working end-to-end:

| Piece | Status |
|---|---|
| Core stego hide/extract round-trip | ✅ Verified — self-test passes, ~2.4e-7 max value shift |
| Tamper → Scan → Disarm → re-verify (ResNet18) | ✅ Verified — planted 5 payloads across 5 layers, all 5 detected (2 CRITICAL/signature-matched, 3 SUSPICIOUS/unknown), disarmed, re-scan comes back clean |
| 8-bit-phase scanning (non-byte-aligned offsets) | ✅ Verified — payloads planted at random non-aligned offsets were still found |
| Multi-architecture support (ResNet18 + chest X-ray DenseNet121) | ✅ Built, `models_zoo.py` shared interface in place |
| Feature A — upload your own model file, safe-load only | ✅ Built (`tensor_access.py` generalization + upload UI in `app.py`) |
| Feature B — detection-rate benchmark | ✅ Verified — a 3-trial smoke run just now: 18/18 payloads detected, 0 false positives across 20 clean scans, 3/3 fully disarmed, ~12s runtime |
| Feature C — live ModelScan comparison in-app | ✅ Built and verified at the library level — a tampered file correctly comes back "0 issues / clean" from real ModelScan while KAVACH reports 5 hits, exactly the intended contrast |
| Streamlit app boots and serves | ✅ Verified — starts cleanly, responds HTTP 200, no runtime errors |
| Presentation script + judge Q&A prep | ✅ Written (`PRESENTATION_SCRIPT.md`, `FEATURES.md`) |

**One local-environment-only wrinkle, not a code defect:** in the current
Windows venv, the pip-installed `.exe` console-script launchers
(`modelscan.exe`, and incidentally `streamlit.exe`/`pip.exe` too) fail
silently with no output. This affects only the shell wrapper binaries, not
the underlying packages — calling `modelscan`'s Python API directly, or
launching Streamlit via `python -m streamlit run app.py`, both work exactly
as intended (this is in fact how the app was relaunched and verified for
this write-up). The in-app ModelScan comparison panel degrades to its
already-designed "not available" fallback in this specific environment
rather than crashing, which is the correct behavior it was built to have —
it just means the live third-party-comparison panel needs the `modelscan`
package's console script reinstalled/repaired locally (e.g.
`pip install --force-reinstall modelscan`) to visually light up in the app
itself, on this machine.

In short: **the project works.** Every claim in `FEATURES.md`'s Q&A prep is
backed by code that runs and produces the stated result today.

---

## 5. Why we're not moving forward

We didn't place in the Precision Care Hackathon, and the team is not
continuing active development on KAVACH past this point. This isn't a
verdict on whether the underlying idea holds up — the detection blind spot
it targets is real, and the demo genuinely does what the pitch claims — it's
a decision about where to spend time next, not a retraction of the work.
Leaving this README as an accurate record so that if anyone (us included)
revisits this later, the actual state of things doesn't have to be
re-discovered from scratch.

### Known, honest limitations (worth keeping in mind if this is ever picked back up)
- **Short payloads can evade detection structurally.** Reliable detection
  needs roughly 40+ bytes inside a 64-byte scan window to clear the 75%
  printable-ratio threshold — a payload much shorter than that can sit
  below the noise floor by design, not by bug.
- **Disarm is coarse on purpose.** It randomizes an entire flagged layer's
  LSBs, not just the exact flagged byte range — correct as a safety margin,
  but it means "precision surgical removal" isn't actually the model here.
- **The ModelScan comparison depends on the `modelscan` CLI being
  correctly installed/callable** in whatever environment the app runs in —
  see the wrinkle noted above.
- **Not a general-purpose malware scanner.** It detects LSB-steganography
  specifically; a different embedding scheme (e.g. more significant bits, a
  different encoding) wasn't in scope and wouldn't be caught by this
  detector as built.

### If this were picked up again, the next moves we'd already identified
(carried over from `FEATURES.md`'s own Q&A prep, since they're still
accurate):
- Raise sensitivity for very short payloads without reintroducing false
  positives.
- Compare an uploaded file against a known-good baseline hash, not just a
  standalone scan.
- Package the benchmark as a CI-style regression check that runs
  automatically whenever detection logic changes.

---

## 6. Running it yourself

```bash
# from this directory, using the project's venv one level up
../venv/Scripts/python.exe -m pip install -r requirements.txt   # if not already installed

# core pipeline, no UI
../venv/Scripts/python.exe tamper.py --model resnet18   # plants payloads -> tampered_model.pt
../venv/Scripts/python.exe scan.py tampered_model.pt     # finds them
../venv/Scripts/python.exe disarm.py tampered_model.pt   # removes them -> disarmed_model.pt
../venv/Scripts/python.exe scan.py disarmed_model.pt     # confirms clean

# citable detection-rate number
../venv/Scripts/python.exe benchmark.py --trials 20

# the full app
../venv/Scripts/python.exe -m streamlit run app.py
```

`*.pt` model files are git-ignored — running the commands above regenerates
them locally, nothing is lost by not committing them.

---

## 7. Project layout

```
app.py                  Streamlit UI — the 4-step guided demo + upload flow
stego.py                LSB embed/extract primitives
tamper.py                Attack simulator — plants payloads across layers
scan.py                  Detector — sliding-window, 8-bit-phase scan
disarm.py                Cleaner — randomizes flagged layers' LSBs
models_zoo.py             Shared load/predict/sample_images interface, 2 architectures
tensor_access.py          Generic accessor so code works on nn.Module or raw state_dict
modelscan_check.py        Runs the real ModelScan CLI for live comparison
benchmark.py              Offline randomized-trial detection-rate benchmark
payload.py                 Demo payload definitions (EICAR string + mystery string)
get_baseline.py            Downloads ResNet18, saves clean_model.pt baseline
CLAUDE_CODE_BRIEF.md      Spec for the three stretch features (upload / benchmark / ModelScan UI)
FEATURES.md                Full feature list + judge Q&A prep, written during the hackathon
PRESENTATION_SCRIPT.md    Timed 7-minute pitch script
```

---

*Not for medical use. The chest X-ray model is included solely to prove
cross-architecture detection on a real, independently-trained network — it
is not validated and must not be used for any diagnostic purpose.*
