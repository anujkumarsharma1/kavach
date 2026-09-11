# Kavach — Features Implemented

## What is this project?

AI models (like the ones that recognize images or read X-rays) are really just huge
files full of numbers — millions of tiny decimal weights the model learned during
training. Kavach is about a sneaky attack on those files: you can hide a secret
message (a virus signature, stolen credentials, a backdoor config — anything) inside
those numbers by tweaking the very last, least significant digit of each one. The
change is so tiny (about 0.0000002 in value) that the model's predictions barely
move at all — the model still works completely normally, but it's now secretly
carrying a hidden payload nobody would notice just by using it.

## What problem does it solve?

Today's malware scanners for AI models (tools like Fickling or ModelScan) only check
whether *loading* a model file can run dangerous code — they inspect the file's
"instructions," not the actual numbers inside it. A payload hidden in the weight
values themselves, the way described above, is completely invisible to that entire
category of tool: the file loads safely, runs no code, and passes those scans clean.

**Kavach is "antivirus for AI models"** built specifically for that blind spot:
1. **Scan** — sweep every number in every layer of a model and catch the statistical
   fingerprint a hidden payload leaves behind, even with zero prior knowledge of what
   was hidden, how long it is, or exactly where it sits.
2. **Disarm** — once something suspicious is found, scramble just enough of those
   numbers to destroy the hidden payload for good, while leaving the model's real
   behavior (its actual predictions) essentially untouched.

It also ships with a companion "attack simulator" (`tamper.py`) that plants real
payloads into a real pretrained model, so the whole hide → find → remove cycle can be
demonstrated live, not just claimed.

---

## Core steganography (`stego.py`)
- Hide/read arbitrary bytes in the least-significant bit of every float32 weight
  (~1e-7 value shift — invisible to model accuracy).
- `start` offset support: payloads can be embedded/extracted starting at *any*
  float-index, not just index 0.
- `get_param_by_name` walks a dotted `named_parameters()` name down to the actual
  tensor, so other files can address any layer generically.

## Tampering (`tamper.py`)
- `multi_tamper` plants payloads across multiple layers in one pass, chosen only by
  "is this layer big enough?" — never by hardcoded layer name, so it works unmodified
  on any architecture.
- Two payload types per run: a known-signature EICAR test string (2 locations) and an
  "unknown" mystery payload not in the signature list (3 locations) — proves both
  signature matching and pure statistical/anomaly detection.
- Random, non-zero, non-byte-aligned offsets each run — proves detection doesn't
  depend on a fixed embed position.
- Candidate layers are also required to clear the scanner's own worst-case bit-phase
  window size, so nothing gets planted somewhere too small for the scanner to ever see.
- CLI: `python tamper.py --model resnet18|chest-xray` — saves the whole tampered model
  object to `tampered_model.pt` (architecture travels with the file, no separate flag
  needed downstream).

## Scanning (`scan.py`)
- Sliding-window printable-byte-ratio detector — no foreknowledge of payload length or
  location required; both are recovered, not assumed.
- Checks all 8 possible bit-phase alignments per layer, so a payload starting at any
  float-index is found, not just ones that land on a byte boundary.
- Merges overlapping flagged windows into single reported regions.
- Matches decoded bytes against a known-signatures list (`CRITICAL`) or flags
  statistically anomalous but unmatched regions (`SUSPICIOUS`).
- CLI: `python scan.py <model.pt>` — works on whatever architecture the file contains.

## Disarming (`disarm.py`)
- Randomizes every LSB in each flagged layer (whole layer, not just the exact flagged
  bytes), destroying any payload at negligible cost to accuracy.
- CLI: `python disarm.py [model.pt]` — defaults to `tampered_model.pt`, accepts any
  file path, architecture-agnostic.

## Multi-architecture support (`models_zoo.py`)
- One shared interface (`load` / `predict` / `sample_images`) behind a `MODEL_ZOO`
  dict, so `stego.py`/`scan.py`/`disarm.py`/`tamper.py` never reference a specific
  architecture.
- **ImageNet ResNet18** (torchvision) — the original demo model.
- **Chest X-ray DenseNet121** (torchxrayvision) — a real, independently-trained
  clinical-imaging model, used to prove architecture-agnostic detection. Clearly
  labeled NOT FOR MEDICAL USE.

## Works on any model file, not just the demo models (`tensor_access.py`)
- A shared accessor layer so scanning/tampering/disarming work identically whether
  given a real loaded model or just a plain uploaded weights file — no separate code
  path needed for "our demo model" vs. "someone else's file."

## Upload your own model file (in `app.py`)
- Anyone can upload their own `.pt` weights file and have it scanned live — proving
  detection isn't rigged to work only on Kavach's own demo models.
- Loaded only through PyTorch's safest possible mode (`weights_only=True`), which
  refuses to load anything that isn't plain numeric data — a genuinely malicious file
  gets rejected outright, before it ever gets a chance to run.
- Scans immediately on upload (before any tampering), then offers a "tamper this file
  and scan again" button to prove live detection on a file no one has seen before.

## Live comparison against a real third-party scanner (`modelscan_check.py`)
- Every scan also runs the independent, real `ModelScan` tool in the background and
  shows both results side by side: "ModelScan: Clean" vs. "Kavach: N hit(s) found" —
  a concrete, live demonstration of the blind spot described above.

## Detection-rate benchmark (`benchmark.py`)
- An offline script that repeats the full hide → find → remove cycle 20 times with
  randomized payload types, lengths, and locations, plus 20 separate clean-model
  scans to check for false alarms.
- Produces one citable stat: **100% of planted payloads detected (176/176), 0 false
  positives, 20/20 fully disarmed** — verified on both the image model and the X-ray
  model.

## Streamlit app (`app.py`)
- Peach + white themed UI: gradient hero banner, numbered step cards, styled metric
  tiles, tabbed results (Scan Report / Re-Verification / Predictions).
- Sidebar model picker: built-in demo models, or upload your own file (locks once
  something is loaded; Reset demo to switch).
- 4-step guided workflow: Load Baseline → Tamper → Scan → Disarm.
- Scan report with severity counts, a findings table, extracted-byte previews, a bar
  chart of peak printable ratio per layer, and the live ModelScan comparison.
- Post-disarm re-verification confirming extraction now fails.
- Prediction-agreement panel showing the model's real image/X-ray predictions stay
  consistent across clean → tampered → disarmed stages.

## Supporting tooling
- `get_baseline.py` — standalone script to download ResNet18 and save a local
  `clean_model.pt` baseline with reference predictions.
- Self-tests baked into `stego.py`, `tamper.py`, and `models_zoo.py --check-xray` for
  verifying each stage in isolation before wiring it into the live demo.

---

## Judge Q&A Prep

Format: **7 minutes to present, 5 minutes of Q&A.** That's tight — in the pitch,
lead with the blind spot (existing scanners check code execution, not weight
values) and the live demo (tamper → scan → disarm, ideally on an uploaded file).
Save depth for the Q&A below; keep each answer here to what you can say in ~20-30
seconds, then offer to go deeper if they want more.

**1. What problem does this actually solve — why do we need this on top of
existing model-security tools?**
Tools like Fickling and ModelScan check whether *loading* a model file can execute
malicious code — they inspect the pickle deserialization opcodes, not the actual
numeric weight values. A payload hidden in the least-significant bits of the
weights themselves runs zero code and passes those scans cleanly — that entire
tool category simply can't see it. Kavach is built for exactly that blind spot,
as a complement to those tools, not a replacement.

**2. How does hiding data in weights actually work?**
Every weight is a 32-bit float. We only ever touch the very last bit (the LSB) of
each number — flipping it shifts the value by about 0.0000002, which is smaller
than normal training noise. Do that across enough weights and you can smuggle
arbitrary bytes through a model file that still trains and predicts completely
normally.

**3. How does the scanner find a payload without knowing its length, content, or
location in advance?**
It slides a fixed-size window (64 bytes) across every layer's extracted bits and
measures what fraction of bytes in that window are printable ASCII. Normal weight
noise is printable about 37-43% of the time by chance; a real hidden payload spikes
that to 90-100%. No foreknowledge of the payload is needed — the anomaly is purely
statistical.

**4. What about a payload that doesn't start at a "clean" byte boundary?**
That's a real failure mode we specifically solved: if you only ever decode starting
from bit 0, a payload that starts at, say, bit index 3 comes out completely
garbled. We scan all 8 possible bit-phase alignments per layer, so a payload is
found no matter which of the 8 possible offsets it starts at.

**5. How do you know it doesn't produce false positives on normal, untampered
models?**
We tuned the detection threshold (75% printable-ratio) against real full-model
scans until false positives hit zero, and our 20-trial benchmark script confirms
it: 0 false positives across 20 completely clean model scans, on two different
architectures.

**6. What's your actual detection rate — do you have a number, or just a demo
that happened to work?**
100% — 176 out of 176 planted payloads detected across 20 randomized trials
(varying payload count, length, and location each run), with 0 false positives and
20/20 fully disarmed afterward. We ran this on both ResNet18 and a completely
different architecture (DenseNet121) with identical results.

**7. Does disarming break the model?**
No — disarming randomizes the same class of bit (the LSB) that hiding uses, so the
value shift is the same tiny ~0.0000002 magnitude. We compare the model's actual
predictions on real images before and after disarming specifically to show they
stay in agreement.

**8. Is disarm precise — does it only touch the exact hidden bytes?**
No, and that's deliberate: it randomizes every LSB in the whole flagged layer, not
just the exact bytes that were found. That's a safety margin — it guarantees the
payload is destroyed even if it extends slightly past what the scan's window
happened to report, at no extra cost to model accuracy.

**9. Why prove this on two different architectures (ResNet18 and a chest X-ray
model)?**
To prove the detector is architecture-agnostic, not hardcoded to one model's layer
names or shapes. `tamper.py` and `scan.py` never reference a specific layer by
name — they just ask "is this tensor big enough?" — so the identical code runs
unmodified on a completely unrelated, independently-trained clinical-imaging model.

**10. Wait — are you making a medical claim with the X-ray model?**
No, explicitly not. It's labeled "NOT FOR MEDICAL USE" everywhere it appears in the
app and code. It's used purely as a real, independently-trained architecture to
prove cross-architecture detection — not as a diagnostic tool.

**11. Can a judge trust this isn't rigged to only work on your own demo files?**
That's exactly what the upload feature is for — anyone can upload their own
`.pt` weights file, have it scanned live (should come back clean), then click
"tamper this file and scan again" to watch detection happen on a file neither of
us has seen before, in real time.

**12. Uploading arbitrary files sounds risky — how do you avoid getting owned by
the exact attack you're trying to catch?**
We only load uploads with PyTorch's `weights_only=True` mode, which restricts
unpickling to a safe allowlist (tensors and basic containers) and refuses to load
anything else — including a genuinely malicious file. We deliberately never offer
an "unsafe mode" fallback. If a file can't load safely, that refusal is itself
shown as a useful result.

**13. Is there a size or length limit on what can be hidden — could a short
message evade detection?**
Yes, honestly — detection needs roughly 40+ bytes to reliably exceed the printable-
ratio threshold inside a 64-byte scan window; something much shorter than that
(a handful of bytes) can structurally evade this specific detector, the same way a
whisper below a microphone's noise floor wouldn't register. That's a known,
documented limit of the current window size, not something we're hiding.

**14. How does this compare to just re-encoding or re-saving the model weights
to strip anything hidden?**
That would also work as a blunt fix, but it doesn't tell you whether anything was
there in the first place, and re-saving everything is wasteful compared to
disarming only the handful of layers actually flagged. Kavach's value is in the
detection and localization — knowing exactly what was hidden and where — not just
the cleanup step.

**15. What's next / what would you build with more time?**
A few directions: raising detection sensitivity for very short payloads without
reintroducing false positives, extending the upload flow to compare an uploaded
file against a known-good baseline hash, and packaging the benchmark as a
CI-style regression check that runs automatically whenever the detector logic
changes.
