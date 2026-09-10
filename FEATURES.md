# Kavach — Features Implemented

"Antivirus for AI models" — detects, extracts, and disarms steganographic payloads
hidden in the LSBs of a neural network's float32 weights.

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

## Streamlit app (`app.py`)
- Peach + white themed UI: gradient hero banner, numbered step cards, styled metric
  tiles, tabbed results (Scan Report / Re-Verification / Predictions).
- Sidebar model picker (locks once a baseline is loaded; Reset demo to switch).
- 4-step guided workflow: Load Baseline → Tamper → Scan → Disarm.
- Scan report with severity counts, a findings table, extracted-byte previews, and a
  bar chart of peak printable ratio per layer (planted layers spike above the noise
  floor).
- Post-disarm re-verification confirming extraction now fails.
- Prediction-agreement panel showing the model's real image/X-ray predictions stay
  consistent across clean → tampered → disarmed stages.

## Supporting tooling
- `get_baseline.py` — standalone script to download ResNet18 and save a local
  `clean_model.pt` baseline with reference predictions.
- Self-tests baked into `stego.py`, `tamper.py`, and `models_zoo.py --check-xray` for
  verifying each stage in isolation before wiring it into the live demo.
