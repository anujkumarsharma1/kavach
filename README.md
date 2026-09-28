# KAVACH

A scanner that finds data hidden inside the numeric weights of a PyTorch model file, where pickle-based scanners like ModelScan do not look, and then destroys it.

**Status:** working demo. Built for the Precision Care Hackathon, 2026. Development stopped on 16 Sep 2026.

## What it does
Every weight in a model is a float32. Overwriting only its least-significant bit changes the value by about 1e-7, so an attacker can hide any byte string (a malware signature, credentials, a C2 config) across thousands of weights. The model still loads and predicts normally. Tools like ModelScan and Fickling check whether loading the file can execute code, so a payload stored in weight values passes them clean.

KAVACH plants such payloads into a real pretrained model (`tamper.py`), scans every parameter tensor for them without knowing their content, length or position (`scan.py`), and scrambles the flagged layers' LSBs so the payload is gone (`disarm.py`). A Streamlit app runs the full Load → Tamper → Scan → Disarm cycle, puts ModelScan's verdict on the same file next to KAVACH's, and lets you upload your own `.pt` state_dict to scan.

## How it's built
Python, NumPy, PyTorch, torchvision (ImageNet ResNet18), torchxrayvision (chest X-ray DenseNet121, used only as a second architecture, **not for medical use**), Streamlit, and optionally `modelscan` for the side-by-side comparison.

Design decisions:
- **Statistics, not signatures, for detection.** A 64-byte window slides over each layer's LSB stream in 8-byte steps, and a window is flagged when at least 75% of its bytes are printable ASCII. The code comments put trained-weight noise at about 37 to 43% printable. The known-signature list only decides whether a hit is labelled `CRITICAL` (signature matched) or `SUSPICIOUS` (anomalous, unknown). That is how it catches payloads it has never seen.
- **All 8 bit phases are scanned.** A payload can start at any float index. Decoding from the wrong bit offset turns it into non-printable garbage, so each layer is scanned at all 8 alignments. The first version only scanned phase 0 and missed payloads planted at non-aligned offsets.
- **No architecture is named in the detection code.** `tamper.py`, `scan.py` and `disarm.py` never refer to a layer by name. `tensor_access.py` gives the same interface over an `nn.Module` or a raw state_dict, and `models_zoo.py` is the only file that knows which model is loaded. Uploaded files are loaded with `torch.load(..., weights_only=True)` so a malicious pickle cannot execute through the app.
- **Disarm is deliberately coarse.** It randomizes every LSB in a flagged layer, not only the flagged byte range, in case the payload extends past what the window reported. The change is the same size as the embedding itself.

## How to run
Tested on Windows during the hackathon; the commands below are the platform-neutral form.

    git clone https://github.com/anujkumarsharma1/kavach && cd kavach
    python -m venv venv
    source venv/bin/activate            # Windows: venv\Scripts\activate
    pip install -r requirements.txt

    python stego.py                      # hide/extract round-trip self-test (NumPy only)
    python tamper.py --model resnet18    # plants 5 payloads -> tampered_model.pt
    python scan.py tampered_model.pt     # finds them
    python disarm.py tampered_model.pt   # destroys them -> disarmed_model.pt
    python scan.py disarmed_model.pt     # confirms clean

    python benchmark.py --trials 20      # detection-rate and false-positive benchmark
    python -m streamlit run app.py       # full UI

Notes:
- The first run downloads the ResNet18 weights from `download.pytorch.org`. The `chest-xray` option downloads torchxrayvision weights.
- The ResNet18 prediction panel reads images from `sample_images/`, which is git-ignored. Add a few `.jpg` or `.png` files there. Chest X-ray samples are already in `chest_sample_images/`.
- `*.pt` files are git-ignored; the commands above regenerate them.
- The ModelScan panel needs the `modelscan` command on your PATH. Without it, the panel shows "not available" and the rest of the app keeps working.

## What it measured
| Check | Result | Source |
|---|---|---|
| Stego round-trip: max weight value shift | 2.38e-07 (round-trip OK) | `python stego.py`, 10,000 synthetic float32 weights (seed 0), NumPy 2.4.6, run 28 Sep 2026 |
| Demo run, ResNet18: planted payloads detected | 5/5 (2 CRITICAL, 3 SUSPICIOUS); re-scan clean after disarm | `tamper.py` → `scan.py` → `disarm.py`, run 16 Sep 2026 |
| Benchmark smoke run, ResNet18: payloads detected | 18/18 across 3 trials; 3/3 trials clean after disarm; ~12 s total | `benchmark.py --trials 3`, run 16 Sep 2026 |
| Benchmark smoke run, ResNet18: false positives on the clean model | 0 findings | same run; see caveat below |
| ModelScan on a tampered file | 0 issues ("clean") while KAVACH reported 5 findings | `modelscan_check.py`, run 16 Sep 2026 |
| Chest X-ray DenseNet121 detection rate | not yet measured | |

Caveats:
- 3 trials is a smoke test, not a detection rate. The next step is `python benchmark.py --model both --trials 20` with the output committed.
- The clean-model check loads the same pretrained weights on every trial, so its 20 "clean scans" are one deterministic result repeated. It shows 0 false positives on one model, not a false-positive rate. Measuring a real rate needs many different clean models.

## Limitations
- **Short payloads get past it.** A flagged window needs 75% printable bytes, so a payload much shorter than about 40 bytes cannot raise a 64-byte window above the threshold. This follows from the window size, not from a bug.
- **It only detects printable LSB payloads.** A payload that is encrypted or compressed (non-printable bytes), or hidden in higher bits, is not caught.
- **The command-line scripts trust their input.** `scan.py` and `disarm.py` load files with `torch.load(..., weights_only=False)`, which executes the file's pickle code. Only the app's upload path uses `weights_only=True`. Do not run the CLI on a file you do not trust.

## Project layout
    app.py              Streamlit UI: guided 4-step demo and file upload
    stego.py            LSB embed/extract primitives
    tamper.py           Attack simulator: plants payloads across layers
    scan.py             Detector: sliding window over all 8 bit phases
    disarm.py           Randomizes flagged layers' LSBs
    benchmark.py        Randomized tamper/scan/disarm trials and clean-model scans
    models_zoo.py       Shared load/predict/sample-images interface for both models
    tensor_access.py    Same accessor for an nn.Module or a raw state_dict
    modelscan_check.py  Runs the ModelScan CLI for the side-by-side comparison
    payload.py          Demo payloads (EICAR test string and an unknown string)
    get_baseline.py     Saves a clean ResNet18 and baseline predictions
    FEATURES.md         Feature list and Q&A prep written during the hackathon
    CLAUDE_CODE_BRIEF.md  Spec for the upload, benchmark and ModelScan features

*Not for medical use. The chest X-ray model is included only to show that detection works on a second, independently trained architecture. It is not validated for any diagnostic purpose.*
