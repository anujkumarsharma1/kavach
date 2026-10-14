# KAVACH v2: plan (rebuild the UI, make upload work, deploy on GitHub)

Status: plan only. No code has changed. Written 14 Oct 2026 against `main` @ c016e44.

## 1. What "string injection" means in this repo

KAVACH does not look for pickle code execution. It looks for **byte strings hidden in the least-significant bit (LSB) of float32 weights** (`stego.py`, `scan.py`):

- Each layer's LSB stream is read at all 8 bit phases and packed into bytes.
- A 64-byte window slides in 8-byte steps. It is flagged when 75% or more of its bytes are printable ASCII.
- A flag is `CRITICAL` if it contains a known signature (only EICAR today) and `SUSPICIOUS` otherwise.
- Disarm randomises the LSBs of each flagged layer (`disarm.py`).

Pickle `os`/`eval`/`exec` detection is handed to the external `modelscan` CLI and is only shown as a side-by-side comparison (`modelscan_check.py`).

v2 keeps this scope: **detect, show and disarm strings hidden in weight values.** It adds one limited thing. The safe pickle reader in section 4 has to walk the opcodes anyway, so it reports any non-allowlisted `GLOBAL` import (for example `os.system`) as "file refused, unsafe import X". That replaces the `modelscan` dependency, which no longer installs (see 2.3). Whether it gets more prominence than that is open question Q1.

## 2. Current architecture

| File | Role |
|---|---|
| `app.py` (661 lines) | Streamlit UI. A sidebar picks a model (ResNet18, chest X-ray DenseNet121, or "Upload your own file"). Demo flow: Load → Tamper → Scan → Disarm, plus a prediction comparison and tabs. Upload flow: lines 391-511 |
| `stego.py` | `embed_payload` / `extract_bits`, which assume float32 (`view(np.uint32)`) |
| `scan.py` | `scan_layer` / `scan_model`: vectorised window scan |
| `tamper.py` | `multi_tamper`: plants EICAR and a mystery payload at random layers and offsets |
| `disarm.py` | LSB randomisation |
| `tensor_access.py` | One interface over an `nn.Module` or a flat state_dict |
| `models_zoo.py` | The torchvision and torchxrayvision demo models |
| `modelscan_check.py` | Re-saves the tensors with `torch.save` and shells out to `modelscan` |
| `benchmark.py`, `get_baseline.py` | Offline measurements |

Everything runs server-side in Python and needs torch, so it **cannot run on GitHub Pages as it is**.

### 2.1 Root cause of "upload any model doesn't work" (reproduced on this box)

I ran the real app (Streamlit 1.63, torch 2.14.1 CPU, Python 3.13) under headless Chrome, uploaded four test files and also called `scan_model` directly.

| Upload | Result | Where it breaks |
|---|---|---|
| ResNet18 `state_dict()` saved as .pt, 46.8 MB | Works: 122 tensors scanned | — |
| Whole model saved with `torch.save(model)` | **Refused**: `UnpicklingError: Weights only load failed` | `app.py:409` loads with `weights_only=True`, which by design rejects pickled `nn.Module` objects. That is how most people save a "model" |
| Training checkpoint `{'epoch':1,'state_dict':{...}}` | **False clean**: "Loaded safely — 2 tensor(s)", and 0 weights were actually scanned | `scan.py:137` goes through `tensor_access.iter_named_tensors`, which only walks the top level. `np.asarray(dict)` is a 0-d object, so `scan_layer` skips it silently |
| bf16 state_dict (common for LLMs) | **Crash**: `TypeError: Got unsupported ScalarType BFloat16` | `tensor_access.py:22`, `tensor_like.detach().numpy()` |
| fp16 state_dict | Loads, but the scan is **wrong**. Tamper crashes: `ValueError: Payload needs 544 weight slots…` | `stego.py:35` and `stego.py:20` do `view(np.uint32)` on any dtype, so fp16 is read as pairs of values (the wrong bit is taken as the LSB, and the element count halves) |
| .safetensors, .bin, .onnx, .gguf, .h5, .ckpt | Can't be selected at all | `app.py:401`: `type=["pt", "pth"]` |

There is also an install-time break. `requirements.txt` pins `modelscan==0.8.8`, which only ships for Python ≤3.12. On Python 3.13, `pip install -r requirements.txt` fails with "No matching distribution found for modelscan". The file also lists numpy, streamlit and pillow twice, once pinned and once unpinned. Smaller limits: Streamlit's default 200 MB upload cap, and the whole file plus a `deepcopy` are held in RAM.

**In short:** the uploader only accepts a flat float32 `state_dict` in `.pt`/`.pth`. Whole models are refused at `app.py:409`, nested checkpoints silently scan nothing (`tensor_access.py` iterates only the top level), bf16 crashes at `tensor_access.py:22`, fp16 is misread at `stego.py:35`, and every other format is blocked at `app.py:401`.

## 3. Target architecture and deploy choice

**Recommendation: a fully static app on GitHub Pages, with the scan running in the browser (TypeScript in a Web Worker). No backend.**

Why this option:
1. The scanner is about 150 lines of NumPy: bit extraction, `packbits`, a cumulative-sum window. That ports directly to `Uint8Array`/`Uint32Array`/`Uint16Array`. No torch is needed to *read* weights, because every target format is a documented container around raw little-endian tensor bytes.
2. Model files never leave the user's machine. For a security tool that is the strongest privacy claim it can make, and it removes upload limits and server RAM limits.
3. Zero cost and zero cold starts. A free Railway or HF Spaces container with torch would sleep, has RAM limits (an estimated 512 MB to 1 GB on cheap tiers), and makes a 1 GB upload infeasible.
4. Nothing is unpickled, by construction: there is no Python `pickle` in the browser.

Options I rejected:
- **Pyodide:** it can't load torch, so the parsers would have to be rewritten anyway, plus about 10 MB of startup (estimate). It is only worth it to reuse NumPy code.
- **Pages + Railway FastAPI:** costs money, files leave the device, and the size limits are real. Keep it as a fallback only if a future feature needs torch, such as the prediction comparison.
- **Pages + HF Spaces:** free, but it sleeps and has upload limits. The same objections apply.

What's lost: the live "ResNet18 predicts the same class after tampering" demo, because that needs inference. v2 replaces it with a measured, static statement taken from the existing `baseline_predictions.json` / `benchmark.py` output. Bringing it back would need onnxruntime-web (open question Q3).

```
GitHub Pages (static, built by GitHub Actions from main)
└── index.html + app bundle (Vite + TypeScript, no UI framework or a small one such as Preact)
    ├── ui/          screens (section 5)
    ├── worker.ts    runs off the main thread; receives a File and streams slices with File.slice()
    │   ├── formats/ safetensors · gguf · onnx(protobuf) · torch-zip(.pt/.pth/.bin/.ckpt) · legacy-pickle · h5 · numpy(.npy/.npz)
    │   ├── pickle-walker.ts   opcode reader, never executes anything (section 4.3)
    │   ├── scan.ts            port of scan.py (8 phases, window 64, stride 8, threshold 0.75)
    │   ├── disarm.ts          port of disarm.py; writes a cleaned copy (safetensors output)
    │   └── tamper.ts          demo only: plants EICAR + mystery payload into the user's own file
    └── samples/     small demo files: clean, tampered, unsafe-pickle (section 7)
python/ (kept)       the existing code becomes the reference implementation for parity tests in CI
```

## 4. Upload pipeline

### 4.1 Accepted formats and how each is read

| Extension | How tensors are found | Notes |
|---|---|---|
| `.safetensors` | 8-byte header length, JSON header, then offsets and dtype | Easiest and safest format. Also scan `__metadata__` strings |
| `.pt` `.pth` `.bin` `.ckpt` (torch ≥1.6 zip) | Read the zip central directory, walk `data.pkl` opcodes to map storage key → dtype/shape/offset, then read `data/<key>` raw | Handles nested checkpoints because the walk covers every tensor wherever it sits in the pickle |
| legacy torch pickle (pre-1.6) | Same opcode walk; the storages follow the pickle | Best effort; mark "partial" if not understood |
| `.onnx` | Minimal protobuf reader over `graph.initializer` (`raw_data` or `float_data`); also `external_data` | Report external-data files as "not uploaded, not scanned" |
| `.gguf` | Header, KV metadata (strings scanned), then the tensor info table | Quantised tensors (Q4_K etc.) have no float LSB, so they get "metadata scanned, weights not scannable" |
| `.h5` / `.keras` | Phase 2. HDF5 needs a parser such as h5wasm (WASM, about 1 MB, estimate) | Lazy-loaded only when an .h5 is dropped |
| `.npy` / `.npz` | Header plus raw data | Cheap to add |

Dtype handling (the fix for the fp16/bf16 bugs): fp32 takes bit 0 of each `uint32`, fp16 and bf16 take bit 0 of each `uint16`, and fp64 takes bit 0 of each `uint64`. Integer tensors (for example `num_batches_tracked`) and quantised blocks are listed as "skipped: no float LSB", **never counted as clean**.

### 4.2 Size limits
- A soft warning at 2 GB and a hard limit of 4 GB in v2 (estimate; the browser memory budget varies). Tensors are processed one at a time from `File.slice()`, so the peak RAM is roughly the largest tensor plus about 10% (estimate), not the file size.
- The worker reports progress per tensor and can be cancelled.
- Mobile limit: 500 MB, with a message telling the user to use a laptop for anything bigger.

### 4.3 Safe parse, never unpickle
- The pickle walker reads opcodes and builds a passive tree. It understands `GLOBAL`/`STACK_GLOBAL`, `REDUCE`, `BUILD`, `BINPERSID`, `MARK`, the dict, list and tuple opcodes, and the string and int opcodes. **Nothing is executed or imported.**
- Allowlisted globals (`torch._utils._rebuild_tensor_v2`, `torch.*Storage`, `collections.OrderedDict`, and the same few that torch's `weights_only` loader accepts) are used to map storages.
- Any other global, such as `os.system`, `builtins.eval`, `subprocess.*` or `posix.system`, gets the result "Refused: this file imports X when loaded". The weights are still scanned, since they are just bytes. This is the scope-limited replacement for modelscan (see Q1).
- Hard caps: 50 MB of pickle stream, a stack depth of 10,000 and 1,000,000 opcodes (estimates). Anything beyond them is refused rather than guessed at.
- Every parser bounds-checks offsets against the file size. A malformed file gives "Couldn't read: <reason>", never a crash and never "clean".

### 4.4 How results are shown
Three verdicts, never more:
- **Clean:** "N tensors, M weights scanned, no hidden strings." Skipped tensors are listed and counted.
- **Hidden strings found:** one card per region showing layer, bit range, phase, printable ratio against the noise floor, a decoded preview (escaped, monospace, never rendered as HTML), and CRITICAL/SUSPICIOUS.
- **Couldn't fully scan:** the reason, plus what *was* covered.

Also on the results screen:
- A per-layer chart of peak printable ratio against the 75% line. This is the existing "one layer spikes" chart, redrawn as a plain SVG.
- **Disarm** produces `<name>.kavach-clean.safetensors`. The browser re-scans it before offering the download. Re-encoding to the original .pt format is out of scope for v2.
- **Export report** writes a JSON file of the findings.

## 5. UI screens and flow ("Fable" direction)

Visual system:
- Warm paper background (#F6F1E7 range), ink text (#1F1B16) and one accent: a deep vermilion close to the current peach/orange identity (#C2410C range).
- Heading font Fraunces or Instrument Serif; body font Inter or Söhne-like.
- Large type scale: 56/40/24/18/16.
- Cards: radius 16, soft shadow (0 1px 2px plus 0 8px 24px at low alpha), medium 24 px gaps.
- One pill-shaped primary button per screen.
- Motion: 200 to 300 ms ease-out fades and rises, and a continuous progress line during the scan. No glass, no gradients on text, no neon, no emoji headings.
- Keyboard and screen-reader friendly. Reduced motion is respected.

Screens:
1. **Home:** headline "Find what's hidden inside a model file." One line of explanation, one pill button "Scan a model", and a secondary link "Try a sample".
2. **Drop:** a large drop zone with the accepted formats listed in plain words and "Your file never leaves this device".
3. **Scanning:** file name, a progress line, the current tensor name and a running count. Cancel is available.
4. **Result:** the verdict card, then finding cards, then the layer chart, then the skipped list. Actions: Disarm, Export report, Scan another.
5. **Disarm done:** "Re-scanned: clean" with a download pill.
6. **Try it live:** takes the user's clean file, plants EICAR plus the mystery payload with `tamper.ts`, and runs the scan. This keeps the hackathon "live proof" moment.
7. **How it works:** a short explainer with three illustrated steps (LSB → byte stream → window), the comparison with pickle scanners, and limits. The "not for medical use" note sits here.

The flow is linear (Home → Drop → Scanning → Result → Disarm) with no sidebar. The detection parameters move into "How it works".

## 6. Repo and deploy mechanics
- Keep the Python code in `python/` (moved, unchanged apart from the dtype and nesting fixes used for parity) and add `web/`.
- `.github/workflows/pages.yml`: on push to main, run `npm ci`, the tests and `npm run build`, then `actions/upload-pages-artifact` and `deploy-pages`.
- Pages source is "GitHub Actions". The URL will be `https://anujkumarsharma1.github.io/kavach/` (Vite `base: '/kavach/'`).
- Work happens on a `v2` branch with a preview, and is merged only when Cyber approves.
- Fix `requirements.txt`: drop the duplicate lines and either make `modelscan` optional (a separate `requirements-modelscan.txt`) or pin `python_requires <3.13`.

## 7. Test plan

Test files are generated by a script (`tests/make_samples.py`, run in CI). Nothing malicious is downloaded or committed.
- **Benign:** random-init ResNet18 in fp32, fp16 and bf16, each saved as state_dict .pt, whole-model .pt, nested checkpoint, .safetensors, .onnx (torch.onnx.export of a tiny CNN) and a tiny .gguf (written with the `gguf` Python package). Plus pretrained ResNet18 weights for a real noise floor.
- **String-injected:** the same files after `multi_tamper` with EICAR plus the mystery payload at unaligned offsets (index % 8 ≠ 0), at minimum and maximum lengths, in the first and last tensor and across tensor boundaries.
- **Unsafe pickle (built safely):** a pickle stream written by hand with `GLOBAL 'os' 'system'` + `REDUCE` on the harmless argument string `"echo kavach-test"`. It is only ever *parsed*, never loaded. A test asserts that the walker reports `os.system`, and the CI job has no step that calls `pickle.load`/`torch.load(weights_only=False)` on it.
- **Malformed:** truncated zip, a safetensors header length larger than the file, a bad protobuf, a pickle bomb (deep nesting), an empty file and a 0-byte tensor.

Checks:
- **Parity:** for every sample, `scan.ts` findings (layer, bit_start, bit_end, ratio) equal Python `scan_model` exactly.
- **False positives:** 0 findings on pretrained ResNet18 and DenseNet121 (README reports 0 in 1,120 clean-layer scans at 0.75).
- **Performance:** time a 100 MB and a 1 GB safetensors scan on a laptop in Chrome and Firefox, and record the measured numbers in the README.
- **UI:** Playwright flows for every screen, axe accessibility with no serious violations, and a 375 px mobile layout.

## 8. Milestones (effort figures are estimates for focused work)

| # | Milestone | Effort (estimate) |
|---|---|---|
| M0 | Fix the Python reference: dtype-aware LSB, nested checkpoint walk, bf16, clean up requirements. Add the sample generator and its tests | 0.5 day |
| M1 | `scan.ts` + `disarm.ts` + `tamper.ts` ports, with Python parity tests | 1 day |
| M2 | Parsers: safetensors, torch zip with the pickle walker, ONNX, GGUF, npy. Malformed-file tests | 2 days |
| M3 | New UI: design system and the 7 screens, Web Worker wiring, progress and cancel | 2 days |
| M4 | Disarm to safetensors download, JSON report export, samples on the "Try a sample" screen | 0.5 day |
| M5 | GitHub Actions Pages deploy, README rewrite with measured numbers, Playwright and axe tests | 0.5 day |
| M6 (optional) | .h5 through h5wasm; legacy pre-1.6 pickle storages | 1 day |

Core total: about 6.5 days (M0 to M5), or about 7.5 days with M6. Both are estimates.

## 9. Risks
- **Torch zip/pickle variants** (legacy formats, `_rebuild_parameter`, sparse or quantised tensors): unknown ones show "partial scan" instead of "clean". Test against real Hugging Face `pytorch_model.bin` files.
- **Browser memory** on multi-GB files: mitigated by per-tensor slicing. A single huge tensor (an embedding matrix of 1 GB or more) still needs a chunked scan. Plan to scan in 64 MB chunks with a 72-byte overlap (window plus phase).
- **Threshold false positives** on unusual weights (pruned or zeroed weights, quantised-then-dequantised weights). Mitigation: show the ratio against the noise floor and call it "suspicious", not "malware".
- **Scope creep** into full pickle malware scanning (Q1).
- **Disarm changes weights** by about 1e-7 per value. That is documented; accuracy claims must come from measurements, not assertions.
- **Losing the live prediction demo** (Q3).
- **GGUF quantised weights** can't carry LSB payloads the same way, so v2 says "not scannable" rather than claiming coverage.

## 10. Open questions for Cyber
- **Q1.** Should a dangerous pickle import (`os.system`, `eval`) found while parsing be a headline verdict, or only a "refused" note while v2 stays about strings hidden in weights?
- **Q2.** Is fully in-browser, with no backend and files never leaving the device, acceptable? Or do you want Railway anyway, for example to keep Python as the scanner?
- **Q3.** Should the "model still predicts the same" live demo be kept (it needs onnxruntime-web, about +1 day, estimate), or replaced with the measured static result?
- **Q4.** Should the URL be `anujkumarsharma1.github.io/kavach/`, and can the Streamlit app be retired, or should it be kept running somewhere?
- **Q5.** Is .h5/Keras support needed in v2, or can it wait (M6)?
