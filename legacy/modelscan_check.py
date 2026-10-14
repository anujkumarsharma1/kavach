"""
Runs the real, independent `modelscan` CLI (Protect AI's pickle-opcode
scanner) against a temp copy of whatever KAVACH is currently looking at, so
the app can show a live "ModelScan says X, KAVACH says Y" contrast on the
same screen instead of a manual alt-tab to a terminal.

Verified against modelscan 0.8.8's actual CLI before writing this:
- `modelscan scan --help` confirms a real `-r/--reporting-format json` flag
  -- used here instead of scraping console text.
- A clean, well-formed .pt file: exit 0, `summary.total_issues == 0`,
  `summary.scanned.total_scanned == 1`.
- A malicious file built with `torch.save(Evil(), ...)` (a `__reduce__`
  pointing at `os.system`): exit 1, `summary.total_issues == 1`, a CRITICAL
  entry in `issues` naming the unsafe operator.
- A malformed/unrecognized pickle (not in torch.save's container format):
  exit 3, `summary.scanned.total_scanned == 0` -- modelscan didn't actually
  scan anything. This is NOT the same as "clean" and is reported as such
  below (`clean=None`), not silently treated as a pass.
"""
import json
import os
import shutil
import subprocess
import tempfile

import torch


def run_modelscan(model_or_state_dict, timeout=30):
    """Returns (clean, summary_dict, raw_output).

    clean is True/False when modelscan actually scanned the file, or None
    when it either isn't installed or couldn't recognize/scan the file at
    all (distinct from a real clean bill of health -- never report an
    unscanned file as "Clean"). torch.save accepts either an nn.Module or
    a plain state_dict natively, so this works for both KAVACH's own demo
    models and an uploaded state_dict (Feature A).
    """
    if shutil.which("modelscan") is None:
        return None, None, "modelscan not installed -- `pip install modelscan` to enable this comparison."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model_or_state_dict, tmp_path)

        result = subprocess.run(
            ["modelscan", "scan", "-p", tmp_path, "-r", "json"],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = result.stdout or result.stderr

        try:
            # modelscan prints an informational banner line to stdout
            # before the JSON blob -- the JSON itself always starts at
            # the first '{'.
            json_start = raw.index("{")
            parsed = json.loads(raw[json_start:])
        except (ValueError, json.JSONDecodeError):
            return None, None, raw

        summary = parsed.get("summary", {})
        total_scanned = summary.get("scanned", {}).get("total_scanned", 0)
        if total_scanned == 0:
            return None, parsed, raw
        clean = summary.get("total_issues", 0) == 0
        return clean, parsed, raw
    except subprocess.TimeoutExpired:
        return None, None, f"modelscan timed out after {timeout}s."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

