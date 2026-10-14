import "@fontsource/fraunces/500.css";
import "@fontsource/fraunces/600.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "./styles.css";
import type { Report } from "./core/model";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const drop = $("drop"), progress = $("progress"), result = $("result");
const input = $<HTMLInputElement>("file");
const bar = $("bar");

let worker: Worker | null = null;

let fileName = "";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const fmtBytes = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(0)} KB`);
const fmtInt = (n: number) => n.toLocaleString("en-US");
const decode = (hex: string) => hex.match(/../g)!.map((h) => { const c = parseInt(h, 16); return c >= 32 && c <= 126 ? String.fromCharCode(c) : "·"; }).join("");

function show(section: "drop" | "progress" | "result") {
  drop.hidden = section !== "drop";
  progress.hidden = section !== "progress";
  result.hidden = section !== "result";
  const target = section === "drop" ? drop : section === "progress" ? progress : result;
  target.classList.remove("enter");
  void target.offsetWidth;
  target.classList.add("enter");
  if (section !== "drop") target.scrollIntoView({ block: "start", behavior: "smooth" });
}

function newWorker() {
  worker?.terminate();
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "progress") {
      const pct = m.total ? Math.round((m.done / m.total) * 100) : 0;
      (bar.firstElementChild as HTMLElement).style.width = `${pct}%`;
      bar.setAttribute("aria-valuenow", String(pct));
      $("progress-detail").textContent = m.current ? `${m.done + 1} of ${m.total} tensors · ${m.current}` : `${m.total} tensors`;
    } else if (m.type === "result") {

      renderResult(m.report);
    } else if (m.type === "disarmed") {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(m.blob);
      a.download = fileName.replace(/\.(pt|pth|safetensors)$/i, "") + ".kavach-clean.safetensors";
      a.click();
      const note = $("disarm-note");
      note.textContent = m.remaining === 0
        ? "Re-scanned the cleaned file: no hidden strings. Download started."
        : `Re-scan still finds ${m.remaining} region(s). The file was not cleaned.`;
      ($("disarm") as HTMLButtonElement).disabled = false;
    } else if (m.type === "error") {
      renderError(m.message);
    }
  };
}

function scan(file: File) {
  fileName = file.name;
  newWorker();
  $("progress-file").textContent = `${file.name} · ${fmtBytes(file.size)}`;
  $("progress-detail").textContent = "Reading the file layout";
  (bar.firstElementChild as HTMLElement).style.width = "0%";
  show("progress");
  worker!.postMessage({ type: "scan", file });
}

function renderError(message: string) {
  result.innerHTML = `
    <div class="card verdict v-error">
      <p class="eyebrow">Couldn't scan this file</p>
      <h2>Not readable as a PyTorch or safetensors model</h2>
      <p class="mono small">${esc(message)}</p>
      <p class="muted">Nothing in the file was run. A file Kavach can't read is not a clean file.</p>
      <div class="actions"><button class="pill" id="again">Scan another file</button></div>
    </div>`;
  $("again").onclick = reset;
  show("result");
}

function chart(r: Report) {
  const entries = Object.entries(r.peaks);
  if (!entries.length) return "";
  const flagged = new Set(r.findings.map((f) => f.layer));
  const w = 640, h = 160, pad = 24;
  const bw = Math.max(1, (w - pad) / entries.length);
  const y = (v: number) => h - pad - v * (h - 2 * pad);
  const bars = entries.map(([name, v], i) =>
    `<rect x="${pad + i * bw}" y="${y(v)}" width="${Math.max(1, bw - 1)}" height="${h - pad - y(v)}" class="${flagged.has(name) ? "hot" : ""}"><title>${esc(name)}: ${(v * 100).toFixed(1)}%</title></rect>`).join("");
  return `
    <figure class="card chart">
      <figcaption><h3>Most text-like window in each tensor</h3><p class="muted small">Random noise sits near 37%. Bars above the 75% line hold hidden text.</p></figcaption>
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Peak printable ratio per tensor">
        <line x1="${pad}" x2="${w}" y1="${y(0.75)}" y2="${y(0.75)}" class="limit" />
        <text x="${pad}" y="${y(0.75) - 6}" class="axis">75%</text>
        ${bars}
      </svg>
    </figure>`;
}

function renderResult(r: Report) {
  const kind = r.verdict === "Clean" ? "v-clean" : r.verdict === "Dangerous" ? "v-danger" : "v-warn";
  const headline = r.verdict === "Clean"
    ? "No hidden strings found"
    : r.verdict === "Dangerous"
      ? "Dangerous: this file runs code when loaded"
      : `${r.findings.length} hidden string${r.findings.length === 1 ? "" : "s"} found`;
  const layers = [...new Set(r.findings.map((f) => f.layer))];
  const findings = r.findings.map((f) => `
    <article class="card finding">
      <div class="finding-head"><span class="tag ${f.severity === "CRITICAL" ? "tag-crit" : "tag-sus"}">${f.severity === "CRITICAL" ? "Known signature" : "Unknown text"}</span><h3 class="mono">${esc(f.layer)}</h3></div>
      <pre class="decoded">${esc(decode(f.decoded_hex))}</pre>
      <dl class="meta">
        <div><dt>Bits</dt><dd>${fmtInt(f.bit_start)} to ${fmtInt(f.bit_end)}</dd></div>
        <div><dt>Alignment</dt><dd>${f.bit_phase} of 8</dd></div>
        <div><dt>Printable</dt><dd>${(f.printable_ratio * 100).toFixed(1)}%</dd></div>
        ${f.matched_signature ? `<div><dt>Signature</dt><dd>${esc(f.matched_signature)}</dd></div>` : ""}
      </dl>
    </article>`).join("");
  const danger = r.dangerousImports.length ? `
    <article class="card danger">
      <h3>Loading this file with torch.load would call</h3>
      <ul class="mono">${r.dangerousImports.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>
      <p class="muted">Don't open it with <code>torch.load(weights_only=False)</code>. Kavach only read it; nothing ran.</p>
    </article>` : "";
  const custom = r.customCode.length ? `<p class="muted small">Also imports custom classes: <span class="mono">${r.customCode.map(esc).join(", ")}</span>. Loading it needs that code.</p>` : "";
  result.innerHTML = `
    <div class="card verdict ${kind}">
      <p class="eyebrow">${esc(r.file)} · ${fmtBytes(r.bytes)}</p>
      <h2>${headline}</h2>
      <dl class="stats">
        <div><dt>Tensors</dt><dd>${fmtInt(r.tensors)}</dd></div>
        <div><dt>Weights scanned</dt><dd>${fmtInt(r.scannedWeights)}</dd></div>
        <div><dt>Skipped</dt><dd>${fmtInt(r.skipped.length)}</dd></div>
        <div><dt>Findings</dt><dd>${fmtInt(r.findings.length)}</dd></div>
      </dl>
      ${custom}
      <div class="actions">
        ${r.findings.length ? `<button class="pill" id="disarm">Disarm and download</button>` : ""}
        <button class="${r.findings.length ? "ghost" : "pill"}" id="report">Download report</button>
        <button class="ghost" id="again">Scan another file</button>
      </div>
      <p class="muted small" id="disarm-note">${r.findings.length ? `Disarm randomises the lowest bit of the ${layers.length} flagged tensor${layers.length === 1 ? "" : "s"} and saves every tensor as .safetensors.` : ""}</p>
    </div>
    ${danger}
    ${findings ? `<div class="stack">${findings}</div>` : ""}
    ${chart(r)}
    ${r.skipped.length ? `<details class="card skipped"><summary>${r.skipped.length} tensor${r.skipped.length === 1 ? "" : "s"} skipped (not floating point)</summary><p class="mono small">${r.skipped.map(esc).join(", ")}</p></details>` : ""}`;
  $("again").onclick = reset;
  $("report").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], { type: "application/json" }));
    a.download = r.file.replace(/\.(pt|pth|safetensors)$/i, "") + ".kavach-report.json";
    a.click();
  };
  const d = document.getElementById("disarm") as HTMLButtonElement | null;
  if (d) d.onclick = () => {
    d.disabled = true;
    $("disarm-note").textContent = "Writing the cleaned file and re-scanning it";
    worker!.postMessage({ type: "disarm", layers });
  };
  show("result");
}

function reset() {
  worker?.terminate();
  worker = null;

  input.value = "";
  show("drop");
}

input.onchange = () => { if (input.files?.[0]) scan(input.files[0]); };
$("cancel").onclick = reset;
const zone = $("dropzone");
zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
zone.addEventListener("dragleave", () => zone.classList.remove("over"));
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("over");
  const f = e.dataTransfer?.files[0];
  if (f) scan(f);
});
document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) => {
  b.onclick = async () => {
    const name = b.dataset.sample!;
    const res = await fetch(`${import.meta.env.BASE_URL}samples/${name}`);
    scan(new File([await res.blob()], name));
  };
});
