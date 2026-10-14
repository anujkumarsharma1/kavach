import { disarmToSafetensors } from "./core/disarm";
import { Loaded, scanModel } from "./core/model";
import { blobSource } from "./core/source";

let last: { file: File; loaded: Loaded } | null = null;

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === "scan") {
      const file: File = msg.file;
      const { report, loaded } = await scanModel(blobSource(file), file.name, (p) => postMessage({ type: "progress", ...p }));
      last = { file, loaded };
      postMessage({ type: "result", report });
    } else if (msg.type === "disarm" && last) {
      const flagged = new Set<string>(msg.layers);
      const blob = await disarmToSafetensors(blobSource(last.file), last.loaded.tensors, flagged, (b) => crypto.getRandomValues(b));
      const check = await scanModel(blobSource(blob), "disarmed.safetensors");
      postMessage({ type: "disarmed", blob, remaining: check.report.findings.length });
    }
  } catch (e) {
    postMessage({ type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
