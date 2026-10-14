import { readSafetensors } from "./safetensors";
import { Finding, Progress, scanTensors } from "./scan";
import { FormatError, Source, TensorRef } from "./source";
import { readTorchZip } from "./torchzip";

export type Verdict = "Clean" | "Hidden strings found" | "Dangerous";

export interface Report {
  file: string;
  bytes: number;
  format: "pytorch" | "safetensors";
  verdict: Verdict;
  dangerousImports: string[];
  customCode: string[];
  tensors: number;
  scannedWeights: number;
  skipped: string[];
  peaks: Record<string, number>;
  findings: Finding[];
  scanner: { window_bytes: number; stride_bytes: number; threshold: number; bit_phases: number };
}

export interface Loaded { format: Report["format"]; tensors: TensorRef[]; dangerous: string[]; custom: string[] }

export async function loadModel(src: Source, fileName: string): Promise<Loaded> {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".safetensors")) {
    const { tensors } = await readSafetensors(src);
    return { format: "safetensors", tensors, dangerous: [], custom: [] };
  }
  if (lower.endsWith(".pt") || lower.endsWith(".pth")) {
    const t = await readTorchZip(src);
    return { format: "pytorch", tensors: t.tensors, dangerous: t.dangerous, custom: t.custom };
  }
  throw new FormatError("only .pt, .pth and .safetensors files are supported");
}

export async function scanModel(src: Source, fileName: string, onProgress?: (p: Progress) => void, isCancelled?: () => boolean): Promise<{ report: Report; loaded: Loaded }> {
  const loaded = await loadModel(src, fileName);
  const r = await scanTensors(src, loaded.tensors, onProgress, isCancelled);
  const verdict: Verdict = loaded.dangerous.length ? "Dangerous" : r.findings.length ? "Hidden strings found" : "Clean";
  return {
    loaded,
    report: {
      file: fileName, bytes: src.size, format: loaded.format, verdict,
      dangerousImports: loaded.dangerous, customCode: loaded.custom,
      tensors: loaded.tensors.length, scannedWeights: r.scannedWeights, skipped: r.skipped,
      peaks: r.peaks, findings: r.findings,
      scanner: { window_bytes: 64, stride_bytes: 8, threshold: 0.75, bit_phases: 8 },
    },
  };
}
