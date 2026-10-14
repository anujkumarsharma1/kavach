import { DType, FLOAT_DTYPES, ITEMSIZE, readTensorBytes, Source, TensorRef } from "./source";

export const WINDOW_BYTES = 64;
export const STRIDE_BYTES = 8;
export const THRESHOLD = 0.75;
export const SIGNATURES = ["EICAR-STANDARD-ANTIVIRUS-TEST-FILE"];

export interface Finding {
  layer: string;
  bit_start: number;
  bit_end: number;
  bit_phase: number;
  printable_ratio: number;
  severity: "CRITICAL" | "SUSPICIOUS";
  matched_signature: string | null;
  decoded_hex: string;
}

// One LSB per element: byte 0 of each little-endian element holds bit 0.
export function lsbBits(bytes: Uint8Array, itemsize: number): Uint8Array {
  const n = bytes.length / itemsize;
  const bits = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += itemsize) bits[i] = bytes[j] & 1;
  return bits;
}

const SIG_BYTES = SIGNATURES.map((s) => new TextEncoder().encode(s));

function indexOf(hay: Uint8Array, needle: Uint8Array): number {
  outer: for (let i = 0; i + needle.length <= hay.length; i++) {
    for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");

export function scanBits(bits: Uint8Array, layer: string): { findings: Finding[]; peak: number } {
  const findings: Finding[] = [];
  let peak = 0;
  for (let phase = 0; phase < 8; phase++) {
    const usable = Math.floor((bits.length - phase) / 8);
    if (usable < WINDOW_BYTES) continue;
    const data = new Uint8Array(usable);
    for (let i = 0, b = phase; i < usable; i++, b += 8) {
      data[i] = (bits[b] << 7) | (bits[b + 1] << 6) | (bits[b + 2] << 5) | (bits[b + 3] << 4) |
        (bits[b + 4] << 3) | (bits[b + 5] << 2) | (bits[b + 6] << 1) | bits[b + 7];
    }
    const cum = new Int32Array(usable + 1);
    for (let i = 0; i < usable; i++) cum[i + 1] = cum[i] + (data[i] >= 32 && data[i] <= 126 ? 1 : 0);
    const regions: [number, number, number][] = [];
    for (let s = 0; s + WINDOW_BYTES <= usable; s += STRIDE_BYTES) {
      const r = (cum[s + WINDOW_BYTES] - cum[s]) / WINDOW_BYTES;
      if (r > peak) peak = r;
      if (r < THRESHOLD) continue;
      const last = regions[regions.length - 1];
      if (last && s <= last[1]) { last[1] = Math.max(last[1], s + WINDOW_BYTES); last[2] = Math.max(last[2], r); }
      else regions.push([s, s + WINDOW_BYTES, r]);
    }
    for (const [s, e, r] of regions) {
      const decoded = data.subarray(s, e);
      const sig = SIG_BYTES.findIndex((x) => indexOf(decoded, x) >= 0);
      findings.push({
        layer, bit_start: phase + s * 8, bit_end: phase + e * 8, bit_phase: phase, printable_ratio: r,
        severity: sig >= 0 ? "CRITICAL" : "SUSPICIOUS", matched_signature: sig >= 0 ? SIGNATURES[sig] : null,
        decoded_hex: hex(decoded),
      });
    }
  }
  return { findings, peak };
}

export interface ScanResult {
  findings: Finding[];
  peaks: Record<string, number>;
  skipped: string[];
  scannedWeights: number;
}

export interface Progress { done: number; total: number; current: string }

export async function scanTensors(src: Source, tensors: TensorRef[], onProgress?: (p: Progress) => void, isCancelled?: () => boolean): Promise<ScanResult> {
  const result: ScanResult = { findings: [], peaks: {}, skipped: [], scannedWeights: 0 };
  for (let i = 0; i < tensors.length; i++) {
    if (isCancelled?.()) throw new Error("cancelled");
    const t = tensors[i];
    onProgress?.({ done: i, total: tensors.length, current: t.name });
    if (!FLOAT_DTYPES.has(t.dtype as DType)) { result.skipped.push(t.name); continue; }
    const bits = lsbBits(await readTensorBytes(src, t), ITEMSIZE[t.dtype as DType]);
    const { findings, peak } = scanBits(bits, t.name);
    result.peaks[t.name] = peak;
    result.scannedWeights += bits.length;
    result.findings.push(...findings);
  }
  onProgress?.({ done: tensors.length, total: tensors.length, current: "" });
  return result;
}
