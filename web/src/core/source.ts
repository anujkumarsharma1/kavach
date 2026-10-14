export interface Source {
  size: number;
  read(offset: number, length: number): Promise<Uint8Array>;
}

export function blobSource(blob: Blob): Source {
  return {
    size: blob.size,
    async read(offset, length) {
      if (offset < 0 || length < 0 || offset + length > blob.size) throw new FormatError("a section points past the end of the file");
      return new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer());
    },
  };
}

export class FormatError extends Error {}

export type DType = "F32" | "F16" | "BF16" | "F64" | "I64" | "I32" | "I16" | "I8" | "U8" | "BOOL";
export const ITEMSIZE: Record<DType, number> = { F32: 4, F16: 2, BF16: 2, F64: 8, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };
export const FLOAT_DTYPES = new Set<DType>(["F32", "F16", "BF16", "F64"]);

// A tensor whose values live at `byteOffset` in the file. Element i of the
// logical (row-major) tensor sits at storage element `storageOffset + sum(idx*stride)`.
export interface TensorRef {
  name: string;
  dtype: DType | string;
  shape: number[];
  stride: number[];
  storageOffset: number;
  byteOffset: number;
  storageBytes: number;
}

export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

// Returns the tensor's elements in logical order as raw little-endian bytes.
export async function readTensorBytes(src: Source, t: TensorRef): Promise<Uint8Array> {
  const size = ITEMSIZE[t.dtype as DType];
  const n = numel(t.shape);
  const raw = await src.read(t.byteOffset, t.storageBytes);
  const contiguous = isContiguous(t.shape, t.stride);
  if (contiguous) {
    const start = t.storageOffset * size;
    if (start + n * size > raw.length) throw new FormatError(`tensor ${t.name} is larger than its storage`);
    return raw.subarray(start, start + n * size);
  }
  const out = new Uint8Array(n * size);
  const idx = new Array(t.shape.length).fill(0);
  let s = t.storageOffset;
  for (let i = 0; i < n; i++) {
    const at = s * size;
    if (at < 0 || at + size > raw.length) throw new FormatError(`tensor ${t.name} reads outside its storage`);
    out.set(raw.subarray(at, at + size), i * size);
    for (let d = t.shape.length - 1; d >= 0; d--) {
      idx[d]++;
      s += t.stride[d];
      if (idx[d] < t.shape[d]) break;
      s -= t.stride[d] * t.shape[d];
      idx[d] = 0;
    }
  }
  return out;
}

export function isContiguous(shape: number[], stride: number[]): boolean {
  let expect = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    if (shape[d] !== 1 && stride[d] !== expect) return false;
    expect *= shape[d];
  }
  return true;
}

export function contiguousStride(shape: number[]): number[] {
  const s = new Array(shape.length);
  let acc = 1;
  for (let d = shape.length - 1; d >= 0; d--) { s[d] = acc; acc *= shape[d]; }
  return s;
}
