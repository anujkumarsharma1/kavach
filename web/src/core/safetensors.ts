import { contiguousStride, DType, FormatError, ITEMSIZE, numel, Source, TensorRef } from "./source";

export const MAX_HEADER = 100 * 1024 * 1024;

export async function readSafetensors(src: Source): Promise<{ tensors: TensorRef[]; metadata: Record<string, string> }> {
  if (src.size < 8) throw new FormatError("file is too small to be a safetensors file");
  const lenBytes = await src.read(0, 8);
  const headerLen = Number(new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true));
  if (headerLen > MAX_HEADER || 8 + headerLen > src.size) throw new FormatError("safetensors header length is larger than the file");
  let header: Record<string, unknown>;
  try {
    header = JSON.parse(new TextDecoder().decode(await src.read(8, headerLen)));
  } catch {
    throw new FormatError("safetensors header is not valid JSON");
  }
  const base = 8 + headerLen;
  const tensors: TensorRef[] = [];
  let metadata: Record<string, string> = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") { metadata = (info ?? {}) as Record<string, string>; continue; }
    const { dtype, shape, data_offsets } = info as { dtype: string; shape: number[]; data_offsets: [number, number] };
    if (!Array.isArray(shape) || !Array.isArray(data_offsets)) throw new FormatError(`tensor ${name} has a malformed header entry`);
    const [start, end] = data_offsets;
    if (start < 0 || end < start || base + end > src.size) throw new FormatError(`tensor ${name} points past the end of the file`);
    const size = ITEMSIZE[dtype as DType];
    if (size && numel(shape) * size !== end - start) throw new FormatError(`tensor ${name} size does not match its shape`);
    tensors.push({ name, dtype, shape, stride: contiguousStride(shape), storageOffset: 0, byteOffset: base + start, storageBytes: end - start });
  }
  return { tensors, metadata };
}
