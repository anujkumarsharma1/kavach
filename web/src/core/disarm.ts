import { ITEMSIZE, readTensorBytes, Source, TensorRef, DType } from "./source";

// Writes every readable tensor to a new safetensors file. Tensors named in
// `flagged` get a fresh random bit 0 in every element, which destroys any
// string stored there and changes each value by at most one unit in the last place.
export async function disarmToSafetensors(src: Source, tensors: TensorRef[], flagged: Set<string>, random: (b: Uint8Array) => void): Promise<Blob> {
  const parts: BlobPart[] = [];
  const header: Record<string, unknown> = { __metadata__: { format: "pt", producer: "kavach-disarm" } };
  let offset = 0;
  const used = new Set<string>();
  for (const t of tensors) {
    const size = ITEMSIZE[t.dtype as DType];
    if (!size || used.has(t.name)) continue;
    used.add(t.name);
    const bytes = new Uint8Array(await readTensorBytes(src, t));
    if (flagged.has(t.name)) {
      const noise = new Uint8Array(bytes.length / size);
      for (let i = 0; i < noise.length; i += 65536) random(noise.subarray(i, Math.min(i + 65536, noise.length)));
      for (let i = 0, j = 0; i < noise.length; i++, j += size) bytes[j] = (bytes[j] & 0xfe) | (noise[i] & 1);
    }
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + bytes.length] };
    offset += bytes.length;
    parts.push(bytes);
  }
  let json = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - (json.length % 8)) % 8;
  json = new TextEncoder().encode(JSON.stringify(header) + " ".repeat(pad));
  const len = new Uint8Array(8);
  new DataView(len.buffer).setBigUint64(0, BigInt(json.length), true);
  return new Blob([len, json as BlobPart, ...parts], { type: "application/octet-stream" });
}
