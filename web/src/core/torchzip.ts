import { isCustom, isDangerous, PickleError, PyCall, PyGlobal, PyPersistent, readPickle } from "./pickle";
import { DType, FormatError, Source, TensorRef } from "./source";

export const MAX_PICKLE = 64 * 1024 * 1024;

interface ZipEntry { name: string; method: number; compSize: number; size: number; localOffset: number }

const STORAGE_DTYPES: Record<string, DType> = {
  FloatStorage: "F32", HalfStorage: "F16", BFloat16Storage: "BF16", DoubleStorage: "F64",
  LongStorage: "I64", IntStorage: "I32", ShortStorage: "I16", CharStorage: "I8", ByteStorage: "U8", BoolStorage: "BOOL",
};
const TORCH_DTYPES: Record<string, DType> = {
  float32: "F32", float: "F32", float16: "F16", half: "F16", bfloat16: "BF16", float64: "F64", double: "F64",
  int64: "I64", long: "I64", int32: "I32", int: "I32", int16: "I16", int8: "I8", uint8: "U8", bool: "BOOL",
};

async function readZipDirectory(src: Source): Promise<ZipEntry[]> {
  const tailLen = Math.min(src.size, 65557);
  const tail = await src.read(src.size - tailLen, tailLen);
  const tv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) if (tv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new FormatError("not a zip archive (no end-of-directory record)");
  let count = tv.getUint16(eocd + 10, true);
  let cdSize = tv.getUint32(eocd + 12, true);
  let cdOffset = tv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    const loc = eocd - 20;
    if (loc < 0 || tv.getUint32(loc, true) !== 0x07064b50) throw new FormatError("zip64 locator missing");
    const z64Off = Number(tv.getBigUint64(loc + 8, true));
    const z = await src.read(z64Off, 56);
    const zv = new DataView(z.buffer, z.byteOffset, 56);
    if (zv.getUint32(0, true) !== 0x06064b50) throw new FormatError("zip64 directory record missing");
    count = Number(zv.getBigUint64(32, true));
    cdSize = Number(zv.getBigUint64(40, true));
    cdOffset = Number(zv.getBigUint64(48, true));
  }
  const cd = await src.read(cdOffset, cdSize);
  const v = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
  const entries: ZipEntry[] = [];
  let p = 0;
  for (let i = 0; i < count; i++) {
    if (p + 46 > cd.length || v.getUint32(p, true) !== 0x02014b50) throw new FormatError("zip directory is corrupt");
    const method = v.getUint16(p + 10, true);
    let compSize = v.getUint32(p + 20, true);
    let size = v.getUint32(p + 24, true);
    const nameLen = v.getUint16(p + 28, true);
    const extraLen = v.getUint16(p + 30, true);
    const commentLen = v.getUint16(p + 32, true);
    let localOffset = v.getUint32(p + 42, true);
    const name = new TextDecoder().decode(cd.subarray(p + 46, p + 46 + nameLen));
    let e = p + 46 + nameLen;
    const extraEnd = e + extraLen;
    while (e + 4 <= extraEnd) {
      const id = v.getUint16(e, true), len = v.getUint16(e + 2, true);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) { size = Number(v.getBigUint64(q, true)); q += 8; }
        if (compSize === 0xffffffff) { compSize = Number(v.getBigUint64(q, true)); q += 8; }
        if (localOffset === 0xffffffff) { localOffset = Number(v.getBigUint64(q, true)); }
      }
      e += 4 + len;
    }
    entries.push({ name, method, compSize, size, localOffset });
    p = extraEnd + commentLen;
  }
  return entries;
}

async function dataOffset(src: Source, e: ZipEntry): Promise<number> {
  const h = await src.read(e.localOffset, 30);
  const hv = new DataView(h.buffer, h.byteOffset, 30);
  if (hv.getUint32(0, true) !== 0x04034b50) throw new FormatError(`zip entry ${e.name} is corrupt`);
  return e.localOffset + 30 + hv.getUint16(26, true) + hv.getUint16(28, true);
}

const g = (v: unknown): PyGlobal | undefined => ((v as PyGlobal)?.kind === "global" ? (v as PyGlobal) : undefined);
const isCall = (v: unknown): v is PyCall => (v as PyCall)?.kind === "call";
const fnName = (c: PyCall) => { const f = g(c.fn); return f ? `${f.module}.${f.name}` : ""; };

interface RawTensor { storage: PyPersistent; offset: number; shape: number[]; stride: number[] }

function asTensor(v: unknown): RawTensor | undefined {
  if (!isCall(v)) return undefined;
  const name = fnName(v);
  if (name === "torch._utils._rebuild_parameter" || name === "torch._utils._rebuild_parameter_with_state") return asTensor(v.args[0]);
  if (name === "torch._utils._rebuild_tensor_v2" || name === "torch._utils._rebuild_tensor") {
    const [storage, offset, shape, stride] = v.args as [PyPersistent, number, number[], number[]];
    if ((storage as PyPersistent)?.kind !== "persid") return undefined;
    return { storage, offset: Number(offset), shape: (shape as number[]).map(Number), stride: (stride as number[]).map(Number) };
  }
  return undefined;
}

const MODULE_CONTAINERS = new Set(["_parameters", "_buffers", "_modules"]);

// Names follow nn.Module.state_dict(): a module's _parameters/_buffers/_modules
// levels are dropped, dict keys and list indices are joined with dots.
function collect(value: unknown, path: string, out: { name: string; t: RawTensor }[], seen: Set<unknown>) {
  const t = asTensor(value);
  if (t) { out.push({ name: path, t }); return; }
  if (value === null || typeof value !== "object") return;
  if (seen.has(value)) return;
  seen.add(value);
  const join = (k: unknown) => (path ? `${path}.${String(k)}` : String(k));
  if (value instanceof Map) { for (const [k, v] of value) collect(v, join(k), out, seen); return; }
  if (Array.isArray(value)) { value.forEach((v, i) => collect(v, join(i), out, seen)); return; }
  if (isCall(value)) {
    const state = Array.isArray(value.state) && value.state.length === 2 ? value.state[0] : value.state;
    if (state instanceof Map) {
      for (const [k, v] of state) collect(v, MODULE_CONTAINERS.has(String(k)) ? path : join(k), out, seen);
    } else if (state !== undefined) {
      collect(state, path, out, seen);
    }
    for (const [k, v] of value.dict) collect(v, join(k), out, seen);
    value.list.forEach((v, i) => collect(v, join(i), out, seen));
  }
}

export interface TorchFile { tensors: TensorRef[]; dangerous: string[]; custom: string[] }

export async function readTorchZip(src: Source): Promise<TorchFile> {
  if (src.size < 4) throw new FormatError("file is empty or too small");
  const magic = await src.read(0, 2);
  if (magic[0] === 0x80) throw new FormatError("this is a legacy (pre-1.6) PyTorch file; re-save it with a current torch.save");
  if (magic[0] !== 0x50 || magic[1] !== 0x4b) throw new FormatError("not a PyTorch .pt/.pth file (not a zip archive)");
  const entries = await readZipDirectory(src);
  const pkl = entries.find((e) => e.name.endsWith("/data.pkl") || e.name === "data.pkl");
  if (!pkl) throw new FormatError("no data.pkl inside the archive");
  const prefix = pkl.name.slice(0, pkl.name.length - "data.pkl".length);
  const order = entries.find((e) => e.name === `${prefix}byteorder`);
  if (order && order.method === 0) {
    const b = await src.read(await dataOffset(src, order), order.size);
    if (new TextDecoder().decode(b).trim() === "big") throw new FormatError("big-endian PyTorch files are not supported");
  }
  if (pkl.method !== 0) throw new FormatError("compressed PyTorch archives are not supported (torch.save writes them uncompressed)");
  if (pkl.size > MAX_PICKLE) throw new FormatError("the pickle inside this file is larger than 64 MB");

  let parsed;
  try {
    parsed = readPickle(await src.read(await dataOffset(src, pkl), pkl.size));
  } catch (e) {
    if (e instanceof PickleError) throw new FormatError(e.message);
    throw e;
  }
  const label = (x: PyGlobal) => `${x.module}.${x.name}`;
  const dangerous = [...new Set(parsed.globals.filter(isDangerous).map(label))];
  const custom = [...new Set(parsed.globals.filter(isCustom).map(label))];

  const found: { name: string; t: RawTensor }[] = [];
  collect(parsed.value, "", found, new Set());

  const byName = new Map(entries.map((e) => [e.name, e]));
  const tensors: TensorRef[] = [];
  for (const { name, t } of found) {
    const id = t.storage.id as unknown[];
    if (!Array.isArray(id) || id[0] !== "storage") throw new FormatError(`tensor ${name} has an unknown storage reference`);
    const [, type, key] = id;
    const tg = g(type);
    const dtype: DType | string = tg ? STORAGE_DTYPES[tg.name] ?? TORCH_DTYPES[tg.name] ?? tg.name : "unknown";
    const entry = byName.get(`${prefix}data/${key}`);
    if (!entry) throw new FormatError(`storage ${String(key)} for tensor ${name} is missing from the archive`);
    if (entry.method !== 0) throw new FormatError(`storage for ${name} is compressed`);
    tensors.push({
      name, dtype, shape: t.shape, stride: t.stride, storageOffset: t.offset,
      byteOffset: await dataOffset(src, entry), storageBytes: entry.size,
    });
  }
  return { tensors, dangerous, custom };
}
