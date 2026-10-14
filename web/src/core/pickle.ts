// Reads a pickle stream as data. No object is ever constructed from a
// global and no code from the file runs: GLOBAL only records a name, and
// REDUCE / NEWOBJ only record which name was called with which arguments.

export interface PyGlobal { kind: "global"; module: string; name: string }
export interface PyCall { kind: "call"; fn: PyGlobal | unknown; args: unknown[]; state?: unknown; dict: Map<unknown, unknown>; list: unknown[] }
export interface PyPersistent { kind: "persid"; id: unknown }

export const MAX_OPS = 5_000_000;
export const MAX_STACK = 100_000;

export class PickleError extends Error {}

const MARK = Symbol("mark");

export interface PickleResult { value: unknown; globals: PyGlobal[] }

export function readPickle(buf: Uint8Array): PickleResult {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const dec = new TextDecoder("utf-8");
  const latin = new TextDecoder("latin1");
  let pos = 0;
  const stack: unknown[] = [];
  const memo = new Map<number, unknown>();
  const globals: PyGlobal[] = [];
  let ops = 0;

  const need = (n: number) => { if (pos + n > buf.length) throw new PickleError("pickle stream ends early"); };
  const u8 = () => { need(1); return buf[pos++]; };
  const u16 = () => { need(2); const v = view.getUint16(pos, true); pos += 2; return v; };
  const u32 = () => { need(4); const v = view.getUint32(pos, true); pos += 4; return v; };
  const i32 = () => { need(4); const v = view.getInt32(pos, true); pos += 4; return v; };
  const bytes = (n: number) => { need(n); const v = buf.subarray(pos, pos + n); pos += n; return v; };
  const line = () => { const end = buf.indexOf(0x0a, pos); if (end < 0) throw new PickleError("pickle stream ends early"); const s = latin.decode(buf.subarray(pos, end)); pos = end + 1; return s; };
  const push = (v: unknown) => { if (stack.length >= MAX_STACK) throw new PickleError("pickle stack too deep"); stack.push(v); };
  const pop = () => { if (!stack.length) throw new PickleError("pickle stack underflow"); return stack.pop(); };
  const top = () => { if (!stack.length) throw new PickleError("pickle stack underflow"); return stack[stack.length - 1]; };
  const popMark = () => {
    const i = stack.lastIndexOf(MARK);
    if (i < 0) throw new PickleError("pickle MARK missing");
    const items = stack.splice(i);
    items.shift();
    return items;
  };
  const mkGlobal = (module: string, name: string) => { const g: PyGlobal = { kind: "global", module, name }; globals.push(g); return g; };
  const call = (fn: unknown, args: unknown): PyCall => ({ kind: "call", fn, args: Array.isArray(args) ? args : [args], dict: new Map(), list: [] });
  const isOrderedDict = (fn: unknown) => { const g = fn as PyGlobal; return g?.kind === "global" && g.module === "collections" && g.name === "OrderedDict"; };
  const setItem = (target: unknown, k: unknown, v: unknown) => {
    if (target instanceof Map) target.set(k, v);
    else if ((target as PyCall)?.kind === "call") (target as PyCall).dict.set(k, v);
    else throw new PickleError("SETITEM on a non-dict");
  };
  const append = (target: unknown, v: unknown) => {
    if (Array.isArray(target)) target.push(v);
    else if ((target as PyCall)?.kind === "call") (target as PyCall).list.push(v);
    else if (target instanceof Set) target.add(v);
    else throw new PickleError("APPEND on a non-list");
  };
  const longFromBytes = (b: Uint8Array) => {
    let v = 0n;
    for (let i = b.length - 1; i >= 0; i--) v = (v << 8n) | BigInt(b[i]);
    if (b.length && b[b.length - 1] & 0x80) v -= 1n << BigInt(8 * b.length);
    return Number(v);
  };

  for (;;) {
    if (++ops > MAX_OPS) throw new PickleError("pickle has too many opcodes");
    const op = u8();
    switch (op) {
      case 0x80: u8(); break; // PROTO
      case 0x95: bytes(8); break; // FRAME
      case 0x2e: return { value: pop(), globals }; // STOP
      case 0x28: push(MARK); break; // MARK
      case 0x30: pop(); break; // POP
      case 0x31: popMark(); break; // POP_MARK
      case 0x32: push(top()); break; // DUP
      case 0x4e: push(null); break; // NONE
      case 0x88: push(true); break; // NEWTRUE
      case 0x89: push(false); break; // NEWFALSE
      case 0x4b: push(u8()); break; // BININT1
      case 0x4d: push(u16()); break; // BININT2
      case 0x4a: push(i32()); break; // BININT
      case 0x8a: push(longFromBytes(bytes(u8()))); break; // LONG1
      case 0x8b: push(longFromBytes(bytes(i32()))); break; // LONG4
      case 0x49: { const s = line(); push(s === "01" ? true : s === "00" ? false : Number(s)); break; } // INT
      case 0x4c: push(Number(line().replace(/L$/, ""))); break; // LONG
      case 0x46: push(Number(line())); break; // FLOAT
      case 0x47: { need(8); push(view.getFloat64(pos, false)); pos += 8; break; } // BINFLOAT
      case 0x58: push(dec.decode(bytes(u32()))); break; // BINUNICODE
      case 0x8c: push(dec.decode(bytes(u8()))); break; // SHORT_BINUNICODE
      case 0x8d: { const n = Number(view.getBigUint64((need(8), pos), true)); pos += 8; push(dec.decode(bytes(n))); break; } // BINUNICODE8
      case 0x56: push(line()); break; // UNICODE
      case 0x53: push(line().replace(/^['"]|['"]$/g, "")); break; // STRING
      case 0x54: push(latin.decode(bytes(i32()))); break; // BINSTRING
      case 0x55: push(latin.decode(bytes(u8()))); break; // SHORT_BINSTRING
      case 0x42: push(bytes(u32()).slice()); break; // BINBYTES
      case 0x43: push(bytes(u8()).slice()); break; // SHORT_BINBYTES
      case 0x8e: { const n = Number(view.getBigUint64((need(8), pos), true)); pos += 8; push(bytes(n).slice()); break; } // BINBYTES8
      case 0x96: { const n = Number(view.getBigUint64((need(8), pos), true)); pos += 8; push(bytes(n).slice()); break; } // BYTEARRAY8
      case 0x7d: push(new Map()); break; // EMPTY_DICT
      case 0x64: { const it = popMark(); const m = new Map(); for (let i = 0; i + 1 < it.length; i += 2) m.set(it[i], it[i + 1]); push(m); break; } // DICT
      case 0x5d: push([]); break; // EMPTY_LIST
      case 0x6c: push(popMark()); break; // LIST
      case 0x29: push([]); break; // EMPTY_TUPLE
      case 0x74: push(popMark()); break; // TUPLE
      case 0x85: push([pop()]); break; // TUPLE1
      case 0x86: { const b = pop(); const a = pop(); push([a, b]); break; } // TUPLE2
      case 0x87: { const c = pop(); const b = pop(); const a = pop(); push([a, b, c]); break; } // TUPLE3
      case 0x8f: push(new Set()); break; // EMPTY_SET
      case 0x90: { const it = popMark(); const s = top(); for (const v of it) append(s, v); break; } // ADDITEMS
      case 0x91: push(new Set(popMark())); break; // FROZENSET
      case 0x73: { const v = pop(); const k = pop(); setItem(top(), k, v); break; } // SETITEM
      case 0x75: { const it = popMark(); const t = top(); for (let i = 0; i + 1 < it.length; i += 2) setItem(t, it[i], it[i + 1]); break; } // SETITEMS
      case 0x61: { const v = pop(); append(top(), v); break; } // APPEND
      case 0x65: { const it = popMark(); const t = top(); for (const v of it) append(t, v); break; } // APPENDS
      case 0x70: memo.set(Number(line()), top()); break; // PUT
      case 0x71: memo.set(u8(), top()); break; // BINPUT
      case 0x72: memo.set(u32(), top()); break; // LONG_BINPUT
      case 0x94: memo.set(memo.size, top()); break; // MEMOIZE
      case 0x67: push(memo.get(Number(line()))); break; // GET
      case 0x68: { const k = u8(); if (!memo.has(k)) throw new PickleError("pickle memo miss"); push(memo.get(k)); break; } // BINGET
      case 0x6a: { const k = u32(); if (!memo.has(k)) throw new PickleError("pickle memo miss"); push(memo.get(k)); break; } // LONG_BINGET
      case 0x63: { const module = line(); const name = line(); push(mkGlobal(module, name)); break; } // GLOBAL
      case 0x93: { const name = String(pop()); const module = String(pop()); push(mkGlobal(module, name)); break; } // STACK_GLOBAL
      case 0x52: { const args = pop(); const fn = pop(); push(isOrderedDict(fn) ? new Map() : call(fn, args)); break; } // REDUCE
      case 0x81: { const args = pop(); const cls = pop(); push(call(cls, args)); break; } // NEWOBJ
      case 0x92: { pop(); const args = pop(); const cls = pop(); push(call(cls, args)); break; } // NEWOBJ_EX
      case 0x6f: { const it = popMark(); const cls = it.shift(); push(call(cls, it)); break; } // OBJ
      case 0x69: { const module = line(); const name = line(); push(call(mkGlobal(module, name), popMark())); break; } // INST
      case 0x62: { const state = pop(); const t = top(); if ((t as PyCall)?.kind === "call") (t as PyCall).state = state; break; } // BUILD
      case 0x51: { const id = pop(); push({ kind: "persid", id } as PyPersistent); break; } // BINPERSID
      case 0x50: push({ kind: "persid", id: line() } as PyPersistent); break; // PERSID
      default:
        throw new PickleError(`unsupported pickle opcode 0x${op.toString(16)} at byte ${pos - 1}`);
    }
  }
}

const DANGEROUS_MODULES = [
  "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "runpy", "pty", "importlib",
  "pickle", "_pickle", "marshal", "ctypes", "webbrowser", "urllib", "requests", "http", "ftplib",
  "smtplib", "telnetlib", "code", "codeop", "commands", "platform", "multiprocessing", "signal",
  "pdb", "timeit", "cProfile", "profile", "asyncio", "threading", "_thread", "tempfile", "glob",
  "zipimport", "pip", "setuptools", "distutils", "types", "operator", "functools",
];
const DANGEROUS_BUILTINS = new Set([
  "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "globals",
  "locals", "vars", "input", "breakpoint", "memoryview", "help", "exit", "quit",
]);

export function isDangerous(g: PyGlobal): boolean {
  const root = g.module.split(".")[0];
  if (root === "builtins" || root === "__builtin__") return DANGEROUS_BUILTINS.has(g.name);
  if (root === "torch") return g.module === "torch.hub" || g.name === "load" || g.module.startsWith("torch.utils.cpp_extension");
  if (root === "numpy") return g.name === "load" || g.name === "loadtxt" || g.module.startsWith("numpy.testing") || g.module.startsWith("numpy.distutils");
  return DANGEROUS_MODULES.includes(root);
}

const KNOWN_ROOTS = new Set(["torch", "collections", "numpy", "builtins", "__builtin__", "_codecs", "copyreg", "math"]);
export function isCustom(g: PyGlobal): boolean {
  return !isDangerous(g) && !KNOWN_ROOTS.has(g.module.split(".")[0]);
}
