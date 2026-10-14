import { openAsBlob } from "node:fs";
import { readFileSync } from "node:fs";
import { webcrypto } from "node:crypto";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { disarmToSafetensors } from "../src/core/disarm";
import { scanModel } from "../src/core/model";
import { blobSource } from "../src/core/source";

const FIX = join(__dirname, "fixtures");
const expected = JSON.parse(readFileSync(join(FIX, "expected.json"), "utf8"));

const open = async (name: string) => blobSource(await openAsBlob(join(FIX, name)));
const key = (f: { layer: string; bit_start: number; bit_phase: number }) => `${f.layer}|${f.bit_phase}|${f.bit_start}`;
const sortF = (a: any[]) => [...a].sort((x, y) => key(x).localeCompare(key(y)));

describe("matches the Python reference", () => {
  for (const name of Object.keys(expected)) {
    it(name, async () => {
      const want = expected[name];
      const { report } = await scanModel(await open(name), name);
      expect(sortF(report.findings)).toEqual(sortF(want.findings));
      expect(Object.keys(report.peaks).sort()).toEqual(Object.keys(want.peaks).sort());
      for (const [k, v] of Object.entries(want.peaks)) expect(report.peaks[k]).toBeCloseTo(v as number, 12);
      expect([...report.skipped].sort()).toEqual([...want.skipped].sort());
      expect(report.dangerousImports).toEqual(want.dangerous);
      const verdict = want.dangerous.length ? "Dangerous" : want.findings.length ? "Hidden strings found" : "Clean";
      expect(report.verdict).toBe(verdict);
    });
  }
});

describe("disarm", () => {
  for (const name of ["infected-nested-fp32.pt", "infected-flat-fp16.pt", "infected-safetensors-bf16.safetensors", "infected-whole-fp32.pt"]) {
    it(`${name} re-scans clean and keeps unflagged tensors byte-identical`, async () => {
      const src = await open(name);
      const { report, loaded } = await scanModel(src, name);
      const flagged = new Set(report.findings.map((f) => f.layer));
      expect(flagged.size).toBeGreaterThan(0);
      const blob = await disarmToSafetensors(src, loaded.tensors, flagged, (b) => webcrypto.getRandomValues(b));
      const out = blobSource(blob);
      const again = await scanModel(out, "disarmed.safetensors");
      expect(again.report.findings).toEqual([]);
      expect(again.report.tensors).toBe(loaded.tensors.length);
      for (const [layer, peak] of Object.entries(report.peaks)) {
        if (!flagged.has(layer)) expect(again.report.peaks[layer]).toBe(peak);
      }
    });
  }
});

describe("malformed files are refused, never reported clean", () => {
  const cases: [string, RegExp][] = [
    ["malformed-truncated.pt", /zip|end|past/i],
    ["malformed-empty.pt", /empty|small/i],
    ["malformed-random.pt", /not a PyTorch/i],
    ["malformed-header.safetensors", /header length/i],
    ["malformed-legacy.pt", /legacy/i],
    ["malformed-deflated.pt", /compressed/i],
  ];
  for (const [name, msg] of cases) {
    it(name, async () => {
      await expect(scanModel(await open(name), name)).rejects.toThrow(msg);
    });
  }
  it("rejects other extensions", async () => {
    await expect(scanModel(await open("clean-flat-fp32.pt"), "model.onnx")).rejects.toThrow(/only/);
  });
});
