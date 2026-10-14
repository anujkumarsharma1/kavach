import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

test("clean sample scans clean", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "Clean model" }).click();
  await expect(page.getByRole("heading", { name: "No hidden strings found" })).toBeVisible();
});

test("infected checkpoint shows findings, disarm downloads a clean safetensors", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "Infected checkpoint" }).click();
  await expect(page.getByRole("heading", { name: "2 hidden strings found" })).toBeVisible();
  await expect(page.getByText("Known signature")).toBeVisible();
  const dl = page.waitForEvent("download");
  await page.getByRole("button", { name: "Disarm and download" }).click();
  const file = await dl;
  expect(file.suggestedFilename()).toBe("infected-checkpoint.kavach-clean.safetensors");
  await expect(page.locator("#disarm-note")).toContainText("no hidden strings");
  const rep = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download report" }).click();
  const report = JSON.parse(readFileSync((await (await rep).path())!, "utf8"));
  expect(report.findings).toHaveLength(2);
});

test("dangerous pickle is flagged Dangerous", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "Dangerous pickle" }).click();
  await expect(page.getByRole("heading", { name: /Dangerous/ })).toBeVisible();
  await expect(page.getByText("posix.system")).toBeVisible();
});

test("uploaded file goes through the picker, and a bad file is refused", async ({ page }) => {
  await page.goto("./");
  await page.setInputFiles("#file", join(process.cwd(), "tests", "fixtures", "infected-flat-fp16.pt"));
  await expect(page.getByRole("heading", { name: "2 hidden strings found" })).toBeVisible();
  await page.getByRole("button", { name: "Scan another file" }).click();
  await page.setInputFiles("#file", join(process.cwd(), "tests", "fixtures", "malformed-truncated.pt"));
  await expect(page.getByRole("heading", { name: /Not readable/ })).toBeVisible();
});
