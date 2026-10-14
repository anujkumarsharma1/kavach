import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/kavach/",
  worker: { format: "es" },
  test: { include: ["tests/**/*.test.ts"], testTimeout: 30000 },
});
