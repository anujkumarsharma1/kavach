import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 60000,
  use: {
    baseURL: process.env.BASE_URL ?? "http://localhost:4173/kavach/",
    launchOptions: process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  },
  webServer: process.env.BASE_URL ? undefined : { command: "npm run preview", url: "http://localhost:4173/kavach/", reuseExistingServer: true },
});
