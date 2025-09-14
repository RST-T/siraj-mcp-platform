import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/results.xml' }]
  ],
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    headless: process.env.CI ? true : false,
  },

  projects: [
    {
      name: 'setup',
      testMatch: /global\.setup\.ts/,
    },
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
      dependencies: ['setup'],
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
      dependencies: ['setup'],
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
      dependencies: ['setup'],
    },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
      dependencies: ['setup'],
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 12'] },
      dependencies: ['setup'],
    },
  ],

  webServer: [
    {
      command: 'npm run dev',
      cwd: './frontend',
      port: 3000,
      reuseExistingServer: !process.env.CI,
      env: {
        NODE_ENV: 'test',
      },
    },
    {
      command: 'python -m uvicorn main:app --host 0.0.0.0 --port 8000',
      cwd: './backend',
      port: 8000,
      reuseExistingServer: !process.env.CI,
      env: {
        ENVIRONMENT: 'test',
        DATABASE_URL: 'sqlite:///test.db',
      },
    },
  ],

  globalSetup: require.resolve('./tests/global.setup.ts'),
  globalTeardown: require.resolve('./tests/global.teardown.ts'),
});