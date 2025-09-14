import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log('Starting global test teardown...');

  try {
    // Clean up test database
    const testDbPath = path.join(process.cwd(), 'backend', 'test.db');
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
      console.log('Test database cleaned up');
    }

    // Clean up authentication state
    const authPath = path.join(process.cwd(), 'tests', 'auth.json');
    if (fs.existsSync(authPath)) {
      fs.unlinkSync(authPath);
      console.log('Authentication state cleaned up');
    }

    // Clean up test artifacts
    const testResultsPath = path.join(process.cwd(), 'test-results');
    if (fs.existsSync(testResultsPath)) {
      fs.rmSync(testResultsPath, { recursive: true, force: true });
      console.log('Test artifacts cleaned up');
    }

    console.log('Global teardown completed successfully');
  } catch (error) {
    console.error('Global teardown failed:', error);
    // Don't throw - allow tests to complete even if cleanup fails
  }
}

export default globalTeardown;