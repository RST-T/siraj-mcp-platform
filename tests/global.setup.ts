import { chromium, FullConfig } from '@playwright/test';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

async function globalSetup(config: FullConfig) {
  console.log('Starting global test setup...');

  // Create test database
  const testDbPath = path.join(process.cwd(), 'backend', 'test.db');
  if (fs.existsSync(testDbPath)) {
    fs.unlinkSync(testDbPath);
  }

  try {
    // Run database migrations for test environment
    console.log('Running database migrations...');
    execSync('python -m alembic upgrade head', { 
      cwd: path.join(process.cwd(), 'backend'),
      env: { 
        ...process.env, 
        DATABASE_URL: 'sqlite:///test.db',
        ENVIRONMENT: 'test'
      },
      stdio: 'pipe'
    });

    // Seed test data
    console.log('Seeding test data...');
    execSync('python scripts/seed_test_data.py', {
      cwd: path.join(process.cwd(), 'backend'),
      env: { 
        ...process.env, 
        DATABASE_URL: 'sqlite:///test.db',
        ENVIRONMENT: 'test'
      },
      stdio: 'pipe'
    });

    // Create test user authentication state
    const browser = await chromium.launch();
    const page = await browser.newPage();
    
    // Navigate to login page and authenticate test user
    await page.goto('/api/auth/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Wait for authentication to complete
    await page.waitForURL('/dashboard');
    
    // Save authentication state
    await page.context().storageState({ 
      path: path.join(process.cwd(), 'tests', 'auth.json') 
    });
    
    await browser.close();
    
    console.log('Global setup completed successfully');
  } catch (error) {
    console.error('Global setup failed:', error);
    throw error;
  }
}

export default globalSetup;