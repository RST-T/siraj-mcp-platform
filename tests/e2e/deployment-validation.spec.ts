import { test, expect } from '@playwright/test';

test.describe('Deployment Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Set longer timeout for deployment checks
    test.setTimeout(60000);
  });

  test('Frontend deployment health check', async ({ page }) => {
    await page.goto('/');
    
    // Check page loads successfully
    await expect(page).toHaveTitle(/SIRAJ/);
    
    // Check critical elements are present
    await expect(page.locator('nav')).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
    
    // Check for error messages
    const errorElements = page.locator('[data-testid="error"]');
    await expect(errorElements).toHaveCount(0);
    
    // Check network requests complete successfully
    const response = await page.waitForResponse('/api/health');
    expect(response.status()).toBe(200);
  });

  test('Backend API health check', async ({ request }) => {
    const healthResponse = await request.get('/api/health');
    expect(healthResponse.status()).toBe(200);
    
    const healthData = await healthResponse.json();
    expect(healthData.status).toBe('healthy');
    expect(healthData.database).toBe('connected');
    expect(healthData.redis).toBe('connected');
  });

  test('MCP Server connectivity', async ({ request }) => {
    const mcpResponse = await request.post('/api/v1/mcp/health', {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.TEST_API_KEY}`
      }
    });
    
    expect(mcpResponse.status()).toBe(200);
    
    const mcpData = await mcpResponse.json();
    expect(mcpData.server_status).toBe('running');
    expect(mcpData.tools_available).toBeGreaterThan(0);
  });

  test('Database connectivity', async ({ request }) => {
    const dbResponse = await request.get('/api/v1/database/status', {
      headers: {
        'Authorization': `Bearer ${process.env.TEST_API_KEY}`
      }
    });
    
    expect(dbResponse.status()).toBe(200);
    
    const dbData = await dbResponse.json();
    expect(dbData.connection).toBe('active');
    expect(dbData.pool_size).toBeGreaterThan(0);
    expect(dbData.active_connections).toBeGreaterThanOrEqual(0);
  });

  test('Redis connectivity', async ({ request }) => {
    const redisResponse = await request.get('/api/v1/redis/status', {
      headers: {
        'Authorization': `Bearer ${process.env.TEST_API_KEY}`
      }
    });
    
    expect(redisResponse.status()).toBe(200);
    
    const redisData = await redisResponse.json();
    expect(redisData.connection).toBe('active');
    expect(redisData.memory_usage).toBeDefined();
  });

  test('SSL/HTTPS configuration', async ({ page, context }) => {
    if (process.env.BASE_URL?.startsWith('https://')) {
      // Check SSL certificate validity
      await page.goto('/');
      
      const securityState = await context.request.get('/', {
        ignoreHTTPSErrors: false
      });
      
      expect(securityState.status()).toBe(200);
      
      // Verify security headers
      const headers = securityState.headers();
      expect(headers['strict-transport-security']).toBeDefined();
      expect(headers['x-content-type-options']).toBe('nosniff');
      expect(headers['x-frame-options']).toBeDefined();
    }
  });

  test('Environment variables validation', async ({ request }) => {
    const configResponse = await request.get('/api/v1/config/validate', {
      headers: {
        'Authorization': `Bearer ${process.env.TEST_API_KEY}`
      }
    });
    
    expect(configResponse.status()).toBe(200);
    
    const configData = await configResponse.json();
    expect(configData.auth0_configured).toBe(true);
    expect(configData.stripe_configured).toBe(true);
    expect(configData.redis_configured).toBe(true);
    expect(configData.database_configured).toBe(true);
  });

  test('Performance benchmarks', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;
    
    // Page should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
    
    // Check Core Web Vitals
    const metrics = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const metrics = entries.reduce((acc: any, entry) => {
            acc[entry.name] = entry.value;
            return acc;
          }, {});
          resolve(metrics);
        }).observe({ entryTypes: ['measure'] });
      });
    });
    
    // Add performance assertions based on your requirements
    console.log('Performance metrics:', metrics);
  });

  test('Error monitoring setup', async ({ request }) => {
    // Test that error monitoring is properly configured
    const errorResponse = await request.get('/api/v1/monitoring/status', {
      headers: {
        'Authorization': `Bearer ${process.env.TEST_API_KEY}`
      }
    });
    
    expect(errorResponse.status()).toBe(200);
    
    const monitoringData = await errorResponse.json();
    expect(monitoringData.sentry_configured).toBe(true);
    expect(monitoringData.logging_configured).toBe(true);
    expect(monitoringData.metrics_configured).toBe(true);
  });
});