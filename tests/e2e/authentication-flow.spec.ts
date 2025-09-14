import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing authentication
    await page.context().clearCookies();
    await page.context().clearPermissions();
  });

  test('User registration flow', async ({ page }) => {
    await page.goto('/register');
    
    // Fill registration form
    await page.fill('[name="email"]', 'newuser@example.com');
    await page.fill('[name="password"]', 'SecurePassword123!');
    await page.fill('[name="confirmPassword"]', 'SecurePassword123!');
    await page.fill('[name="firstName"]', 'Test');
    await page.fill('[name="lastName"]', 'User');
    
    // Accept terms and conditions
    await page.check('[name="acceptTerms"]');
    
    // Submit registration
    await page.click('button[type="submit"]');
    
    // Should redirect to email verification or dashboard
    await expect(page).toHaveURL(/\/(verify-email|dashboard)/);
    
    // Check for success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });

  test('User login flow', async ({ page }) => {
    await page.goto('/login');
    
    // Fill login form
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    
    // Submit login
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Check user is authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-name"]')).toContainText('Test User');
  });

  test('OAuth login flow (Auth0)', async ({ page }) => {
    await page.goto('/login');
    
    // Click OAuth login button
    await page.click('[data-testid="oauth-login"]');
    
    // Should redirect to Auth0
    await expect(page).toHaveURL(/auth0\.com/);
    
    // Fill Auth0 credentials (if not using mock)
    if (!process.env.MOCK_AUTH) {
      await page.fill('[name="email"]', process.env.TEST_AUTH0_EMAIL || 'test@example.com');
      await page.fill('[name="password"]', process.env.TEST_AUTH0_PASSWORD || 'testpassword');
      await page.click('button[type="submit"]');
    }
    
    // Should redirect back to application
    await expect(page).toHaveURL('/dashboard');
    
    // Verify user is authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('Password reset flow', async ({ page }) => {
    await page.goto('/login');
    
    // Click forgot password link
    await page.click('[data-testid="forgot-password"]');
    
    // Should navigate to password reset page
    await expect(page).toHaveURL('/reset-password');
    
    // Fill email
    await page.fill('[name="email"]', 'test@example.com');
    
    // Submit reset request
    await page.click('button[type="submit"]');
    
    // Check for success message
    await expect(page.locator('[data-testid="reset-sent-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="reset-sent-message"]')).toContainText(/check your email/i);
  });

  test('User logout flow', async ({ page }) => {
    // First login
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should redirect to home page
    await expect(page).toHaveURL('/');
    
    // Check user is logged out
    await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-menu"]')).not.toBeVisible();
  });

  test('Protected route access control', async ({ page }) => {
    // Try to access protected route without authentication
    await page.goto('/dashboard');
    
    // Should redirect to login
    await expect(page).toHaveURL('/login');
    
    // Check for login required message
    await expect(page.locator('[data-testid="login-required"]')).toBeVisible();
  });

  test('API key generation after authentication', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Go to API keys page
    await page.goto('/dashboard/api-keys');
    
    // Generate new API key
    await page.click('[data-testid="generate-api-key"]');
    
    // Fill API key details
    await page.fill('[name="keyName"]', 'Test API Key');
    await page.selectOption('[name="keyType"]', 'development');
    
    // Submit
    await page.click('button[type="submit"]');
    
    // Check API key is displayed
    await expect(page.locator('[data-testid="api-key-list"]')).toContainText('Test API Key');
    await expect(page.locator('[data-testid="api-key-value"]')).toBeVisible();
    
    // Verify key format
    const apiKeyElement = page.locator('[data-testid="api-key-value"]');
    const apiKey = await apiKeyElement.textContent();
    expect(apiKey).toMatch(/^sk_[a-zA-Z0-9]{32,}$/);
  });

  test('Session management and token refresh', async ({ page, context }) => {
    // Login and store authentication
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Store authentication state
    const storageState = await context.storageState();
    
    // Simulate token expiration by modifying local storage
    await page.evaluate(() => {
      const expiredToken = JSON.stringify({
        access_token: 'expired_token',
        expires_at: Date.now() - 1000,
        refresh_token: 'refresh_token'
      });
      localStorage.setItem('auth_token', expiredToken);
    });
    
    // Make an API request that should trigger token refresh
    const response = await page.goto('/dashboard/profile');
    
    // Should not redirect to login (token should be refreshed)
    await expect(page).toHaveURL('/dashboard/profile');
    
    // Check that new token is stored
    const newToken = await page.evaluate(() => {
      return JSON.parse(localStorage.getItem('auth_token') || '{}');
    });
    
    expect(newToken.access_token).not.toBe('expired_token');
    expect(newToken.expires_at).toBeGreaterThan(Date.now());
  });

  test('User role and permission validation', async ({ page }) => {
    // Login as regular user
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Try to access admin route
    await page.goto('/admin');
    
    // Should be redirected or show access denied
    const currentUrl = page.url();
    const hasAccessDenied = await page.locator('[data-testid="access-denied"]').isVisible();
    
    expect(currentUrl.includes('/admin') && hasAccessDenied || !currentUrl.includes('/admin')).toBe(true);
  });

  test('Multi-factor authentication flow', async ({ page }) => {
    // Skip if MFA is not enabled in test environment
    test.skip(!process.env.TEST_MFA_ENABLED, 'MFA not enabled in test environment');
    
    await page.goto('/login');
    await page.fill('[name="email"]', 'mfa-user@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Should show MFA challenge
    await expect(page.locator('[data-testid="mfa-challenge"]')).toBeVisible();
    
    // Enter MFA code (use test code if available)
    await page.fill('[name="mfaCode"]', process.env.TEST_MFA_CODE || '123456');
    await page.click('button[type="submit"]');
    
    // Should complete login
    await expect(page).toHaveURL('/dashboard');
  });
});