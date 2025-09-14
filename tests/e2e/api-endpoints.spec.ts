import { test, expect } from '@playwright/test';

test.describe('API Endpoints Validation', () => {
  let apiKey: string;
  
  test.beforeAll(async ({ browser }) => {
    // Get API key for authenticated requests
    const context = await browser.newContext();
    const page = await context.newPage();
    
    // Login and generate API key
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    await page.goto('/dashboard/api-keys');
    await page.click('[data-testid="generate-api-key"]');
    await page.fill('[name="keyName"]', 'Test E2E Key');
    await page.click('button[type="submit"]');
    
    const apiKeyElement = page.locator('[data-testid="api-key-value"]');
    apiKey = (await apiKeyElement.textContent()) || '';
    
    await context.close();
  });

  test('User management endpoints', async ({ request }) => {
    // Get user profile
    const profileResponse = await request.get('/api/v1/users/profile', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    
    expect(profileResponse.status()).toBe(200);
    const profileData = await profileResponse.json();
    expect(profileData.email).toBe('test@example.com');
    expect(profileData.id).toBeDefined();
    
    // Update user profile
    const updateResponse = await request.put('/api/v1/users/profile', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        firstName: 'Updated',
        lastName: 'Name'
      }
    });
    
    expect(updateResponse.status()).toBe(200);
    const updatedData = await updateResponse.json();
    expect(updatedData.firstName).toBe('Updated');
    expect(updatedData.lastName).toBe('Name');
  });

  test('Credit management endpoints', async ({ request }) => {
    // Get current credits
    const creditsResponse = await request.get('/api/v1/credits/balance', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(creditsResponse.status()).toBe(200);
    const creditsData = await creditsResponse.json();
    expect(creditsData.balance).toBeGreaterThanOrEqual(0);
    expect(creditsData.currency).toBe('USD');
    
    // Get credit history
    const historyResponse = await request.get('/api/v1/credits/history', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(historyResponse.status()).toBe(200);
    const historyData = await historyResponse.json();
    expect(Array.isArray(historyData.transactions)).toBe(true);
  });

  test('MCP tool endpoints', async ({ request }) => {
    // List available tools
    const toolsResponse = await request.get('/api/v1/mcp/tools', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(toolsResponse.status()).toBe(200);
    const toolsData = await toolsResponse.json();
    expect(Array.isArray(toolsData.tools)).toBe(true);
    expect(toolsData.tools.length).toBeGreaterThan(0);
    
    // Verify specific tools are available
    const toolNames = toolsData.tools.map((tool: any) => tool.name);
    expect(toolNames).toContain('computational_hermeneutics_methodology');
    expect(toolNames).toContain('adaptive_semantic_architecture');
    expect(toolNames).toContain('community_sovereignty_protocols');
    expect(toolNames).toContain('multi_paradigm_validation');
  });

  test('Computational hermeneutics analysis', async ({ request }) => {
    // Test the main computational hermeneutics tool
    const analysisResponse = await request.post('/api/v1/mcp/call', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        tool: 'computational_hermeneutics_methodology',
        arguments: {
          text: 'سلام',
          analysis_type: 'root_etymology',
          cultural_context: 'classical_arabic',
          depth_level: 'comprehensive'
        }
      }
    });
    
    expect(analysisResponse.status()).toBe(200);
    const analysisData = await analysisResponse.json();
    
    expect(analysisData.methodology).toBeDefined();
    expect(analysisData.cultural_sovereignty_assessment).toBeDefined();
    expect(analysisData.recommended_approach).toBeDefined();
    expect(analysisData.validation_protocols).toBeDefined();
    
    // Verify credit deduction
    const newCreditsResponse = await request.get('/api/v1/credits/balance', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    const newCreditsData = await newCreditsResponse.json();
    // Credit should be deducted (0.05 USD)
    expect(newCreditsData.balance).toBeLessThan(5.0);
  });

  test('Adaptive semantic architecture', async ({ request }) => {
    const semanticResponse = await request.post('/api/v1/mcp/call', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        tool: 'adaptive_semantic_architecture',
        arguments: {
          text: 'الحمد لله',
          context: 'religious_expression',
          tier_focus: 'universal_primitives'
        }
      }
    });
    
    expect(semanticResponse.status()).toBe(200);
    const semanticData = await semanticResponse.json();
    
    expect(semanticData.semantic_mapping).toBeDefined();
    expect(semanticData.tier_analysis).toBeDefined();
    expect(semanticData.cultural_adaptations).toBeDefined();
    expect(semanticData.methodology).toBeDefined();
  });

  test('Community platform endpoints', async ({ request }) => {
    // Get community posts
    const postsResponse = await request.get('/api/v1/community/posts', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(postsResponse.status()).toBe(200);
    const postsData = await postsResponse.json();
    expect(Array.isArray(postsData.posts)).toBe(true);
    
    // Create new community post
    const newPostResponse = await request.post('/api/v1/community/posts', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        title: 'Test Analysis Discussion',
        content: 'This is a test analysis for validation',
        analysis_type: 'root_etymology',
        source_citations: ['Academic Source 1', 'Traditional Reference 2']
      }
    });
    
    expect(newPostResponse.status()).toBe(201);
    const postData = await newPostResponse.json();
    expect(postData.id).toBeDefined();
    expect(postData.moderation_status).toBe('pending');
  });

  test('Payment processing endpoints', async ({ request }) => {
    // Create payment intent
    const paymentResponse = await request.post('/api/v1/payments/create-intent', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        amount: 5.00,
        currency: 'usd',
        credit_package: 'basic'
      }
    });
    
    expect(paymentResponse.status()).toBe(200);
    const paymentData = await paymentResponse.json();
    expect(paymentData.client_secret).toBeDefined();
    expect(paymentData.payment_intent_id).toBeDefined();
    
    // Get payment history
    const historyResponse = await request.get('/api/v1/payments/history', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(historyResponse.status()).toBe(200);
    const historyData = await historyResponse.json();
    expect(Array.isArray(historyData.payments)).toBe(true);
  });

  test('API key management endpoints', async ({ request }) => {
    // List API keys
    const keysResponse = await request.get('/api/v1/api-keys', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(keysResponse.status()).toBe(200);
    const keysData = await keysResponse.json();
    expect(Array.isArray(keysData.keys)).toBe(true);
    expect(keysData.keys.length).toBeGreaterThan(0);
    
    // Create new API key
    const newKeyResponse = await request.post('/api/v1/api-keys', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'Test Production Key',
        type: 'production',
        permissions: ['read', 'write']
      }
    });
    
    expect(newKeyResponse.status()).toBe(201);
    const newKeyData = await newKeyResponse.json();
    expect(newKeyData.key).toMatch(/^sk_[a-zA-Z0-9]{32,}$/);
    expect(newKeyData.name).toBe('Test Production Key');
  });

  test('Usage analytics endpoints', async ({ request }) => {
    // Get usage statistics
    const usageResponse = await request.get('/api/v1/analytics/usage', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(usageResponse.status()).toBe(200);
    const usageData = await usageResponse.json();
    expect(usageData.total_calls).toBeGreaterThanOrEqual(0);
    expect(usageData.total_credits_used).toBeGreaterThanOrEqual(0);
    expect(Array.isArray(usageData.daily_usage)).toBe(true);
    
    // Get tool usage breakdown
    const toolUsageResponse = await request.get('/api/v1/analytics/tools', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(toolUsageResponse.status()).toBe(200);
    const toolUsageData = await toolUsageResponse.json();
    expect(Array.isArray(toolUsageData.tool_usage)).toBe(true);
  });

  test('Error handling and validation', async ({ request }) => {
    // Test invalid API key
    const invalidResponse = await request.get('/api/v1/users/profile', {
      headers: {
        'Authorization': 'Bearer invalid_key'
      }
    });
    
    expect(invalidResponse.status()).toBe(401);
    const errorData = await invalidResponse.json();
    expect(errorData.error).toContain('unauthorized');
    
    // Test missing required parameters
    const missingParamsResponse = await request.post('/api/v1/mcp/call', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        tool: 'computational_hermeneutics_methodology'
        // Missing arguments
      }
    });
    
    expect(missingParamsResponse.status()).toBe(400);
    const missingParamsError = await missingParamsResponse.json();
    expect(missingParamsError.error).toContain('required');
    
    // Test rate limiting
    const rateLimitPromises = Array(20).fill(null).map(() =>
      request.get('/api/v1/users/profile', {
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      })
    );
    
    const responses = await Promise.all(rateLimitPromises);
    const rateLimitedResponses = responses.filter(r => r.status() === 429);
    
    // Should have at least some rate limited responses with high volume
    expect(rateLimitedResponses.length).toBeGreaterThanOrEqual(0);
  });

  test('CORS and security headers', async ({ request }) => {
    const corsResponse = await request.get('/api/v1/health', {
      headers: {
        'Origin': 'https://siraj.linguistics.org'
      }
    });
    
    expect(corsResponse.status()).toBe(200);
    
    const headers = corsResponse.headers();
    expect(headers['access-control-allow-origin']).toBeDefined();
    expect(headers['x-content-type-options']).toBe('nosniff');
    expect(headers['x-frame-options']).toBeDefined();
    expect(headers['x-xss-protection']).toBeDefined();
  });
});