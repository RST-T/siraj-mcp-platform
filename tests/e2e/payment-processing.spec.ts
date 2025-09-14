import { test, expect } from '@playwright/test';

test.describe('Payment Processing', () => {
  let apiKey: string;
  
  test.beforeAll(async ({ browser }) => {
    // Get API key for authenticated requests
    const context = await browser.newContext();
    const page = await context.newPage();
    
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    await page.goto('/dashboard/api-keys');
    await page.click('[data-testid="generate-api-key"]');
    await page.fill('[name="keyName"]', 'Payment Test Key');
    await page.click('button[type="submit"]');
    
    const apiKeyElement = page.locator('[data-testid="api-key-value"]');
    apiKey = (await apiKeyElement.textContent()) || '';
    
    await context.close();
  });

  test('Credit purchase flow - UI', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    // Navigate to credit purchase
    await page.goto('/dashboard/credits');
    
    // Check current balance
    const balanceElement = page.locator('[data-testid="credit-balance"]');
    const initialBalance = await balanceElement.textContent();
    
    // Select credit package
    await page.click('[data-testid="credit-package-5"]');
    
    // Proceed to payment
    await page.click('[data-testid="purchase-credits"]');
    
    // Should navigate to payment page
    await expect(page).toHaveURL('/dashboard/credits/purchase');
    
    // Verify payment form is loaded
    await expect(page.locator('[data-testid="stripe-payment-form"]')).toBeVisible();
    
    // Fill in test payment information (if in test mode)
    if (process.env.STRIPE_TEST_MODE === 'true') {
      // Use Stripe test card
      const cardElement = page.locator('[data-testid="card-number"]');
      await cardElement.fill('4242424242424242');
      
      const expiryElement = page.locator('[data-testid="card-expiry"]');
      await expiryElement.fill('12/25');
      
      const cvcElement = page.locator('[data-testid="card-cvc"]');
      await cvcElement.fill('123');
      
      const nameElement = page.locator('[data-testid="card-name"]');
      await nameElement.fill('Test User');
      
      // Submit payment
      await page.click('[data-testid="submit-payment"]');
      
      // Wait for payment processing
      await page.waitForSelector('[data-testid="payment-success"]', { timeout: 30000 });
      
      // Verify success message
      await expect(page.locator('[data-testid="payment-success"]')).toBeVisible();
      
      // Check that credits were added
      await page.goto('/dashboard/credits');
      const newBalanceElement = page.locator('[data-testid="credit-balance"]');
      const newBalance = await newBalanceElement.textContent();
      
      // Balance should be higher than initial
      expect(parseFloat(newBalance || '0')).toBeGreaterThan(parseFloat(initialBalance || '0'));
    }
  });

  test('Stripe payment intent creation', async ({ request }) => {
    const paymentIntentResponse = await request.post('/api/v1/payments/create-intent', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        amount: 5.00,
        currency: 'usd',
        credit_package: 'basic',
        payment_method_types: ['card']
      }
    });
    
    expect(paymentIntentResponse.status()).toBe(200);
    
    const paymentData = await paymentIntentResponse.json();
    expect(paymentData.client_secret).toBeDefined();
    expect(paymentData.payment_intent_id).toBeDefined();
    expect(paymentData.amount).toBe(500); // Amount in cents
    expect(paymentData.currency).toBe('usd');
    
    // Verify payment intent structure
    expect(paymentData.client_secret).toMatch(/^pi_.*_secret_/);
  });

  test('Payment confirmation and credit addition', async ({ request }) => {
    // Create payment intent first
    const paymentIntentResponse = await request.post('/api/v1/payments/create-intent', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        amount: 25.00,
        currency: 'usd',
        credit_package: 'premium'
      }
    });
    
    const paymentData = await paymentIntentResponse.json();
    const paymentIntentId = paymentData.payment_intent_id;
    
    // Get initial credit balance
    const initialCreditsResponse = await request.get('/api/v1/credits/balance', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    const initialCredits = await initialCreditsResponse.json();
    
    // Simulate successful payment (in test environment)
    if (process.env.STRIPE_TEST_MODE === 'true') {
      const confirmResponse = await request.post('/api/v1/payments/confirm', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        data: {
          payment_intent_id: paymentIntentId,
          payment_method: 'pm_card_visa' // Test payment method
        }
      });
      
      expect(confirmResponse.status()).toBe(200);
      
      const confirmData = await confirmResponse.json();
      expect(confirmData.status).toBe('succeeded');
      
      // Check that credits were added
      const newCreditsResponse = await request.get('/api/v1/credits/balance', {
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });
      const newCredits = await newCreditsResponse.json();
      
      expect(newCredits.balance).toBeGreaterThan(initialCredits.balance);
      expect(newCredits.balance).toBe(initialCredits.balance + 25.00);
    }
  });

  test('Stripe webhook handling', async ({ request }) => {
    // Test webhook endpoint exists and handles Stripe events
    const mockWebhookPayload = {
      id: 'evt_test_webhook',
      object: 'event',
      type: 'payment_intent.succeeded',
      data: {
        object: {
          id: 'pi_test_payment_intent',
          amount: 500,
          currency: 'usd',
          status: 'succeeded',
          metadata: {
            user_id: 'test_user_id',
            credit_package: 'basic'
          }
        }
      }
    };
    
    // Note: In real webhook testing, you'd use Stripe's signature
    const webhookResponse = await request.post('/webhooks/stripe', {
      headers: {
        'Content-Type': 'application/json',
        'Stripe-Signature': process.env.STRIPE_TEST_WEBHOOK_SIGNATURE || 'test_signature'
      },
      data: mockWebhookPayload
    });
    
    expect(webhookResponse.status()).toBe(200);
    
    const webhookData = await webhookResponse.json();
    expect(webhookData.received).toBe(true);
  });

  test('Payment failure handling', async ({ request }) => {
    // Create payment intent that will fail
    const paymentIntentResponse = await request.post('/api/v1/payments/create-intent', {
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
    
    const paymentData = await paymentIntentResponse.json();
    
    // Simulate payment failure (in test environment)
    if (process.env.STRIPE_TEST_MODE === 'true') {
      const failResponse = await request.post('/api/v1/payments/confirm', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        data: {
          payment_intent_id: paymentData.payment_intent_id,
          payment_method: 'pm_card_visa_chargeDeclined' // Test failing payment method
        }
      });
      
      expect(failResponse.status()).toBe(400);
      
      const failData = await failResponse.json();
      expect(failData.error).toBeDefined();
      expect(failData.error.type).toBe('card_error');
    }
  });

  test('Refund processing', async ({ request }) => {
    // Skip if not in test environment
    test.skip(process.env.STRIPE_TEST_MODE !== 'true', 'Refund tests only in test environment');
    
    // First create and confirm a payment
    const paymentIntentResponse = await request.post('/api/v1/payments/create-intent', {
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
    
    const paymentData = await paymentIntentResponse.json();
    
    const confirmResponse = await request.post('/api/v1/payments/confirm', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        payment_intent_id: paymentData.payment_intent_id,
        payment_method: 'pm_card_visa'
      }
    });
    
    expect(confirmResponse.status()).toBe(200);
    
    // Now request a refund
    const refundResponse = await request.post('/api/v1/payments/refund', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        payment_intent_id: paymentData.payment_intent_id,
        amount: 5.00,
        reason: 'requested_by_customer'
      }
    });
    
    expect(refundResponse.status()).toBe(200);
    
    const refundData = await refundResponse.json();
    expect(refundData.refund_id).toBeDefined();
    expect(refundData.status).toBe('succeeded');
    expect(refundData.amount).toBe(500); // Amount in cents
  });

  test('Payment history and receipts', async ({ request }) => {
    // Get payment history
    const historyResponse = await request.get('/api/v1/payments/history', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(historyResponse.status()).toBe(200);
    
    const historyData = await historyResponse.json();
    expect(Array.isArray(historyData.payments)).toBe(true);
    
    // Verify payment structure
    if (historyData.payments.length > 0) {
      const payment = historyData.payments[0];
      expect(payment.id).toBeDefined();
      expect(payment.amount).toBeDefined();
      expect(payment.currency).toBeDefined();
      expect(payment.status).toBeDefined();
      expect(payment.created_at).toBeDefined();
      expect(payment.receipt_url).toBeDefined();
    }
    
    // Get specific payment receipt
    if (historyData.payments.length > 0) {
      const paymentId = historyData.payments[0].id;
      const receiptResponse = await request.get(`/api/v1/payments/${paymentId}/receipt`, {
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });
      
      expect(receiptResponse.status()).toBe(200);
      
      const receiptData = await receiptResponse.json();
      expect(receiptData.receipt_url).toBeDefined();
      expect(receiptData.receipt_number).toBeDefined();
    }
  });

  test('Credit usage tracking after payment', async ({ request }) => {
    // Get initial balance
    const initialBalance = await request.get('/api/v1/credits/balance', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    const initialData = await initialBalance.json();
    
    // Use a credit by making an API call
    const toolCallResponse = await request.post('/api/v1/mcp/call', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        tool: 'computational_hermeneutics_methodology',
        arguments: {
          text: 'test',
          analysis_type: 'root_etymology',
          cultural_context: 'modern_arabic'
        }
      }
    });
    
    expect(toolCallResponse.status()).toBe(200);
    
    // Check that credit was deducted
    const newBalance = await request.get('/api/v1/credits/balance', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    const newData = await newBalance.json();
    
    expect(newData.balance).toBe(initialData.balance - 0.05);
    
    // Check credit history includes the usage
    const historyResponse = await request.get('/api/v1/credits/history', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    const historyData = await historyResponse.json();
    const recentTransaction = historyData.transactions[0];
    
    expect(recentTransaction.type).toBe('debit');
    expect(recentTransaction.amount).toBe(0.05);
    expect(recentTransaction.description).toContain('API call');
  });

  test('Subscription management (if implemented)', async ({ request }) => {
    // Skip if subscriptions not implemented
    test.skip(!process.env.FEATURE_SUBSCRIPTIONS, 'Subscriptions not implemented');
    
    // Create subscription
    const subscriptionResponse = await request.post('/api/v1/subscriptions/create', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        price_id: 'price_test_subscription',
        payment_method: 'pm_card_visa'
      }
    });
    
    expect(subscriptionResponse.status()).toBe(200);
    
    const subscriptionData = await subscriptionResponse.json();
    expect(subscriptionData.subscription_id).toBeDefined();
    expect(subscriptionData.status).toBe('active');
    
    // Get subscription status
    const statusResponse = await request.get('/api/v1/subscriptions/status', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    const statusData = await statusResponse.json();
    expect(statusData.subscription).toBeDefined();
    expect(statusData.subscription.status).toBe('active');
  });
});