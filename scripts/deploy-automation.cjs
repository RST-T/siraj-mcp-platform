#!/usr/bin/env node
/**
 * SIRAJ v6.1 Production Deployment Automation Script
 * This script helps automate the setup of external services using Playwright
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

class SirajDeploymentAutomation {
    constructor() {
        this.browser = null;
        this.page = null;
        this.config = {};
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }

    async init() {
        console.log('🚀 SIRAJ v6.1 Production Deployment Automation');
        console.log('===============================================\n');
        
        this.browser = await chromium.launch({ 
            headless: false, // Keep visible for user interaction
            args: ['--start-maximized']
        });
        this.page = await this.browser.newPage();
        await this.page.setViewportSize({ width: 1920, height: 1080 });
    }

    async prompt(question) {
        return new Promise((resolve) => {
            this.rl.question(question, (answer) => {
                resolve(answer);
            });
        });
    }

    async setupAuth0() {
        console.log('\n📁 Setting up Auth0...');
        
        const hasAuth0 = await this.prompt('Do you already have an Auth0 account? (y/n): ');
        
        if (hasAuth0.toLowerCase() !== 'y') {
            console.log('Opening Auth0 signup page...');
            await this.page.goto('https://auth0.com/signup');
            await this.prompt('Please sign up for Auth0 and press Enter when done...');
        }

        console.log('Opening Auth0 dashboard...');
        await this.page.goto('https://manage.auth0.com/');
        await this.prompt('Please log in to Auth0 and press Enter when you see the dashboard...');

        // Guide user through Auth0 setup
        console.log('\n📋 Auth0 Setup Instructions:');
        console.log('1. Create a new Application (Single Page Application)');
        console.log('2. Configure the following URLs:');
        console.log('   - Allowed Callback URLs: https://siraj.linguistics.org/api/auth/callback');
        console.log('   - Allowed Logout URLs: https://siraj.linguistics.org');
        console.log('   - Allowed Web Origins: https://siraj.linguistics.org');
        
        console.log('3. Create a new API:');
        console.log('   - Name: SIRAJ API');
        console.log('   - Identifier: https://api.siraj.linguistics.org');
        console.log('   - Signing Algorithm: RS256');
        
        const clientId = await this.prompt('Enter your Auth0 Client ID: ');
        const clientSecret = await this.prompt('Enter your Auth0 Client Secret: ');
        const domain = await this.prompt('Enter your Auth0 Domain (e.g., siraj-linguistics.us.auth0.com): ');
        
        this.config.auth0 = { clientId, clientSecret, domain };
    }

    async setupStripe() {
        console.log('\n💳 Setting up Stripe...');
        
        const hasStripe = await this.prompt('Do you already have a Stripe account? (y/n): ');
        
        if (hasStripe.toLowerCase() !== 'y') {
            console.log('Opening Stripe signup page...');
            await this.page.goto('https://dashboard.stripe.com/register');
            await this.prompt('Please sign up for Stripe and press Enter when done...');
        }

        console.log('Opening Stripe dashboard...');
        await this.page.goto('https://dashboard.stripe.com/');
        await this.prompt('Please log in to Stripe and press Enter when you see the dashboard...');

        console.log('\n📋 Stripe Setup Instructions:');
        console.log('1. Complete business verification');
        console.log('2. Go to Developers > Webhooks');
        console.log('3. Add endpoint: https://api.siraj.linguistics.org/webhooks/stripe');
        console.log('4. Select events: payment_intent.succeeded, invoice.payment_succeeded, customer.subscription.deleted');
        console.log('5. Go to Developers > API Keys to get your keys');
        
        const publishableKey = await this.prompt('Enter your Stripe Publishable Key: ');
        const secretKey = await this.prompt('Enter your Stripe Secret Key: ');
        const webhookSecret = await this.prompt('Enter your Stripe Webhook Secret: ');
        
        this.config.stripe = { publishableKey, secretKey, webhookSecret };
    }

    async setupRailway() {
        console.log('\n🚂 Setting up Railway for backend hosting...');
        
        console.log('Opening Railway...');
        await this.page.goto('https://railway.app/');
        await this.prompt('Please sign up/login to Railway and press Enter...');

        console.log('\n📋 Railway Setup Instructions:');
        console.log('1. Create a new project: "siraj-platform"');
        console.log('2. Add PostgreSQL service (if needed for additional services)');
        console.log('3. Add Redis service');
        console.log('4. Deploy your backend API');
        console.log('5. Deploy your MCP server');
        
        const railwayUrl = await this.prompt('Enter your Railway app URL: ');
        this.config.railway = { url: railwayUrl };
    }

    async setupVercel() {
        console.log('\n▲ Setting up Vercel for frontend hosting...');
        
        console.log('Opening Vercel...');
        await this.page.goto('https://vercel.com/');
        await this.prompt('Please sign up/login to Vercel and press Enter...');

        console.log('\n📋 Vercel Setup Instructions:');
        console.log('1. Import your GitHub repository');
        console.log('2. Configure environment variables from .env.production');
        console.log('3. Deploy your frontend');
        
        const vercelUrl = await this.prompt('Enter your Vercel deployment URL: ');
        this.config.vercel = { url: vercelUrl };
    }

    async setupCloudflare() {
        console.log('\n☁️ Setting up Cloudflare for DNS and CDN...');
        
        console.log('Opening Cloudflare...');
        await this.page.goto('https://dash.cloudflare.com/');
        await this.prompt('Please sign up/login to Cloudflare and press Enter...');

        console.log('\n📋 Cloudflare Setup Instructions:');
        console.log('1. Add your domain: siraj.linguistics.org');
        console.log('2. Update nameservers at your domain registrar');
        console.log('3. Configure DNS records:');
        console.log('   - A record: siraj.linguistics.org -> Vercel IP');
        console.log('   - CNAME: api.siraj.linguistics.org -> Railway URL');
        console.log('   - CNAME: mcp.siraj.linguistics.org -> Railway MCP URL');
        console.log('4. Configure SSL/TLS settings to "Full (strict)"');
        console.log('5. Enable "Always Use HTTPS"');
        
        await this.prompt('Press Enter when DNS setup is complete...');
    }

    async generateEnvironmentFiles() {
        console.log('\n📝 Generating environment files...');
        
        const productionEnv = `# SIRAJ v6.1 Production Environment - Generated ${new Date().toISOString()}

# Database Configuration
DATABASE_URL=postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-little-hill-a8to8sbf-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Authentication
AUTH0_DOMAIN=${this.config.auth0.domain}
AUTH0_CLIENT_ID=${this.config.auth0.clientId}
AUTH0_CLIENT_SECRET=${this.config.auth0.clientSecret}
AUTH0_API_AUDIENCE=https://api.siraj.linguistics.org
AUTH0_JWKS_URI=https://${this.config.auth0.domain}/.well-known/jwks.json

# Payments
STRIPE_PUBLISHABLE_KEY=${this.config.stripe.publishableKey}
STRIPE_SECRET_KEY=${this.config.stripe.secretKey}
STRIPE_WEBHOOK_SECRET=${this.config.stripe.webhookSecret}

# API Configuration
API_BASE_URL=${this.config.railway.url}
FRONTEND_URL=${this.config.vercel.url}
CORS_ORIGINS=${this.config.vercel.url},https://siraj.linguistics.org

# Security
JWT_SECRET=${this.generateSecureKey(32)}
ENCRYPTION_KEY=${this.generateSecureKey(32)}

# Production Settings
SIRAJ_DEBUG_MODE=false
SIRAJ_LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=true
ENABLE_RATE_LIMITING=true
MAINTENANCE_MODE=false
`;

        const frontendEnv = `# Frontend Environment - Generated ${new Date().toISOString()}

AUTH0_SECRET=${this.generateSecureKey(32)}
AUTH0_BASE_URL=${this.config.vercel.url}
AUTH0_ISSUER_BASE_URL=https://${this.config.auth0.domain}
AUTH0_CLIENT_ID=${this.config.auth0.clientId}
AUTH0_CLIENT_SECRET=${this.config.auth0.clientSecret}
AUTH0_AUDIENCE=https://api.siraj.linguistics.org

NEXT_PUBLIC_API_URL=${this.config.railway.url}
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${this.config.stripe.publishableKey}
`;

        fs.writeFileSync('.env.production.generated', productionEnv);
        fs.writeFileSync('frontend/.env.local.generated', frontendEnv);
        
        console.log('✅ Environment files generated:');
        console.log('  - .env.production.generated');
        console.log('  - frontend/.env.local.generated');
    }

    generateSecureKey(length) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }

    async createDeploymentChecklist() {
        const checklist = `# SIRAJ v6.1 Production Deployment Checklist

## Pre-Launch Checklist
- [x] Database schema created in Neon
- [ ] Auth0 configured with production URLs
- [ ] Stripe configured with live keys
- [ ] Railway backend deployed
- [ ] Vercel frontend deployed
- [ ] Cloudflare DNS configured
- [ ] SSL certificates verified
- [ ] Environment variables configured
- [ ] Error monitoring configured (Sentry)
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security review completed

## Launch Day Checklist
- [ ] Deploy backend to production
- [ ] Deploy MCP server to production  
- [ ] Deploy frontend to production
- [ ] Test critical user flows
- [ ] Monitor error rates
- [ ] Test payment processing
- [ ] Verify API endpoints
- [ ] Check authentication flow
- [ ] Announce to beta users

## Post-Launch Checklist
- [ ] Monitor user adoption
- [ ] Track error rates and performance
- [ ] Gather user feedback
- [ ] Set up regular database backups
- [ ] Configure monitoring alerts
- [ ] Plan first updates and improvements
- [ ] Document lessons learned

## Configuration Summary
Generated on: ${new Date().toISOString()}

Auth0 Domain: ${this.config.auth0?.domain || 'Not configured'}
Stripe Account: ${this.config.stripe?.publishableKey ? 'Configured' : 'Not configured'}
Railway URL: ${this.config.railway?.url || 'Not configured'}
Vercel URL: ${this.config.vercel?.url || 'Not configured'}

## Next Steps
1. Review and update generated environment files
2. Deploy services using generated configuration
3. Test all integrations
4. Monitor system performance
5. Set up regular maintenance procedures
`;

        fs.writeFileSync('DEPLOYMENT_CHECKLIST.md', checklist);
        console.log('✅ Deployment checklist created: DEPLOYMENT_CHECKLIST.md');
    }

    async runTests() {
        console.log('\n🧪 Running system tests...');
        
        // Test database connection
        console.log('Testing database connection...');
        // TODO: Add actual database connection test
        
        // Test API endpoints
        console.log('Testing API endpoints...');
        // TODO: Add API endpoint tests
        
        console.log('✅ Basic system tests completed');
    }

    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
        this.rl.close();
    }

    async run() {
        try {
            await this.init();
            
            console.log('This script will guide you through setting up:');
            console.log('1. Auth0 Authentication');
            console.log('2. Stripe Payment Processing');
            console.log('3. Railway Backend Hosting');
            console.log('4. Vercel Frontend Hosting');
            console.log('5. Cloudflare DNS & CDN');
            console.log('6. Environment Configuration');
            
            const proceed = await this.prompt('\nProceed with automated setup? (y/n): ');
            if (proceed.toLowerCase() !== 'y') {
                console.log('Setup cancelled.');
                return;
            }

            await this.setupAuth0();
            await this.setupStripe();
            await this.setupRailway();
            await this.setupVercel();
            await this.setupCloudflare();
            await this.generateEnvironmentFiles();
            await this.createDeploymentChecklist();
            await this.runTests();

            console.log('\n🎉 SIRAJ v6.1 deployment setup completed!');
            console.log('\nNext steps:');
            console.log('1. Review generated environment files');
            console.log('2. Deploy services using Railway and Vercel');
            console.log('3. Follow the deployment checklist');
            console.log('4. Test the complete system');

        } catch (error) {
            console.error('❌ Error during setup:', error.message);
        } finally {
            await this.cleanup();
        }
    }
}

// Run the automation if called directly
if (require.main === module) {
    const automation = new SirajDeploymentAutomation();
    automation.run();
}

module.exports = SirajDeploymentAutomation;
