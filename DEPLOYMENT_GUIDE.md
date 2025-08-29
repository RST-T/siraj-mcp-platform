# SIRAJ v6.1 Commercial Platform - Production Deployment Guide

This guide provides step-by-step instructions for deploying the SIRAJ v6.1 Computational Hermeneutics commercial platform to production.

## Overview

The SIRAJ v6.1 platform consists of:
- **Enhanced MCP Server** (Python) - OAuth Resource Server with usage tracking
- **Backend API** (FastAPI) - User management, billing, community platform
- **Frontend Web App** (Next.js) - User interface and dashboard
- **Database** (PostgreSQL) - User data, billing, community content
- **Cache** (Redis) - Session management, rate limiting
- **Security Layer** - Comprehensive security and monitoring

## Prerequisites

### Required Accounts
- [ ] **Auth0 Account** - Authentication provider
- [ ] **Stripe Account** - Payment processing
- [ ] **Railway Account** - Backend hosting
- [ ] **Vercel Account** - Frontend hosting
- [ ] **Cloudflare Account** - CDN and domain management
- [ ] **GitHub Account** - Code repository and CI/CD

### Development Tools
- [ ] Node.js 18+ and npm
- [ ] Python 3.11+ and pip
- [ ] Docker and Docker Compose
- [ ] Git
- [ ] Railway CLI
- [ ] Vercel CLI

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/siraj-team/siraj-mcp-platform
cd siraj-mcp-platform
```

### 2. Environment Variables

Create environment files for each component:

#### Backend (.env)
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/siraj_platform
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Authentication
AUTH0_DOMAIN=your-domain.us.auth0.com
AUTH0_API_AUDIENCE=https://api.siraj.linguistics.org
AUTH0_JWKS_URI=https://your-domain.us.auth0.com/.well-known/jwks.json

# Payments
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Security
ENCRYPTION_KEY=your-32-byte-base64-key
JWT_SECRET=your-jwt-secret

# API Configuration
BILLING_API_URL=https://api.siraj.linguistics.org/api/v1
CORS_ORIGINS=https://siraj.linguistics.org,https://www.siraj.linguistics.org

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

#### Frontend (.env.local)
```bash
# Auth0 Configuration
AUTH0_SECRET=your-auth0-secret
AUTH0_BASE_URL=https://siraj.linguistics.org
AUTH0_ISSUER_BASE_URL=https://your-domain.us.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret
AUTH0_AUDIENCE=https://api.siraj.linguistics.org

# API Configuration
NEXT_PUBLIC_API_URL=https://api.siraj.linguistics.org
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...

# Analytics
NEXT_PUBLIC_GOOGLE_ANALYTICS=GA_MEASUREMENT_ID
```

#### MCP Server (.env)
```bash
# Redis Configuration
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# API Configuration
BILLING_API_URL=https://api.siraj.linguistics.org/api/v1
BILLING_API_KEY=your-billing-api-key

# Auth0 Configuration
AUTH0_DOMAIN=https://your-domain.us.auth0.com/
AUTH0_API_AUDIENCE=https://api.siraj.linguistics.org
AUTH0_JWKS_URI=https://your-domain.us.auth0.com/.well-known/jwks.json
```

## Infrastructure Setup

### 1. Auth0 Configuration

#### Create Auth0 Application
1. Go to Auth0 Dashboard > Applications
2. Create new "Single Page Application" for frontend
3. Create new "Machine to Machine" application for backend
4. Configure allowed URLs:
   - **Allowed Callback URLs**: `https://siraj.linguistics.org/api/auth/callback`
   - **Allowed Logout URLs**: `https://siraj.linguistics.org`
   - **Allowed Web Origins**: `https://siraj.linguistics.org`

#### Create Auth0 API
1. Go to Auth0 Dashboard > APIs
2. Create new API with identifier: `https://api.siraj.linguistics.org`
3. Enable RBAC and add custom claims
4. Create scopes: `read:profile`, `write:profile`, `read:analytics`, `admin:moderate`

#### Configure Resource Server
```json
{
  "identifier": "https://api.siraj.linguistics.org",
  "name": "SIRAJ API",
  "scopes": [
    {
      "value": "read:profile",
      "description": "Read user profile"
    },
    {
      "value": "write:profile", 
      "description": "Modify user profile"
    },
    {
      "value": "admin:moderate",
      "description": "Moderate community content"
    }
  ],
  "signing_alg": "RS256",
  "token_lifetime": 86400,
  "skip_consent_for_verifiable_first_party_clients": true
}
```

### 2. Stripe Configuration

#### Create Stripe Account
1. Sign up for Stripe account
2. Complete business verification
3. Configure webhook endpoints:
   - `https://api.siraj.linguistics.org/webhooks/stripe`
   - Events: `payment_intent.succeeded`, `invoice.payment_succeeded`, `customer.subscription.deleted`

#### Create Products and Prices
```bash
# Credit packages
stripe products create --name "SIRAJ Credits - $5" --description "5 dollars in SIRAJ computational hermeneutics credits"
stripe prices create --product prod_xxx --unit-amount 500 --currency usd

stripe products create --name "SIRAJ Credits - $25" --description "25 dollars in SIRAJ computational hermeneutics credits"
stripe prices create --product prod_xxx --unit-amount 2500 --currency usd

stripe products create --name "SIRAJ Credits - $100" --description "100 dollars in SIRAJ computational hermeneutics credits"
stripe prices create --product prod_xxx --unit-amount 10000 --currency usd
```

### 3. Railway Backend Deployment

#### Setup Railway Project
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and create project
railway login
railway init siraj-platform

# Create services
railway service create --name api
railway service create --name mcp-server
railway service create --name redis

# Add PostgreSQL
railway add --service postgresql
```

#### Deploy Backend API
```bash
cd backend

# Create railway.json
cat > railway.json << EOF
{
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "on-failure"
  }
}
EOF

# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Deploy
railway deploy --service api
```

#### Deploy MCP Server
```bash
cd ../src/server

# Create railway.json for MCP server
cat > railway.json << EOF
{
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "startCommand": "python enhanced_mcp_server.py"
  }
}
EOF

# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../../config ./config
COPY ../../src ./src

CMD ["python", "enhanced_mcp_server.py"]
EOF

# Deploy
railway deploy --service mcp-server
```

### 4. Vercel Frontend Deployment

#### Setup Vercel Project
```bash
cd frontend

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod

# Configure environment variables in Vercel dashboard
# Copy all variables from .env.local
```

#### Configure Vercel Settings
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "routes": [
    {
      "src": "/api/auth/(.*)",
      "dest": "/api/auth/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/$1"
    }
  ],
  "env": {
    "AUTH0_SECRET": "@auth0-secret",
    "AUTH0_BASE_URL": "https://siraj.linguistics.org",
    "AUTH0_ISSUER_BASE_URL": "@auth0-issuer-base-url",
    "AUTH0_CLIENT_ID": "@auth0-client-id",
    "AUTH0_CLIENT_SECRET": "@auth0-client-secret"
  }
}
```

### 5. Database Setup

#### Run Migrations
```bash
# Connect to Railway PostgreSQL
railway connect postgresql

# Run migrations
cd backend
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_data.py
```

#### Database Backup Strategy
```bash
# Create backup script
cat > scripts/backup.sh << EOF
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
DB_URL=$DATABASE_URL
BACKUP_PATH="backups/siraj_platform_$DATE.sql"

pg_dump $DB_URL > $BACKUP_PATH
gzip $BACKUP_PATH

# Upload to S3 or similar
aws s3 cp $BACKUP_PATH.gz s3://siraj-backups/
EOF

chmod +x scripts/backup.sh

# Schedule with cron (daily backups)
echo "0 2 * * * /path/to/scripts/backup.sh" | crontab -
```

## Domain and SSL Configuration

### 1. Cloudflare Setup
```bash
# Add domain to Cloudflare
# Configure DNS records:

# Frontend (A record)
siraj.linguistics.org -> Vercel IP
www.siraj.linguistics.org -> Vercel IP

# Backend (CNAME record)  
api.siraj.linguistics.org -> your-railway-domain.up.railway.app

# MCP Server (CNAME record)
mcp.siraj.linguistics.org -> your-mcp-railway-domain.up.railway.app
```

### 2. SSL Configuration
```bash
# Cloudflare SSL settings:
# - SSL/TLS encryption mode: Full (strict)
# - Always Use HTTPS: On
# - HTTP Strict Transport Security: On
# - Minimum TLS Version: 1.2
```

## Monitoring and Logging

### 1. Application Monitoring
```bash
# Install Sentry for error tracking
pip install sentry-sdk[fastapi]
npm install @sentry/nextjs

# Configure Sentry in backend
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment="production"
)
```

### 2. Infrastructure Monitoring
```bash
# Railway monitoring
railway logs --service api --tail
railway logs --service mcp-server --tail

# Vercel monitoring
vercel logs --follow

# Set up alerts for:
# - High error rates (>5%)
# - High response times (>2s)
# - High memory usage (>80%)
# - Database connection issues
```

### 3. Business Metrics
```python
# Track key metrics
import analytics

def track_api_usage(user_id: str, tool: str, cost: float):
    analytics.track(user_id, 'API Call', {
        'tool': tool,
        'cost': cost,
        'timestamp': datetime.utcnow().isoformat()
    })

def track_credit_purchase(user_id: str, amount: float):
    analytics.track(user_id, 'Credit Purchase', {
        'amount': amount,
        'timestamp': datetime.utcnow().isoformat()
    })
```

## Security Hardening

### 1. Environment Security
```bash
# Rotate secrets regularly
railway env set AUTH0_CLIENT_SECRET=new-secret
vercel env add AUTH0_CLIENT_SECRET production

# Enable 2FA on all accounts
# - GitHub
# - Railway  
# - Vercel
# - Auth0
# - Stripe
# - Cloudflare
```

### 2. Database Security
```sql
-- Create read-only user for analytics
CREATE USER analytics_reader WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE siraj_platform TO analytics_reader;
GRANT USAGE ON SCHEMA public TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_reader;

-- Audit logging
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_data JSONB,
    new_data JSONB,
    user_id UUID,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### 3. API Security
```python
# Rate limiting configuration
RATE_LIMITS = {
    'free': {'daily': 50, 'monthly': 1000},
    'paid': {'daily': 500, 'monthly': 10000}, 
    'enterprise': {'daily': 5000, 'monthly': 100000}
}

# Content validation
def validate_request_content(content: str) -> bool:
    # Check for malicious content
    # Validate against schema
    # Check cultural sensitivity
    pass
```

## Backup and Recovery

### 1. Database Backups
```bash
# Automated daily backups
#!/bin/bash
DATE=$(date +%Y%m%d)
railway run pg_dump $DATABASE_URL | gzip > "backup_$DATE.sql.gz"
aws s3 cp "backup_$DATE.sql.gz" s3://siraj-backups/
```

### 2. Application Backups
```bash
# Code repository (GitHub)
# Configuration (documented and version controlled)
# Secrets (encrypted backup in secure location)
```

### 3. Recovery Procedures
```bash
# Database recovery
gunzip backup_20241201.sql.gz
railway run psql $DATABASE_URL < backup_20241201.sql

# Application recovery
git clone https://github.com/siraj-team/siraj-mcp-platform
# Follow deployment steps
```

## Performance Optimization

### 1. Database Optimization
```sql
-- Indexes for common queries
CREATE INDEX idx_api_usage_user_created ON api_usage(user_id, created_at);
CREATE INDEX idx_community_posts_status ON community_posts(moderation_status);
CREATE INDEX idx_users_tier ON users(tier);

-- Connection pooling
-- Configure in DATABASE_URL: ?pool_size=20&max_overflow=0
```

### 2. Caching Strategy
```python
# Redis caching for expensive operations
@cache.memoize(timeout=300)  # 5 minutes
def get_user_analytics(user_id: str):
    # Expensive analytics query
    pass

# CDN caching for static assets
# Configure Cloudflare Page Rules:
# *.js, *.css, *.png, *.jpg -> Cache Everything, Edge TTL: 1 month
```

### 3. API Optimization
```python
# Database query optimization
def get_community_posts(limit: int = 20, offset: int = 0):
    return db.query(CommunityPost)\
        .options(joinedload(CommunityPost.user))\
        .filter(CommunityPost.moderation_status == 'approved')\
        .order_by(CommunityPost.created_at.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()
```

## Testing in Production

### 1. Smoke Tests
```bash
#!/bin/bash
# Test all critical endpoints

# Health checks
curl -f https://api.siraj.linguistics.org/health
curl -f https://siraj.linguistics.org

# Authentication flow
curl -X POST https://api.siraj.linguistics.org/api/v1/test-auth \
  -H "Authorization: Bearer $TEST_TOKEN"

# Payment processing (test mode)
curl -X POST https://api.siraj.linguistics.org/api/v1/test-payment \
  -H "Content-Type: application/json" \
  -d '{"amount": 5.00, "test": true}'
```

### 2. Load Testing
```javascript
// k6 load test
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 0 }
  ]
};

export default function () {
  let response = http.get('https://api.siraj.linguistics.org/health');
  check(response, { 'status was 200': (r) => r.status == 200 });
}
```

## Launch Checklist

### Pre-Launch
- [ ] All environment variables configured
- [ ] Database migrations completed
- [ ] SSL certificates installed
- [ ] DNS records configured
- [ ] Auth0 configured with production URLs
- [ ] Stripe configured with live keys
- [ ] Error monitoring configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security review completed

### Launch Day
- [ ] Deploy backend to production
- [ ] Deploy MCP server to production  
- [ ] Deploy frontend to production
- [ ] Run database migrations
- [ ] Test critical user flows
- [ ] Monitor error rates
- [ ] Announce to beta users
- [ ] Monitor performance metrics

### Post-Launch
- [ ] Monitor user adoption
- [ ] Track error rates and performance
- [ ] Gather user feedback
- [ ] Plan first updates and improvements
- [ ] Document lessons learned

## Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Check Auth0 configuration
curl https://your-domain.us.auth0.com/.well-known/jwks.json

# Verify API audience and issuer
# Check token expiration and scopes
```

#### Database Connection Issues
```bash
# Test database connectivity
railway run psql $DATABASE_URL -c "SELECT version();"

# Check connection pool status
# Monitor active connections
```

#### Payment Processing Issues
```bash
# Test Stripe webhook
stripe listen --forward-to https://api.siraj.linguistics.org/webhooks/stripe

# Verify webhook signatures
# Check payment intent status
```

#### Performance Issues
```bash
# Check API response times
curl -w "@curl-format.txt" -s https://api.siraj.linguistics.org/health

# Monitor database queries
# Check Redis memory usage
# Analyze CDN cache hit rates
```

### Support Contacts
- **Technical Issues**: tech-support@siraj.linguistics.org
- **Security Issues**: security@siraj.linguistics.org
- **Business Issues**: business@siraj.linguistics.org

## Scaling Considerations

### Horizontal Scaling
- Database read replicas for analytics
- Multiple Railway service instances
- CDN edge locations worldwide
- Redis cluster for cache scaling

### Vertical Scaling
- Upgrade Railway service tiers
- Optimize database queries
- Implement caching layers
- Use connection pooling

This deployment guide provides a complete production setup for the SIRAJ v6.1 commercial platform. Follow each step carefully and test thoroughly before launching to ensure a smooth production deployment.