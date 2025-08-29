"""
SIRAJ v6.1 FastAPI Backend Service
Commercial platform backend with billing, user management, and community features
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import hashlib
import secrets

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
import jwt
from jwt.exceptions import InvalidTokenError
import httpx
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import redis
import stripe

# Initialize FastAPI app
app = FastAPI(
    title="SIRAJ v6.1 Commercial Platform API",
    description="Backend API for SIRAJ Computational Hermeneutics Commercial Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://siraj.linguistics.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://siraj:siraj@localhost:5432/siraj_platform")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# Stripe setup
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_...')

# Auth0 configuration
AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN', 'siraj.us.auth0.com')
AUTH0_API_AUDIENCE = os.getenv('AUTH0_API_AUDIENCE', 'https://api.siraj.linguistics.org')
AUTH0_ALGORITHMS = ['RS256']

# Security
security = HTTPBearer()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    auth0_id = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    tier = Column(String, default="free")  # free, paid, enterprise
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    key_hash = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class UserCredits(Base):
    __tablename__ = "user_credits"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True)
    balance = Column(Float, default=5.0)  # $5 free credits
    monthly_free_used = Column(Float, default=0.0)
    last_reset = Column(DateTime, default=datetime.utcnow)
    total_purchased = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    api_key_id = Column(UUID(as_uuid=True), nullable=False)
    tool_name = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    response_size = Column(Integer, nullable=True)

class CommunityPost(Base):
    __tablename__ = "community_posts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=False, default=list)
    moderation_status = Column(String, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    auth0_id: str
    email: EmailStr
    name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    auth0_id: str
    email: str
    name: Optional[str]
    tier: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class APIKeyCreate(BaseModel):
    name: str

class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str  # Only returned on creation
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True

class CreditsPurchase(BaseModel):
    amount: float = Field(..., ge=5.0, le=1000.0)  # $5 to $1000
    payment_method_id: str

class UsageRecord(BaseModel):
    user_id: str
    api_key_id: str
    tool_name: str
    cost: float
    timestamp: str

class CommunityPostCreate(BaseModel):
    title: str = Field(..., max_length=200)
    content: str = Field(..., max_length=5000)
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class CommunityPostResponse(BaseModel):
    id: str
    user_id: str
    title: str
    content: str
    sources: List[Dict[str, Any]]
    moderation_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Dependency functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Validate JWT token and return current user"""
    try:
        token = credentials.credentials
        
        # For development, allow API key authentication
        if token.startswith('siraj_'):
            return await authenticate_api_key(token, db)
        
        # Validate JWT token (simplified for demo)
        # In production, would verify with Auth0 JWKS
        payload = jwt.decode(
            token, 
            options={"verify_signature": False},  # Skip signature verification for demo
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_API_AUDIENCE
        )
        
        auth0_id = payload.get('sub')
        if not auth0_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.auth0_id == auth0_id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        return user
        
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

async def authenticate_api_key(api_key: str, db: Session) -> User:
    """Authenticate using API key"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    api_key_record = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()
    
    if not api_key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    # Update last used timestamp
    api_key_record.last_used = datetime.utcnow()
    db.commit()
    
    return user

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# User Management
@app.post("/api/v1/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.auth0_id == user_data.auth0_id) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    user = User(
        auth0_id=user_data.auth0_id,
        email=user_data.email,
        name=user_data.name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Initialize user credits
    credits = UserCredits(user_id=user.id, balance=5.0)
    db.add(credits)
    db.commit()
    
    return user

@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@app.get("/api/v1/users/{user_id}")
async def get_user_data(user_id: str, db: Session = Depends(get_db)):
    """Get user data for MCP server authentication"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    credits = db.query(UserCredits).filter(UserCredits.user_id == user.id).first()
    
    # Get usage counts
    today = datetime.utcnow().date()
    month_start = today.replace(day=1)
    
    calls_today = db.query(APIUsage).filter(
        APIUsage.user_id == user.id,
        APIUsage.created_at >= today
    ).count()
    
    calls_month = db.query(APIUsage).filter(
        APIUsage.user_id == user.id,
        APIUsage.created_at >= month_start
    ).count()
    
    return {
        "credits_remaining": credits.balance if credits else 0.0,
        "calls_today": calls_today,
        "calls_month": calls_month
    }

# API Key Management
@app.post("/api/v1/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    # Generate API key
    api_key = f"siraj_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Store API key
    key_record = APIKey(
        user_id=current_user.id,
        key_hash=key_hash,
        name=key_data.name
    )
    db.add(key_record)
    db.commit()
    db.refresh(key_record)
    
    # Return API key (only shown once)
    response = APIKeyResponse(
        id=str(key_record.id),
        name=key_record.name,
        key=api_key,
        created_at=key_record.created_at,
        last_used=key_record.last_used,
        is_active=key_record.is_active
    )
    
    return response

@app.get("/api/v1/api-keys")
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's API keys"""
    keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).all()
    
    return [
        {
            "id": str(key.id),
            "name": key.name,
            "created_at": key.created_at,
            "last_used": key.last_used,
            "is_active": key.is_active
        }
        for key in keys
    ]

@app.get("/api/v1/auth/api-key/{key_hash}")
async def authenticate_api_key_endpoint(key_hash: str, db: Session = Depends(get_db)):
    """Authenticate API key for MCP server"""
    api_key_record = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()
    
    if not api_key_record:
        raise HTTPException(status_code=404, detail="API key not found")
    
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    credits = db.query(UserCredits).filter(UserCredits.user_id == user.id).first()
    
    # Get rate limits based on tier
    rate_limits = {
        'free': {'daily': 50, 'monthly': 1000},
        'paid': {'daily': 500, 'monthly': 10000},
        'enterprise': {'daily': 5000, 'monthly': 100000}
    }
    
    # Get usage counts
    today = datetime.utcnow().date()
    month_start = today.replace(day=1)
    
    calls_today = db.query(APIUsage).filter(
        APIUsage.user_id == user.id,
        APIUsage.created_at >= today
    ).count()
    
    calls_month = db.query(APIUsage).filter(
        APIUsage.user_id == user.id,
        APIUsage.created_at >= month_start
    ).count()
    
    limits = rate_limits.get(user.tier, rate_limits['free'])
    
    return {
        "user_id": str(user.id),
        "api_key_id": str(api_key_record.id),
        "tier": user.tier,
        "credits_remaining": credits.balance if credits else 0.0,
        "daily_limit": limits['daily'],
        "monthly_limit": limits['monthly'],
        "calls_today": calls_today,
        "calls_month": calls_month,
        "scopes": ["read", "write"]
    }

# Billing and Credits
@app.get("/api/v1/credits")
async def get_user_credits(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's credit balance"""
    credits = db.query(UserCredits).filter(UserCredits.user_id == current_user.id).first()
    
    if not credits:
        # Initialize credits for existing users
        credits = UserCredits(user_id=current_user.id, balance=5.0)
        db.add(credits)
        db.commit()
        db.refresh(credits)
    
    return {
        "balance": credits.balance,
        "monthly_free_used": credits.monthly_free_used,
        "last_reset": credits.last_reset,
        "total_purchased": credits.total_purchased
    }

@app.post("/api/v1/credits/purchase")
async def purchase_credits(
    purchase: CreditsPurchase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Purchase additional credits"""
    try:
        # Create Stripe PaymentIntent
        intent = stripe.PaymentIntent.create(
            amount=int(purchase.amount * 100),  # Convert to cents
            currency='usd',
            payment_method=purchase.payment_method_id,
            confirmation_method='manual',
            confirm=True,
            metadata={
                'user_id': str(current_user.id),
                'credit_amount': str(purchase.amount)
            }
        )
        
        if intent.status == 'succeeded':
            # Add credits to user account
            credits = db.query(UserCredits).filter(UserCredits.user_id == current_user.id).first()
            if not credits:
                credits = UserCredits(user_id=current_user.id)
                db.add(credits)
            
            credits.balance += purchase.amount
            credits.total_purchased += purchase.amount
            db.commit()
            
            return {
                "success": True,
                "new_balance": credits.balance,
                "payment_intent_id": intent.id
            }
        else:
            return {"success": False, "error": "Payment requires additional action"}
            
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Usage Tracking
@app.post("/api/v1/usage")
async def record_usage(usage: UsageRecord, db: Session = Depends(get_db)):
    """Record API usage for billing"""
    try:
        # Parse timestamp
        timestamp = datetime.fromisoformat(usage.timestamp.replace('Z', '+00:00'))
        
        # Record usage
        usage_record = APIUsage(
            user_id=usage.user_id,
            api_key_id=usage.api_key_id,
            tool_name=usage.tool_name,
            cost=usage.cost,
            created_at=timestamp
        )
        db.add(usage_record)
        
        # Deduct credits
        credits = db.query(UserCredits).filter(UserCredits.user_id == usage.user_id).first()
        if credits:
            credits.balance = max(0, credits.balance - usage.cost)
        
        db.commit()
        
        return {"success": True, "remaining_credits": credits.balance if credits else 0}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to record usage: {str(e)}")

@app.get("/api/v1/usage")
async def get_usage_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50
):
    """Get user's usage history"""
    usage_records = db.query(APIUsage).filter(
        APIUsage.user_id == current_user.id
    ).order_by(APIUsage.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(record.id),
            "tool_name": record.tool_name,
            "cost": record.cost,
            "created_at": record.created_at,
            "processing_time": record.processing_time
        }
        for record in usage_records
    ]

# Community Platform
@app.post("/api/v1/community/posts", response_model=CommunityPostResponse)
async def create_community_post(
    post_data: CommunityPostCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new community post"""
    post = CommunityPost(
        user_id=current_user.id,
        title=post_data.title,
        content=post_data.content,
        sources=post_data.sources
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    
    return post

@app.get("/api/v1/community/posts")
async def list_community_posts(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = "approved"
):
    """List community posts"""
    query = db.query(CommunityPost)
    
    if status:
        query = query.filter(CommunityPost.moderation_status == status)
    
    posts = query.order_by(CommunityPost.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        {
            "id": str(post.id),
            "user_id": str(post.user_id),
            "title": post.title,
            "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
            "sources": post.sources,
            "moderation_status": post.moderation_status,
            "created_at": post.created_at
        }
        for post in posts
    ]

# Admin endpoints
@app.get("/api/v1/admin/users")
async def admin_list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50
):
    """Admin: List all users"""
    # TODO: Add admin role check
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.patch("/api/v1/admin/posts/{post_id}/moderate")
async def moderate_post(
    post_id: str,
    status: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin: Moderate community post"""
    # TODO: Add admin role check
    if status not in ["approved", "rejected", "pending"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    post = db.query(CommunityPost).filter(CommunityPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post.moderation_status = status
    db.commit()
    
    return {"success": True, "new_status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)