#!/usr/bin/env python3
"""
Test data seeding script for SIRAJ v6.1 platform
This script creates test users, API keys, and sample data for E2E testing
"""

import os
import sys
import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import secrets

# Database models
from models.user import User, APIKey, CreditTransaction
from models.community import CommunityPost, PostValidation
from models.analytics import APIUsage, UserSession
from database import Base

# Configuration
TEST_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///test.db")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TestDataSeeder:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        SessionLocal = sessionmaker(bind=self.engine)
        self.db = SessionLocal()
        
    def create_tables(self):
        """Create all database tables"""
        print("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        print("✅ Database tables created")
        
    def clear_existing_data(self):
        """Clear existing test data"""
        print("Clearing existing test data...")
        
        # Clear tables in correct order (respecting foreign key constraints)
        tables_to_clear = [
            "api_usage",
            "user_sessions", 
            "post_validations",
            "community_posts",
            "credit_transactions",
            "api_keys",
            "users"
        ]
        
        for table in tables_to_clear:
            try:
                self.db.execute(text(f"DELETE FROM {table}"))
            except Exception as e:
                print(f"Warning: Could not clear {table}: {e}")
                
        self.db.commit()
        print("✅ Existing test data cleared")
        
    def create_test_users(self):
        """Create test users with different roles and statuses"""
        print("Creating test users...")
        
        test_users = [
            {
                "email": "test@example.com",
                "password": "testpassword123",
                "first_name": "Test",
                "last_name": "User",
                "tier": "free",
                "is_verified": True,
                "credits": Decimal("5.00"),
                "role": "user"
            },
            {
                "email": "premium@example.com", 
                "password": "premiumpass123",
                "first_name": "Premium",
                "last_name": "User",
                "tier": "premium",
                "is_verified": True,
                "credits": Decimal("25.00"),
                "role": "user"
            },
            {
                "email": "admin@example.com",
                "password": "adminpass123",
                "first_name": "Admin",
                "last_name": "User",
                "tier": "enterprise",
                "is_verified": True,
                "credits": Decimal("100.00"),
                "role": "admin"
            },
            {
                "email": "scholar@example.com",
                "password": "scholarpass123",
                "first_name": "Validated",
                "last_name": "Scholar",
                "tier": "premium",
                "is_verified": True,
                "credits": Decimal("50.00"),
                "role": "scholar"
            },
            {
                "email": "unverified@example.com",
                "password": "unverifiedpass123",
                "first_name": "Unverified",
                "last_name": "User",
                "tier": "free",
                "is_verified": False,
                "credits": Decimal("5.00"),
                "role": "user"
            }
        ]
        
        created_users = []
        
        for user_data in test_users:
            # Hash password
            hashed_password = pwd_context.hash(user_data["password"])
            
            user = User(
                id=str(uuid.uuid4()),
                email=user_data["email"],
                hashed_password=hashed_password,
                first_name=user_data["first_name"],
                last_name=user_data["last_name"],
                tier=user_data["tier"],
                is_verified=user_data["is_verified"],
                credits=user_data["credits"],
                role=user_data.get("role", "user"),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(user)
            created_users.append(user)
            print(f"  Created user: {user.email} ({user.tier})")
            
        self.db.commit()
        print(f"✅ Created {len(created_users)} test users")
        return created_users
        
    def create_api_keys(self, users):
        """Create API keys for test users"""
        print("Creating API keys...")
        
        created_keys = []
        
        for user in users:
            # Create 1-2 API keys per user
            key_count = 2 if user.tier in ["premium", "enterprise"] else 1
            
            for i in range(key_count):
                key_type = "production" if i == 0 else "development"
                
                api_key = APIKey(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    key_hash=secrets.token_urlsafe(32),
                    name=f"{user.first_name}'s {key_type.title()} Key",
                    key_type=key_type,
                    permissions=["read", "write"] if key_type == "production" else ["read"],
                    is_active=True,
                    created_at=datetime.utcnow(),
                    last_used_at=datetime.utcnow() - timedelta(hours=1)
                )
                
                # Generate actual API key string
                api_key.key = f"sk_{secrets.token_urlsafe(32)}"
                
                self.db.add(api_key)
                created_keys.append(api_key)
                print(f"  Created API key: {api_key.name} for {user.email}")
                
        self.db.commit()
        print(f"✅ Created {len(created_keys)} API keys")
        return created_keys
        
    def create_credit_transactions(self, users):
        """Create sample credit transactions"""
        print("Creating credit transactions...")
        
        transactions = []
        
        for user in users:
            # Initial credit grant
            initial_grant = CreditTransaction(
                id=str(uuid.uuid4()),
                user_id=user.id,
                amount=user.credits,
                transaction_type="credit",
                description="Initial free credits",
                created_at=datetime.utcnow() - timedelta(days=7)
            )
            transactions.append(initial_grant)
            
            # Some usage transactions
            if user.tier != "free":
                for i in range(5):
                    usage = CreditTransaction(
                        id=str(uuid.uuid4()),
                        user_id=user.id,
                        amount=Decimal("0.05"),
                        transaction_type="debit",
                        description="API call - computational_hermeneutics_methodology",
                        api_call_id=str(uuid.uuid4()),
                        created_at=datetime.utcnow() - timedelta(days=i+1, hours=i)
                    )
                    transactions.append(usage)
                    
            # Premium users have purchase transactions
            if user.tier in ["premium", "enterprise"]:
                purchase = CreditTransaction(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    amount=Decimal("25.00") if user.tier == "premium" else Decimal("100.00"),
                    transaction_type="credit",
                    description="Credit purchase via Stripe",
                    stripe_payment_intent_id=f"pi_test_{secrets.token_urlsafe(16)}",
                    created_at=datetime.utcnow() - timedelta(days=5)
                )
                transactions.append(purchase)
                
        for transaction in transactions:
            self.db.add(transaction)
            
        self.db.commit()
        print(f"✅ Created {len(transactions)} credit transactions")
        
    def create_community_posts(self, users):
        """Create sample community posts"""
        print("Creating community posts...")
        
        sample_posts = [
            {
                "title": "Analysis of the root ص-ل-ح in Quranic context",
                "content": "I analyzed the tri-literal root ص-ل-ح (s-l-h) which appears frequently in the Quran. My computational analysis suggests connections to concepts of righteousness and reform.",
                "analysis_type": "root_etymology",
                "analysis_data": {
                    "root": "s-l-h",
                    "meaning": "righteousness, reform, correctness",
                    "occurrences": 180,
                    "semantic_field": "moral_action"
                },
                "source_citations": [
                    "Lane's Arabic-English Lexicon",
                    "Al-Mufradat by Raghib al-Isfahani",
                    "Quranic Corpus Database"
                ],
                "moderation_status": "approved"
            },
            {
                "title": "Semantic analysis of peace terminology across Semitic languages",
                "content": "Comparative analysis of peace-related terms: Arabic سلام (salam), Hebrew שלום (shalom), Aramaic שלמא (shlama). All derive from the tri-literal root ش-ل-م/ס-ל-ם.",
                "analysis_type": "comparative_linguistics",
                "analysis_data": {
                    "languages": ["Arabic", "Hebrew", "Aramaic"],
                    "root_variants": ["س-ل-م", "ס-ל-ם"],
                    "semantic_core": "wholeness, completeness, peace"
                },
                "source_citations": [
                    "Comparative Semitic Linguistics by Lipiński",
                    "A Grammar of Biblical Hebrew by Joüon-Muraoka"
                ],
                "moderation_status": "approved"
            },
            {
                "title": "Neural network analysis of Arabic morphological patterns",
                "content": "Applied transformer models to analyze morphological patterns in classical Arabic texts. Discovered interesting correlations between root semantics and morphological variations.",
                "analysis_type": "computational_analysis",
                "analysis_data": {
                    "model": "BERT-Arabic",
                    "dataset_size": 50000,
                    "accuracy": 0.89,
                    "patterns_discovered": 127
                },
                "source_citations": [
                    "AraBERT: Transformer-based Model for Arabic Language Understanding",
                    "Computational Morphology of Arabic - Beesley & Karttunen"
                ],
                "moderation_status": "pending"
            }
        ]
        
        posts = []
        scholars = [user for user in users if user.role in ["scholar", "admin"]]
        
        for i, post_data in enumerate(sample_posts):
            author = scholars[i % len(scholars)]
            
            post = CommunityPost(
                id=str(uuid.uuid4()),
                author_id=author.id,
                title=post_data["title"],
                content=post_data["content"],
                analysis_type=post_data["analysis_type"],
                analysis_data=post_data["analysis_data"],
                source_citations=post_data["source_citations"],
                moderation_status=post_data["moderation_status"],
                created_at=datetime.utcnow() - timedelta(days=i+1),
                updated_at=datetime.utcnow() - timedelta(days=i+1)
            )
            
            posts.append(post)
            self.db.add(post)
            print(f"  Created post: {post.title}")
            
        self.db.commit()
        print(f"✅ Created {len(posts)} community posts")
        return posts
        
    def create_post_validations(self, posts, users):
        """Create sample post validations"""
        print("Creating post validations...")
        
        validations = []
        scholars = [user for user in users if user.role in ["scholar", "admin"]]
        
        for post in posts:
            if post.moderation_status == "approved":
                # Create 2-3 validations per approved post
                for i in range(2):
                    validator = scholars[i % len(scholars)]
                    if validator.id != post.author_id:  # Don't validate own posts
                        
                        validation = PostValidation(
                            id=str(uuid.uuid4()),
                            post_id=post.id,
                            validator_id=validator.id,
                            validation_type="peer_review",
                            status="approved",
                            comments="Methodology is sound and sources are properly cited.",
                            traditional_score=0.85,
                            scientific_score=0.78,
                            computational_score=0.92,
                            overall_score=0.85,
                            created_at=datetime.utcnow() - timedelta(hours=i*6)
                        )
                        
                        validations.append(validation)
                        self.db.add(validation)
                        
        self.db.commit()
        print(f"✅ Created {len(validations)} post validations")
        
    def create_api_usage_data(self, users, api_keys):
        """Create sample API usage data"""
        print("Creating API usage data...")
        
        usage_records = []
        
        mcp_tools = [
            "computational_hermeneutics_methodology",
            "adaptive_semantic_architecture", 
            "community_sovereignty_protocols",
            "multi_paradigm_validation"
        ]
        
        for api_key in api_keys:
            user = next(user for user in users if user.id == api_key.user_id)
            
            # Create usage records for the past week
            for day in range(7):
                calls_per_day = 5 if user.tier == "free" else (20 if user.tier == "premium" else 50)
                
                for call in range(calls_per_day):
                    tool_name = mcp_tools[call % len(mcp_tools)]
                    
                    usage = APIUsage(
                        id=str(uuid.uuid4()),
                        user_id=user.id,
                        api_key_id=api_key.id,
                        tool_name=tool_name,
                        cost=Decimal("0.05"),
                        response_time_ms=150 + (call * 10),  # Simulate variable response times
                        status_code=200,
                        created_at=datetime.utcnow() - timedelta(days=day, hours=call)
                    )
                    
                    usage_records.append(usage)
                    self.db.add(usage)
                    
        self.db.commit()
        print(f"✅ Created {len(usage_records)} API usage records")
        
    def create_user_sessions(self, users):
        """Create sample user sessions"""
        print("Creating user sessions...")
        
        sessions = []
        
        for user in users:
            if user.is_verified:  # Only verified users have sessions
                # Create 3-5 sessions over the past week
                session_count = 3 if user.tier == "free" else 5
                
                for i in range(session_count):
                    session = UserSession(
                        id=str(uuid.uuid4()),
                        user_id=user.id,
                        session_token=secrets.token_urlsafe(32),
                        ip_address=f"192.168.1.{100 + i}",
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        created_at=datetime.utcnow() - timedelta(days=i+1),
                        expires_at=datetime.utcnow() + timedelta(days=30-i),
                        is_active=True
                    )
                    
                    sessions.append(session)
                    self.db.add(session)
                    
        self.db.commit()
        print(f"✅ Created {len(sessions)} user sessions")
        
    def verify_data_integrity(self):
        """Verify that all test data was created correctly"""
        print("Verifying data integrity...")
        
        # Count records
        user_count = self.db.query(User).count()
        api_key_count = self.db.query(APIKey).count() 
        transaction_count = self.db.query(CreditTransaction).count()
        post_count = self.db.query(CommunityPost).count()
        
        print(f"  Users: {user_count}")
        print(f"  API Keys: {api_key_count}")
        print(f"  Credit Transactions: {transaction_count}")
        print(f"  Community Posts: {post_count}")
        
        # Verify relationships
        users_with_keys = self.db.execute(
            text("SELECT COUNT(DISTINCT user_id) FROM api_keys")
        ).scalar()
        
        users_with_transactions = self.db.execute(
            text("SELECT COUNT(DISTINCT user_id) FROM credit_transactions")
        ).scalar()
        
        print(f"  Users with API keys: {users_with_keys}")
        print(f"  Users with transactions: {users_with_transactions}")
        
        # Check for a specific test user
        test_user = self.db.query(User).filter(User.email == "test@example.com").first()
        if test_user:
            print(f"  Test user found: {test_user.email} (ID: {test_user.id})")
            
            # Get their API key
            api_key = self.db.query(APIKey).filter(APIKey.user_id == test_user.id).first()
            if api_key:
                print(f"  Test user API key: {api_key.key[:20]}...")
            else:
                print("  ⚠️  Test user has no API key")
        else:
            print("  ❌ Test user not found")
            
        print("✅ Data integrity check completed")
        
    def close(self):
        """Close database connection"""
        self.db.close()
        
def main():
    """Main function to seed test data"""
    print("SIRAJ v6.1 Test Data Seeder")
    print("===========================")
    
    try:
        seeder = TestDataSeeder(TEST_DATABASE_URL)
        
        # Create tables
        seeder.create_tables()
        
        # Clear existing data
        seeder.clear_existing_data()
        
        # Create test data
        users = seeder.create_test_users()
        api_keys = seeder.create_api_keys(users)
        seeder.create_credit_transactions(users)
        posts = seeder.create_community_posts(users)
        seeder.create_post_validations(posts, users)
        seeder.create_api_usage_data(users, api_keys)
        seeder.create_user_sessions(users)
        
        # Verify data
        seeder.verify_data_integrity()
        
        seeder.close()
        
        print("\n✅ Test data seeding completed successfully!")
        print("\nTest Users:")
        print("- test@example.com (password: testpassword123)")
        print("- premium@example.com (password: premiumpass123)")
        print("- admin@example.com (password: adminpass123)")
        print("- scholar@example.com (password: scholarpass123)")
        print("- unverified@example.com (password: unverifiedpass123)")
        
    except Exception as e:
        print(f"❌ Error seeding test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()