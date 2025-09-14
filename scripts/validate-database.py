#!/usr/bin/env python3
"""
SIRAJ v6.1 Database Validation Script
Validates the production database setup and performs basic functionality tests
"""

import os
import sys
import json
import asyncio
import asyncpg
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SirajDatabaseValidator:
    def __init__(self):
        self.db_url = "postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-little-hill-a8to8sbf-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
        self.conn = None
        self.test_results = {}

    async def connect(self):
        """Connect to the database"""
        try:
            self.conn = await asyncpg.connect(self.db_url)
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    async def validate_tables(self):
        """Validate that all required tables exist"""
        print("\n🔍 Validating database tables...")
        
        required_tables = [
            'users', 'api_keys', 'user_credits', 'api_usage', 
            'community_posts', 'payment_transactions', 'audit_log',
            'system_config', 'root_etymologies', 'quranic_verses',
            'classical_texts', 'hadith_collection'
        ]
        
        existing_tables = await self.conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        
        existing_table_names = [row['table_name'] for row in existing_tables]
        missing_tables = [table for table in required_tables if table not in existing_table_names]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            self.test_results['tables'] = False
        else:
            print(f"✅ All {len(required_tables)} required tables exist")
            self.test_results['tables'] = True
            
        return len(missing_tables) == 0

    async def validate_indexes(self):
        """Validate that performance indexes exist"""
        print("\n🔍 Validating database indexes...")
        
        required_indexes = [
            'idx_users_auth0_id', 'idx_users_email', 'idx_users_tier',
            'idx_api_keys_user_id', 'idx_api_usage_user_created',
            'idx_community_posts_status', 'idx_root_etymologies_family'
        ]
        
        existing_indexes = await self.conn.fetch("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' AND indexname LIKE 'idx_%'
        """)
        
        existing_index_names = [row['indexname'] for row in existing_indexes]
        missing_indexes = [idx for idx in required_indexes if idx not in existing_index_names]
        
        if missing_indexes:
            print(f"⚠️  Missing indexes: {missing_indexes}")
            self.test_results['indexes'] = False
        else:
            print(f"✅ All {len(required_indexes)} required indexes exist")
            self.test_results['indexes'] = True
            
        return len(missing_indexes) == 0

    async def validate_system_config(self):
        """Validate system configuration data"""
        print("\n🔍 Validating system configuration...")
        
        config_keys = await self.conn.fetch("""
            SELECT key, value FROM system_config
        """)
        
        required_configs = ['rate_limits', 'api_costs', 'free_monthly_credits', 'maintenance_mode']
        existing_configs = [row['key'] for row in config_keys]
        missing_configs = [cfg for cfg in required_configs if cfg not in existing_configs]
        
        if missing_configs:
            print(f"❌ Missing configurations: {missing_configs}")
            self.test_results['config'] = False
        else:
            print(f"✅ All {len(required_configs)} system configurations exist")
            for row in config_keys:
                if row['key'] == 'rate_limits':
                    limits = json.loads(row['value'])
                    print(f"   Rate limits configured for {len(limits)} tiers")
                elif row['key'] == 'api_costs':
                    costs = json.loads(row['value'])
                    print(f"   API costs configured for {len(costs)} tools")
            self.test_results['config'] = True
            
        return len(missing_configs) == 0

    async def test_user_operations(self):
        """Test basic user operations"""
        print("\n🔍 Testing user operations...")
        
        try:
            # Test user creation
            test_user_id = await self.conn.fetchval("""
                INSERT INTO users (auth0_id, email, name, tier)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, 'test_validation_user', 'validation@siraj.linguistics.org', 'Validation User', 'free')
            
            # Test credits initialization
            await self.conn.execute("""
                INSERT INTO user_credits (user_id, balance)
                VALUES ($1, $2)
            """, test_user_id, Decimal('5.00'))
            
            # Test API key creation
            api_key_id = await self.conn.fetchval("""
                INSERT INTO api_keys (user_id, key_hash, name)
                VALUES ($1, $2, $3)
                RETURNING id
            """, test_user_id, 'test_hash_123', 'Test API Key')
            
            # Test usage recording
            await self.conn.execute("""
                INSERT INTO api_usage (user_id, api_key_id, tool_name, cost)
                VALUES ($1, $2, $3, $4)
            """, test_user_id, api_key_id, 'test_tool', Decimal('0.05'))
            
            # Cleanup
            await self.conn.execute("DELETE FROM users WHERE id = $1", test_user_id)
            
            print("✅ User operations test passed")
            self.test_results['user_ops'] = True
            return True
            
        except Exception as e:
            print(f"❌ User operations test failed: {e}")
            self.test_results['user_ops'] = False
            return False

    async def test_linguistic_data(self):
        """Test linguistic data access"""
        print("\n🔍 Testing linguistic data access...")
        
        try:
            # Check if we have sample root data
            root_count = await self.conn.fetchval("""
                SELECT COUNT(*) FROM root_etymologies
            """)
            
            if root_count > 0:
                # Test root query
                sample_root = await self.conn.fetchrow("""
                    SELECT root_form, language_family, core_meaning 
                    FROM root_etymologies 
                    LIMIT 1
                """)
                
                print(f"✅ Linguistic data accessible - {root_count} roots available")
                print(f"   Sample: {sample_root['root_form']} ({sample_root['language_family']}) - {sample_root['core_meaning'][:50]}...")
                self.test_results['linguistic_data'] = True
                return True
            else:
                print("⚠️  No linguistic data found - this is normal for a fresh setup")
                self.test_results['linguistic_data'] = True
                return True
                
        except Exception as e:
            print(f"❌ Linguistic data test failed: {e}")
            self.test_results['linguistic_data'] = False
            return False

    async def test_performance(self):
        """Basic performance tests"""
        print("\n🔍 Running performance tests...")
        
        try:
            # Test query performance
            start_time = datetime.now()
            
            await self.conn.fetch("""
                SELECT u.email, uc.balance, COUNT(au.id) as api_calls
                FROM users u
                LEFT JOIN user_credits uc ON u.id = uc.user_id
                LEFT JOIN api_usage au ON u.id = au.user_id
                WHERE u.is_active = true
                GROUP BY u.id, u.email, uc.balance
                LIMIT 100
            """)
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            if query_time < 1.0:
                print(f"✅ Performance test passed - Complex query completed in {query_time:.3f}s")
                self.test_results['performance'] = True
                return True
            else:
                print(f"⚠️  Performance warning - Query took {query_time:.3f}s")
                self.test_results['performance'] = False
                return False
                
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False

    async def generate_report(self):
        """Generate validation report"""
        print("\n📊 Validation Report")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():.<30} {status}")
        
        print("=" * 50)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 Database setup validation PASSED!")
            print("Your SIRAJ v6.1 database is ready for production!")
        else:
            print("⚠️  Some tests failed. Please review and fix issues before proceeding.")
        
        # Save report to file
        report = {
            "timestamp": datetime.now().isoformat(),
            "tests": self.test_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%"
            }
        }
        
        with open('database_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📝 Detailed report saved to: database_validation_report.json")

    async def run(self):
        """Run all validation tests"""
        print("🔍 SIRAJ v6.1 Database Validation")
        print("=" * 50)
        
        if not await self.connect():
            return False
        
        try:
            await self.validate_tables()
            await self.validate_indexes()
            await self.validate_system_config()
            await self.test_user_operations()
            await self.test_linguistic_data()
            await self.test_performance()
            await self.generate_report()
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            return False
        finally:
            if self.conn:
                await self.conn.close()

async def main():
    validator = SirajDatabaseValidator()
    success = await validator.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
