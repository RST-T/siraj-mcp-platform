#!/usr/bin/env python3
"""
Test SIRAJ v6.1 Neon Database Connection
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database.connection_manager import ConnectionManager

async def test_siraj_neon():
    """Test SIRAJ with Neon database"""
    print("Testing SIRAJ v6.1 with Neon database...")
    
    try:
        conn_mgr = ConnectionManager()
        await conn_mgr.initialize()
        
        health = await conn_mgr.health_check()
        print("Database Health Check:")
        
        for db_type, status in health["status"].items():
            emoji = "[+]" if status else "[-]"
            detail = health["details"].get(db_type, "Unknown")
            print(f"  {emoji} {db_type.capitalize()}: {detail}")
        
        if health["status"]["corpus"]:
            print("\n[SUCCESS] SIRAJ v6.1 is ready with Neon database!")
            return True
        else:
            print("\n[WARNING] Some database connections failed")
            return False
            
    except Exception as e:
        print(f"\n[-] Test failed: {e}")
        return False
    finally:
        try:
            await conn_mgr.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = asyncio.run(test_siraj_neon())
    sys.exit(0 if success else 1)
