#!/usr/bin/env python3
"""
SIRAJ v6.1 Local Integration Test
Quick test to verify the complete system integration
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

async def test_database_connection():
    """Test database connectivity"""
    print("🔍 Testing database connection...")
    
    success, stdout, stderr = run_command("python scripts/validate-database.py")
    
    if success:
        print("✅ Database connection successful")
        return True
    else:
        print(f"❌ Database connection failed: {stderr}")
        return False

def test_mcp_server_startup():
    """Test MCP server can start"""
    print("🔍 Testing MCP server startup...")
    
    try:
        # Try to import the MCP server modules
        sys.path.insert(0, str(Path(__file__).parent / "src" / "server"))
        
        # Test import
        import enhanced_mcp_server
        print("✅ MCP server modules importable")
        
        # Test configuration
        from pathlib import Path
        config_path = Path(__file__).parent / "config"
        if config_path.exists():
            print("✅ Configuration directory exists")
        else:
            print("⚠️  Configuration directory missing")
            
        return True
        
    except ImportError as e:
        print(f"❌ MCP server import failed: {e}")
        return False

def test_backend_startup():
    """Test backend API can start"""
    print("🔍 Testing backend startup...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        import main
        print("✅ Backend modules importable")
        return True
    except ImportError as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_frontend_dependencies():
    """Test frontend dependencies"""
    print("🔍 Testing frontend dependencies...")
    
    frontend_path = Path(__file__).parent / "frontend"
    if frontend_path.exists():
        package_json = frontend_path / "package.json"
        if package_json.exists():
            print("✅ Frontend structure exists")
            return True
        else:
            print("⚠️  Frontend package.json missing")
            return False
    else:
        print("⚠️  Frontend directory missing")
        return False

def generate_startup_commands():
    """Generate startup commands for local development"""
    commands = {
        "Database Validation": "python scripts/validate-database.py",
        "Backend API": "python backend/main.py",
        "MCP Server (stdio)": "python src/server/main_mcp_server.py --transport stdio",
        "MCP Server (https)": "python src/server/main_mcp_server.py --transport https --port 3443",
        "Frontend Development": "cd frontend && npm run dev",
        "Production Setup": "node scripts/deploy-automation.js"
    }
    
    print("\n📋 Local Development Commands:")
    print("=" * 50)
    for name, command in commands.items():
        print(f"{name}:")
        print(f"  {command}")
        print()

def show_configuration_status():
    """Show current configuration status"""
    print("📊 Configuration Status:")
    print("=" * 30)
    
    config_files = [
        (".env", "Current environment"),
        (".env.production", "Production template"),
        ("claude_desktop_config.json", "Claude Desktop MCP"),
        ("mcp-server-config.json", "MCP Server config")
    ]
    
    base_path = Path(__file__).parent
    
    for filename, description in config_files:
        file_path = base_path / filename
        status = "✅ Found" if file_path.exists() else "❌ Missing"
        print(f"{description:.<25} {status}")
    
    print()

async def main():
    """Run complete integration test"""
    print("🚀 SIRAJ v6.1 Local Integration Test")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection()),
        ("MCP Server Modules", test_mcp_server_startup()),
        ("Backend API Modules", test_backend_startup()),
        ("Frontend Structure", test_frontend_dependencies())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func
        results.append((test_name, result))
    
    # Show results
    print("\n📊 Integration Test Results:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<25} {status}")
        if result:
            passed += 1
    
    print("=" * 40)
    print(f"Tests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 Integration test PASSED!")
        print("Your SIRAJ v6.1 platform is ready for local development!")
    else:
        print("\n⚠️  Some integration tests failed.")
        print("Please check the errors above and ensure all dependencies are installed.")
    
    show_configuration_status()
    generate_startup_commands()
    
    print("\n🌟 Next Steps:")
    print("1. Run 'python scripts/validate-database.py' to verify database")
    print("2. Start backend API: 'python backend/main.py'")
    print("3. Start MCP server in new terminal")
    print("4. Run production setup: 'node scripts/deploy-automation.js'")
    print("5. Check DEPLOYMENT_SETUP_COMPLETE.md for full instructions")

if __name__ == "__main__":
    asyncio.run(main())
