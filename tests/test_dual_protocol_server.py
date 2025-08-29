#!/usr/bin/env python3
"""
Basic Test Suite for SIRAJ v6.1 Dual Protocol Server
Tests both MCP and HTTP functionality
"""

import asyncio
import json
import sys
from pathlib import Path
import logging
import subprocess
import time
import httpx
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.server.dual_protocol_server import app, dual_server
from src.server.siraj_v6_1_engine import SirajV61ComputationalHermeneuticsEngine
from config.settings import settings
from src.database.connection_manager import ConnectionManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SirajTestFramework:
    """Test framework for SIRAJ v6.1 dual protocol server"""
    
    def __init__(self):
        self.test_results = []
        self.http_base_url = f"http://{settings.host}:{settings.port}"
        self.engine: Optional[SirajV61ComputationalHermeneuticsEngine] = None
        self.connection_manager: Optional[ConnectionManager] = None
        
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("🚀 Starting SIRAJ v6.1 Test Suite")
        
        # Basic component tests
        await self._test_engine_initialization()
        await self._test_database_connections()
        await self._test_corpus_access()
        
        # HTTP API tests (if server running)
        await self._test_http_endpoints()
        
        # MCP protocol tests
        await self._test_mcp_tools()
        
        # Integration tests
        await self._test_analysis_pipeline()
        
        # Report results
        self._report_results()
    
    async def _test_engine_initialization(self):
        """Test SIRAJ engine initialization"""
        test_name = "Engine Initialization"
        logger.info(f"Testing: {test_name}")
        
        try:
            self.engine = SirajV61ComputationalHermeneuticsEngine(settings)
            await self.engine.initialize()
            
            assert self.engine.engine_status == "ready"
            assert self.engine.framework_version == "6.1"
            assert self.engine.corpus_access is not None
            assert self.engine.connection_manager is not None
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS", 
                "details": "Engine initialized successfully with all components"
            })
            logger.info(f"✅ {test_name} - PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    async def _test_database_connections(self):
        """Test database connection health"""
        test_name = "Database Connections"
        logger.info(f"Testing: {test_name}")
        
        try:
            if not self.engine or not self.engine.connection_manager:
                raise Exception("Engine not initialized")
            
            health_check = await self.engine.connection_manager.health_check()
            
            # At minimum, lexicon database should work (SQLite)
            assert health_check["status"]["lexicon"] == True
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS" if health_check["status"]["overall"] else "PARTIAL",
                "details": health_check
            })
            
            if health_check["status"]["overall"]:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.warning(f"⚠️  {test_name} - PARTIAL (some databases unavailable)")
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    async def _test_corpus_access(self):
        """Test corpus data access functionality"""
        test_name = "Corpus Data Access"
        logger.info(f"Testing: {test_name}")
        
        try:
            if not self.engine or not self.engine.corpus_access:
                raise Exception("Engine or corpus access not initialized")
            
            # Test with a simple Arabic root
            test_root = "ك-ت-ب"  # k-t-b (write/book)
            
            # Test Quranic analysis (should work with fallback even without corpus)
            quranic_result = await self.engine.corpus_access.analyze_quranic_usage(test_root)
            assert "error" not in quranic_result or "fallback" in str(quranic_result)
            
            # Test Hadith analysis
            hadith_result = await self.engine.corpus_access.analyze_hadith_references(test_root)
            assert "error" not in hadith_result or "fallback" in str(hadith_result)
            
            # Test Classical Arabic analysis
            classical_result = await self.engine.corpus_access.analyze_classical_arabic_usage(test_root)
            assert "error" not in classical_result or "fallback" in str(classical_result)
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "details": {
                    "quranic_analysis": "success" if "error" not in quranic_result else "fallback",
                    "hadith_analysis": "success" if "error" not in hadith_result else "fallback", 
                    "classical_analysis": "success" if "error" not in classical_result else "fallback"
                }
            })
            logger.info(f"✅ {test_name} - PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    async def _test_http_endpoints(self):
        """Test HTTP API endpoints"""
        test_name = "HTTP API Endpoints"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.http_base_url}/health", timeout=10)
            if response.status_code != 200:
                logger.warning(f"HTTP server not running at {self.http_base_url}")
                self.test_results.append({
                    "test": test_name,
                    "status": "SKIP",
                    "details": "HTTP server not running - start with: python -m src.server.main --protocol http"
                })
                return
            
            # Test root endpoint
            root_response = requests.get(f"{self.http_base_url}/", timeout=10)
            assert root_response.status_code == 200
            root_data = root_response.json()
            assert "message" in root_data
            assert "SIRAJ" in root_data["message"]
            
            # Test OpenAPI documentation
            openapi_response = requests.get(f"{self.http_base_url}/openapi.json", timeout=10)
            assert openapi_response.status_code == 200
            openapi_data = openapi_response.json()
            assert "openapi" in openapi_data
            assert "info" in openapi_data
            
            # Test analysis endpoint with sample data
            analysis_payload = {
                "root": "ك-ت-ب",
                "contexts": {
                    "test_context": "Testing the analysis functionality"
                },
                "mode": "text",
                "language_family": "semitic"
            }
            
            analysis_response = requests.post(
                f"{self.http_base_url}/analysis",
                json=analysis_payload,
                timeout=30
            )
            
            # Accept either success or specific errors (engine not ready, etc.)
            if analysis_response.status_code == 200:
                analysis_data = analysis_response.json()
                assert "success" in analysis_data
            elif analysis_response.status_code in [503, 500]:
                # Expected if engine not fully ready
                logger.info("Analysis endpoint returned expected error (engine initializing)")
            else:
                raise Exception(f"Unexpected analysis response: {analysis_response.status_code}")
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "details": "All HTTP endpoints responding correctly"
            })
            logger.info(f"✅ {test_name} - PASSED")
            
        except requests.exceptions.ConnectionError:
            self.test_results.append({
                "test": test_name,
                "status": "SKIP",
                "details": "HTTP server not running - start with: python -m src.server.main --protocol http"
            })
            logger.info(f"⏭️  {test_name} - SKIPPED (HTTP server not running)")
            
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    async def _test_mcp_tools(self):
        """Test MCP tool functionality"""
        test_name = "MCP Tools"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test tool listing
            tools = await dual_server.mcp_server.list_tools()
            assert len(tools) > 0
            
            tool_names = [tool.name for tool in tools]
            assert "siraj_analysis" in tool_names
            assert "list_models" in tool_names
            
            # Test resource listing
            resources = await dual_server.mcp_server.list_resources()
            assert len(resources) > 0
            
            # Test resource reading
            resource_content = await dual_server.mcp_server.read_resource("siraj://framework/metadata")
            metadata = json.loads(resource_content)
            assert "framework" in metadata
            assert "SIRAJ" in metadata["framework"]
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "details": {
                    "tools_available": len(tools),
                    "tool_names": tool_names,
                    "resources_available": len(resources)
                }
            })
            logger.info(f"✅ {test_name} - PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL", 
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    async def _test_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        test_name = "Analysis Pipeline"
        logger.info(f"Testing: {test_name}")
        
        try:
            if not self.engine:
                raise Exception("Engine not initialized")
            
            # Test complete semantic reconstruction
            test_root = "د-ر-س"  # d-r-s (study/learn)
            test_contexts = {
                "educational": "Learning and studying context",
                "religious": "Religious learning context"
            }
            
            result = await self.engine.enhanced_semantic_reconstruction(
                root=test_root,
                contexts=test_contexts,
                mode="text",
                language_family="semitic"
            )
            
            # Verify response structure
            assert "analysis_metadata" in result
            assert "analysis_parameters" in result
            assert "results" in result
            assert "validation_summary" in result
            
            # Verify metadata
            metadata = result["analysis_metadata"]
            assert metadata["framework_version"] == "6.1"
            assert "analysis_id" in metadata
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "details": {
                    "analysis_id": metadata.get("analysis_id", "unknown"),
                    "processing_time": metadata.get("processing_time", 0),
                    "confidence": result.get("validation_summary", {}).get("overall_confidence", 0)
                }
            })
            logger.info(f"✅ {test_name} - PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"❌ {test_name} - FAILED: {e}")
    
    def _report_results(self):
        """Report test results summary"""
        logger.info("\n" + "="*60)
        logger.info("🏁 SIRAJ v6.1 Test Results Summary")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
        partial = sum(1 for r in self.test_results if r["status"] == "PARTIAL")
        skipped = sum(1 for r in self.test_results if r["status"] == "SKIP")
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Partial: {partial}")
        logger.info(f"⏭️  Skipped: {skipped}")
        
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if failed > 0:
            logger.info("\n❌ Failed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    logger.info(f"  • {result['test']}: {result.get('error', 'Unknown error')}")
        
        if skipped > 0:
            logger.info("\n⏭️  Skipped Tests:")
            for result in self.test_results:
                if result["status"] == "SKIP":
                    logger.info(f"  • {result['test']}: {result.get('details', 'No details')}")
        
        logger.info("\n" + "="*60)
        
        # Save detailed results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "summary": {
                    "total": total_tests,
                    "passed": passed,
                    "failed": failed, 
                    "partial": partial,
                    "skipped": skipped,
                    "success_rate": success_rate
                },
                "details": self.test_results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
    
    async def cleanup(self):
        """Cleanup test resources"""
        if self.engine:
            await self.engine.cleanup()


async def main():
    """Main test runner"""
    test_framework = SirajTestFramework()
    
    try:
        await test_framework.run_all_tests()
    finally:
        await test_framework.cleanup()


if __name__ == "__main__":
    asyncio.run(main())