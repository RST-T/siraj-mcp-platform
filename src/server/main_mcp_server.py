#!/usr/bin/env python3
"""
SIRAJ v6.1 MCP Server with Proper stdio Transport Protocol
Implements JSON-RPC 2.0 over stdio for Claude Desktop integration
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolRequest

# SIRAJ components
from src.server.adaptive_semantic_architecture import AdaptiveSemanticArchitecture
from src.server.community_validation_interface import CommunityValidationInterface
from src.server.multi_paradigm_validator import MultiParadigmValidator
from src.server.siraj_v6_1_engine import SirajV61ComputationalHermeneuticsEngine
from config.settings import settings

# Configure logging to stderr (NEVER stdout for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Use stderr to avoid corrupting JSON-RPC
)
logger = logging.getLogger(__name__)

class SirajMCPServer:
    """SIRAJ v6.1 MCP Server with stdio Transport for Claude Desktop"""
    
    def __init__(self):
        """Initialize the MCP server and SIRAJ components"""
        self.server = Server("siraj-computational-hermeneutics-v6.1")
        
        # Initialize SIRAJ v6.1 components lazily to avoid async issues
        logger.info("SIRAJ v6.1 MCP Server initialized - components will be loaded on first use")
        self._components_initialized = False
        self.asa = None
        self.community_validation = None
        self.multi_paradigm_validator = None
        self.siraj_engine = None
        
        self._register_tools()
    
    def _ensure_components_initialized(self):
        """Initialize SIRAJ components on first use"""
        if not self._components_initialized:
            logger.info("Initializing SIRAJ v6.1 components on first use...")
            try:
                self.asa = AdaptiveSemanticArchitecture(settings)
                self.community_validation = CommunityValidationInterface(settings)
                self.multi_paradigm_validator = MultiParadigmValidator(settings)
                self.siraj_engine = SirajV61ComputationalHermeneuticsEngine(settings)
                self._components_initialized = True
                logger.info("SIRAJ v6.1 components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SIRAJ components: {e}")
                raise
        
    def _register_tools(self):
        """Register MCP tools with the server"""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available SIRAJ methodology tools"""
            return [
                Tool(
                    name="computational_hermeneutics_methodology",
                    description="Revolutionary Computational Hermeneutics framework integrating Islamic Tafsir, comparative linguistics, and modern NLP. Returns comprehensive step-by-step analytical methodology.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "string",
                                "description": "The linguistic root to analyze (e.g., 'k-t-b', 'q-r-a')"
                            },
                            "language_family": {
                                "type": "string",
                                "enum": ["arabic", "hebrew", "aramaic", "semitic", "proto_semitic"],
                                "default": "arabic",
                                "description": "Target language family for analysis"
                            },
                            "cultural_context": {
                                "type": "string",
                                "default": "islamic",
                                "description": "Cultural context (islamic, biblical, quranic, classical, modern)"
                            }
                        },
                        "required": ["root"]
                    }
                ),
                Tool(
                    name="adaptive_semantic_architecture",
                    description="Dynamic 5-tier semantic mapping with cultural adaptation. Provides methodology for semantic node generation and cultural context adaptation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze for semantic architecture"
                            },
                            "cultural_context": {
                                "type": "string",
                                "default": "islamic",
                                "description": "Cultural context for semantic mapping"
                            },
                            "adaptation_triggers": {
                                "type": "object",
                                "description": "Triggers for architectural adaptation",
                                "default": {}
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="community_sovereignty_protocols",
                    description="Cultural sovereignty validation methodology ensuring community authority over cultural knowledge. Returns comprehensive validation framework.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_text": {
                                "type": "string",
                                "description": "Source text or concept"
                            },
                            "target_text": {
                                "type": "string",
                                "description": "Target text or concept"
                            },
                            "root": {
                                "type": "string",
                                "description": "Linguistic root being analyzed"
                            },
                            "cultural_context": {
                                "type": "object",
                                "description": "Cultural context for validation",
                                "default": {}
                            }
                        },
                        "required": ["source_text", "target_text", "root"]
                    }
                ),
                Tool(
                    name="multi_paradigm_validation",
                    description="Convergence validation across Traditional (40%), Scientific (30%), and Computational (30%) paradigms. Returns comprehensive validation methodology.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_results": {
                                "type": "object",
                                "description": "Analysis results to validate"
                            },
                            "validation_context": {
                                "type": "string",
                                "default": "comprehensive",
                                "description": "Context for validation (comprehensive, focused, rapid)"
                            }
                        },
                        "required": ["analysis_results"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls and return methodology"""
            try:
                if name == "computational_hermeneutics_methodology":
                    root = arguments.get("root")
                    language_family = arguments.get("language_family", "arabic")
                    cultural_context = arguments.get("cultural_context", "islamic")
                    
                    logger.info(f"Generating computational hermeneutics methodology for root: {root}")
                    
                    # Generate comprehensive methodology
                    methodology = f"""# SIRAJ v6.1: Computational Hermeneutics Methodology

## Analyzing Root: {root}
**Language Family**: {language_family}
**Cultural Context**: {cultural_context}

## METHODOLOGY OVERVIEW
This framework integrates Islamic Tafsir, comparative linguistics, and modern NLP into a unified analytical approach while respecting cultural sovereignty.

### Phase 1: Traditional Hermeneutical Foundation
1. **Primary Source Analysis**
   - Locate root '{root}' in classical Arabic lexicons (Lisan al-Arab, Taj al-Arus)
   - Identify Quranic occurrences and early tafsir interpretations
   - Map traditional semantic ranges and metaphorical extensions

2. **Cross-Linguistic Comparison** 
   - Compare with Hebrew cognates in biblical texts
   - Analyze Aramaic parallels in Talmudic literature
   - Reconstruct Proto-Semitic etymological patterns

### Phase 2: Computational Linguistic Analysis
1. **Morphological Pattern Analysis**
   - Apply root-pattern morphology extraction
   - Generate all theoretically possible derivations
   - Calculate frequency distributions across corpora

2. **Semantic Vector Mapping**
   - Create embeddings for all root derivatives
   - Cluster semantically related forms
   - Identify polysemy and homonymy patterns

### Phase 3: Cultural-Computational Integration  
1. **Community-Informed Validation**
   - Submit findings to qualified cultural authorities
   - Incorporate traditional scholarly consensus
   - Resolve conflicts between computational and traditional interpretations

2. **Hermeneutical Synthesis**
   - Integrate traditional and computational insights
   - Maintain primacy of cultural authority
   - Document methodology for reproducibility

### Implementation Guidelines
- Use traditional exegetical principles as foundational framework
- Apply computational methods as enhancement, not replacement
- Ensure all findings undergo community validation
- Preserve cultural sovereignty throughout process

**Output**: Comprehensive analytical methodology respecting both traditional scholarship and modern computational capabilities.
"""
                    
                    return [TextContent(type="text", text=methodology)]
                
                elif name == "adaptive_semantic_architecture":
                    text = arguments.get("text")
                    cultural_context = arguments.get("cultural_context", "islamic")
                    adaptation_triggers = arguments.get("adaptation_triggers", {})
                    
                    logger.info(f"Generating ASA methodology for text analysis in {cultural_context} context")
                    
                    # Ensure components are initialized
                    self._ensure_components_initialized()
                    
                    # Use the actual ASA component
                    methodology = self.asa.generate_asa_methodology(text, cultural_context)
                    
                    return [TextContent(type="text", text=methodology)]
                
                elif name == "community_sovereignty_protocols":
                    source_text = arguments.get("source_text")
                    target_text = arguments.get("target_text")
                    root = arguments.get("root")
                    cultural_context = arguments.get("cultural_context", {})
                    
                    logger.info(f"Generating community sovereignty protocols for root: {root}")
                    
                    # Ensure components are initialized
                    self._ensure_components_initialized()
                    
                    # Use the actual community validation component
                    methodology = self.community_validation.generate_community_validation_methodology(
                        source_text, target_text, root
                    )
                    
                    return [TextContent(type="text", text=methodology)]
                
                elif name == "multi_paradigm_validation":
                    analysis_results = arguments.get("analysis_results")
                    validation_context = arguments.get("validation_context", "comprehensive")
                    
                    logger.info(f"Generating multi-paradigm validation methodology")
                    
                    # Ensure components are initialized
                    self._ensure_components_initialized()
                    
                    # Use the actual validator component
                    methodology = self.multi_paradigm_validator.generate_validation_methodology(analysis_results)
                    
                    return [TextContent(type="text", text=methodology)]
                
                else:
                    error_msg = f"Unknown tool: {name}"
                    logger.error(error_msg)
                    return [TextContent(type="text", text=f"Error: {error_msg}")]
                    
            except Exception as e:
                error_msg = f"Error in tool '{name}': {str(e)}"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting SIRAJ v6.1 MCP Server...")
    
    try:
        # Create server instance
        server_instance = SirajMCPServer()
        logger.info("Server instance created successfully")
        
        # Run with stdio transport (required for Claude Desktop)
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(read_stream, write_stream, server_instance.server.create_initialization_options())
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("SIRAJ v6.1 MCP Server starting with stdio transport...")
    asyncio.run(main())