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
    
    async def _ensure_components_initialized(self):
        """Initialize SIRAJ components on first use"""
        if not self._components_initialized:
            logger.info("Initializing SIRAJ v6.1 components on first use...")
            try:
                self.asa = AdaptiveSemanticArchitecture(settings)
                self.community_validation = CommunityValidationInterface(settings)
                self.multi_paradigm_validator = MultiParadigmValidator(settings)
                self.siraj_engine = SirajV61ComputationalHermeneuticsEngine(settings)
                
                # Initialize the engine's async components (database connections, etc.)
                logger.info("Initializing SIRAJ engine async components...")
                await self.siraj_engine.initialize()
                
                self._components_initialized = True
                logger.info("SIRAJ v6.1 components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SIRAJ components: {e}")
                raise
    
    async def _generate_database_informed_methodology(self, root: str, language_family: str, cultural_context: str) -> str:
        """Generate methodology informed by database content"""
        # Initialize components if needed
        await self._ensure_components_initialized()
        
        # Try to get database information about the root
        database_info = await self._query_root_database(root)
        
        # Generate methodology based on database findings
        if database_info and database_info.get('found'):
            return self._create_database_enhanced_methodology(root, language_family, cultural_context, database_info)
        else:
            return self._create_standard_methodology(root, language_family, cultural_context)
    
    async def _query_root_database(self, root: str) -> dict:
        """Query database for root information"""
        try:
            # Initialize database access if not already done
            if hasattr(self.siraj_engine, 'corpus_access'):
                corpus_access = self.siraj_engine.corpus_access
                
                # Query all sources for the root
                quranic_data = await corpus_access.analyze_quranic_usage(root)
                hadith_data = await corpus_access.analyze_hadith_references(root) 
                classical_data = await corpus_access.analyze_classical_arabic_usage(root)
                
                # Check if we found meaningful data
                found = any([
                    quranic_data.get('occurrences', 0) > 0,
                    hadith_data.get('references', 0) > 0,
                    classical_data.get('usage_examples', 0) > 0
                ])
                
                return {
                    'found': found,
                    'quranic': quranic_data,
                    'hadith': hadith_data,
                    'classical': classical_data
                }
            else:
                logger.warning("Corpus access not available, using standard methodology")
                return {'found': False}
        except Exception as e:
            logger.error(f"Database query failed for root {root}: {e}")
            return {'found': False}
    
    def _create_database_enhanced_methodology(self, root: str, language_family: str, cultural_context: str, database_info: dict) -> str:
        """Create methodology enhanced with actual database findings"""
        quranic_data = database_info.get('quranic', {})
        hadith_data = database_info.get('hadith', {})
        classical_data = database_info.get('classical', {})
        
        # Create enhanced methodology sections
        quranic_section = self._generate_quranic_methodology_section(root, quranic_data)
        hadith_section = self._generate_hadith_methodology_section(root, hadith_data)
        classical_section = self._generate_classical_methodology_section(root, classical_data)
        
        methodology = f"""# SIRAJ v6.1: DATABASE-INFORMED Computational Hermeneutics Methodology

## Analyzing Root: {root} 
**Language Family**: {language_family}
**Cultural Context**: {cultural_context}
**Database Status**: ✅ ACTIVE - Using populated Semitic roots database

## METHODOLOGY OVERVIEW
This framework integrates your populated database of 32 Semitic roots with Islamic Tafsir, comparative linguistics, and modern NLP into a unified analytical approach while respecting cultural sovereignty.

### Phase 1: Database-Informed Primary Analysis
{quranic_section}

{hadith_section}

{classical_section}

### Phase 2: Comparative Linguistic Enhancement
1. **Cross-Semitic Root Analysis**
   - Based on database findings, compare with Hebrew cognates
   - Analyze morphological patterns found in your database
   - Trace Proto-Semitic reconstructions using database etymological data

2. **Semantic Field Mapping**
   - Use database semantic classifications to map meaning evolution
   - Apply your JSON-structured derived meanings for computational analysis
   - Cross-reference with McCarthy (1981-1994) prosodic morphology patterns in database

### Phase 3: Computational-Traditional Integration
1. **Database-Guided Computational Analysis**
   - Use database morphological patterns for transformer model input
   - Apply semantic field data from database for vector space modeling
   - Leverage scholarly consensus data for validation weighting

2. **Community-Informed Synthesis**
   - Submit database-enhanced findings to cultural authorities
   - Use database cultural sovereignty metadata for validation protocols
   - Integrate traditional knowledge with computational insights

### Implementation Guidelines
- **Database-First Approach**: Always consult populated database before external searches
- **Methodology-First Principle**: Return analytical frameworks, not cached answers
- **Cultural Sovereignty**: Respect authority hierarchies documented in database
- **Multi-Paradigm Validation**: Use Traditional (ijma'), Scientific, and Computational convergence

**Revolutionary Output**: AI learns sophisticated analytical frameworks enhanced by your 32-root Semitic database while maintaining methodology-first principles.
"""
        return methodology
    
    def _create_standard_methodology(self, root: str, language_family: str, cultural_context: str) -> str:
        """Create standard methodology when database doesn't contain the root"""
        return f"""# SIRAJ v6.1: Computational Hermeneutics Methodology

## Analyzing Root: {root}
**Language Family**: {language_family}
**Cultural Context**: {cultural_context}
**Database Status**: ⚠️  Root not found in populated database - Using standard methodology

## METHODOLOGY OVERVIEW
This framework provides analytical guidance for roots not yet in your 32 Semitic roots database.

### Phase 1: External Source Research
1. **Primary Source Search**
   - Search classical Arabic lexicons for root '{root}'
   - Identify Quranic occurrences using concordance tools
   - Look for early tafsir interpretations

2. **Comparative Analysis**
   - Search for Hebrew/Aramaic cognates
   - Research Proto-Semitic reconstructions
   - Document morphological patterns

### Phase 2: Integration with Database Framework
1. **Cross-Reference with Known Roots**
   - Compare findings with your 32 populated roots
   - Look for semantic field overlaps
   - Apply similar analytical patterns

### Recommendation: Consider adding '{root}' to your Semitic roots database if analysis yields significant scholarly value.
"""
    
    def _generate_quranic_methodology_section(self, root: str, quranic_data: dict) -> str:
        """Generate Quranic analysis methodology based on database findings"""
        occurrences = quranic_data.get('occurrences', 0)
        
        if occurrences > 0:
            return '''1. **Quranic Analysis (DATABASE-INFORMED)**
   - **Database Status**: Found {} occurrences of root '{}' in Quranic corpus
   - **Methodology**: Analyze each occurrence for contextual meaning variation
   - **Cross-Reference**: Apply Wujuh wa Naza'ir (polysemy) analysis to database occurrences
   - **Tafsir Integration**: Consult Al-Tabari methodology for preserved meanings
   - **Action**: Use database occurrence contexts to map semantic evolution patterns'''.format(occurrences, root)
        else:
            return '''1. **Quranic Analysis (DATABASE-GUIDED)**
   - **Database Status**: No occurrences of root '{}' found in Quranic corpus
   - **Methodology**: Use morphological patterns from database to search for related forms
   - **Alternative**: Search for semantic field overlaps with known database roots
   - **Community Consultation**: Submit to Islamic scholars for validation of absence'''.format(root)
    
    def _generate_hadith_methodology_section(self, root: str, hadith_data: dict) -> str:
        """Generate Hadith analysis methodology based on database findings"""
        references = hadith_data.get('references', 0)
        
        if references > 0:
            return f"""2. **Hadith Analysis (DATABASE-INFORMED)**
   - **Database Status**: Found {references} references of root '{root}' in Hadith corpus
   - **Methodology**: Analyze narrator chains (isnad) and authenticity grades from database
   - **Subject Mapping**: Use database subject tags for thematic analysis
   - **Cross-Validation**: Compare database authenticity grades with traditional classifications"""
        else:
            return f"""2. **Hadith Analysis (DATABASE-GUIDED)**
   - **Database Status**: No references of root '{root}' found in Hadith corpus
   - **Methodology**: Search for morphologically related forms using database patterns
   - **Validation**: Cross-reference with traditional Hadith scholarship"""
    
    def _generate_classical_methodology_section(self, root: str, classical_data: dict) -> str:
        """Generate classical Arabic analysis methodology based on database findings"""
        examples = classical_data.get('usage_examples', 0)
        
        if examples > 0:
            return f"""3. **Classical Arabic Analysis (DATABASE-INFORMED)**
   - **Database Status**: Found {examples} usage examples of root '{root}' in classical texts
   - **Methodology**: Analyze literary contexts and cultural markers from database
   - **Diachronic Analysis**: Trace meaning evolution using database temporal data
   - **Genre Analysis**: Apply database genre classifications for usage pattern mapping"""
        else:
            return f"""3. **Classical Arabic Analysis (DATABASE-GUIDED)**
   - **Database Status**: No usage examples of root '{root}' found in classical texts
   - **Methodology**: Use database morphological patterns to guide external research
   - **Pattern Matching**: Apply successful analysis patterns from database roots"""
        
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
                    
                    logger.info(f"Generating DATABASE-INFORMED computational hermeneutics methodology for root: {root}")
                    
                    # Generate database-informed methodology
                    methodology = await self._generate_database_informed_methodology(root, language_family, cultural_context)
                    
                    return [TextContent(type="text", text=methodology)]
                
                elif name == "adaptive_semantic_architecture":
                    text = arguments.get("text")
                    cultural_context = arguments.get("cultural_context", "islamic")
                    adaptation_triggers = arguments.get("adaptation_triggers", {})
                    
                    logger.info(f"Generating ASA methodology for text analysis in {cultural_context} context")
                    
                    # Ensure components are initialized
                    await self._ensure_components_initialized()
                    
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
                    await self._ensure_components_initialized()
                    
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
                    await self._ensure_components_initialized()
                    
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