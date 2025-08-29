# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIRAJ v6.1 is a Computational Hermeneutics MCP (Model Context Protocol) server that provides advanced linguistic analysis capabilities with cultural sovereignty protection, specializing in Arabic and Semitic languages. The system implements a methodology-first approach where AI is taught analytical frameworks rather than returning pre-stored data.

## Build and Development Commands

### Python Development
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -e .  # Install package in development mode

# Run tests
python -m pytest tests/test_framework.py::SirajTestFramework -v --tb=short
pytest tests/ --cov=src --cov-report=html

# Code quality
black src/ --line-length 100
isort src/ --profile black
flake8 src/
mypy src/ --strict
```

### Node.js/TypeScript Development
```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Development mode
npm run dev

# Linting and formatting
npm run lint
npm run format

# Health check
npm run health-check
```

### Running the MCP Server
```bash
# Stdio transport (for Claude Desktop)
python src/server/main_mcp_server.py --transport stdio

# HTTPS transport
python src/server/main_mcp_server.py --transport https --port 3443 --ssl-cert certs/server.crt --ssl-key certs/server.key

# Generate SSL certificates
python scripts/generate-ssl-cert.py
```

## Architecture

### Core Components (✅ FULLY IMPLEMENTED)

**Python Backend (`src/server/`):**
- `main_mcp_server.py`: **✅ COMPLETE** - MCP server with 4 methodology-first tools, full component integration
- `siraj_v6_1_engine.py`: **✅ COMPLETE** - Core computational hermeneutics engine with comprehensive methodology generation
- `adaptive_semantic_architecture.py`: **✅ COMPLETE** - 5-tier dynamic semantic system with universal primitives
- `community_validation_interface.py`: **✅ COMPLETE** - Multi-phase validation workflow with cultural sovereignty protocols
- `multi_paradigm_validator.py`: **✅ COMPLETE** - Traditional/scientific/computational convergence validation (ijmā', statistical, cross-validation)
- `cultural_sovereignty_manager.py`: Integrated into community validation interface
- `transformer_integration.py`: Framework ready for transformer model integration

**TypeScript Wrapper (`src/`):**
- `index.ts`: Node.js wrapper that launches the Python server
- `config-schema.ts`: Configuration validation using Zod

**Database Layer (`src/database/`):**
- `connection_manager.py`: Database connection handling (SQLite, PostgreSQL via Neon)
- `corpus_access.py`: Linguistic corpus access layer

### Key Design Patterns

1. **Methodology-First Approach**: The system provides analytical frameworks rather than cached results
2. **Multi-Paradigm Validation**: Combines traditional, scientific, and computational approaches
3. **Cultural Sovereignty Protection**: Built-in safeguards for sacred texts and cultural content
4. **72-Node Archetypal System**: Deep semantic understanding through archetypal analysis

## MCP Protocol Implementation (✅ METHODOLOGY-FIRST APPROACH)

The server implements 4 comprehensive MCP tools that provide **methodological frameworks, not analysis results**:

### 1. `computational_hermeneutics_methodology`
**✅ FULLY IMPLEMENTED** - Revolutionary framework integrating Islamic Tafsīr, comparative linguistics, and modern NLP
- Teaches 6-phase analysis methodology combining traditional and computational approaches
- Includes cultural sovereignty assessment, multi-paradigm validation, and community consultation protocols
- Returns step-by-step guidance for performing computational hermeneutics analysis

### 2. `adaptive_semantic_architecture` 
**✅ FULLY IMPLEMENTED** - Dynamic 5-tier semantic mapping that evolves based on context
- Universal Tier 1 nodes: 5 cross-culturally validated semantic primitives (inquiry, protection, perception, restoration, selection)
- Adaptive Tiers 2-5: Context-driven node generation with community validation
- Returns methodology for semantic mapping, architecture adaptation assessment, and cultural alignment protocols

### 3. `community_sovereignty_protocols`
**✅ FULLY IMPLEMENTED** - Comprehensive cultural sovereignty and community validation methodology
- Multi-phase validation: preparation → consultation → refinement → approval
- Sensitivity protocols for standard/sensitive/sacred content with appropriate authority requirements
- Returns detailed community engagement methodology respecting traditional knowledge sovereignty

### 4. `multi_paradigm_validation`
**✅ FULLY IMPLEMENTED** - Cross-paradigm validation across traditional, scientific, and computational epistemologies
- Traditional: ijmā' consensus, textual verification, chain of transmission (isnād) criticism
- Scientific: statistical significance, reproducibility, peer review protocols
- Computational: cross-validation, model evaluation, algorithmic verification
- Returns convergence scoring methodology: VCS = (Traditional × 0.4) + (Scientific × 0.3) + (Computational × 0.3)

## Testing Strategy

- Unit tests in `tests/test_framework.py` covering all major components
- Multi-paradigm testing across traditional, scientific, and computational domains
- Cultural sensitivity validation tests
- Performance optimization tests with caching validation

## Database Schema

The system uses SQLite locally with optional Neon PostgreSQL integration. Key tables:
- Linguistic roots and morphological patterns
- Semantic mappings and archetypal nodes
- Community validation records
- Cultural sensitivity thresholds

## Configuration

Environment variables and settings managed through:
- `config/settings.py`: Python configuration
- `.env` file for sensitive credentials
- `claude_desktop_config.json` for Claude Desktop integration

## Revolutionary Methodology-First Paradigm ✨

**SIRAJ v6.1 represents a paradigmatic shift from data-centric to process-centric AI interaction.**

### What Makes This Revolutionary:
1. **Teaches HOW, Not WHAT**: Returns analytical methodologies rather than analysis results
2. **Cultural Sovereignty**: Traditional authorities retain ultimate control over cultural knowledge
3. **Multi-Paradigm Integration**: Systematic convergence across traditional, scientific, and computational epistemologies
4. **Community-Centered**: Built-in protocols for community validation and benefit-sharing
5. **Dynamic Adaptation**: Semantic architecture evolves based on cultural context and community feedback

### Implementation Philosophy:
- **Methodology-First**: Every tool returns step-by-step frameworks for performing analysis
- **Community Authority**: Cultural communities have final say over their knowledge representations
- **Traditional Wisdom Enhanced**: Computational capabilities amplify traditional scholarly methods
- **Transparent Process**: Complete audit trails and validation documentation

## Critical Workflows (✅ FULLY IMPLEMENTED)

### 1. Computational Hermeneutics Analysis
```
User Request → Cultural Context Assessment → Sensitivity Classification → 
Multi-Paradigm Methodology Generation → Community Protocols (if needed) → 
Comprehensive Analysis Framework Delivery
```

### 2. Cultural Sovereignty Protection  
```
Content Analysis → Sensitivity Detection → Community Authority Identification → 
Validation Workflow Design → Community Consultation Methodology → 
Respectful Analysis Framework
```

### 3. Adaptive Semantic Architecture
```
Text Input → Universal Node Mapping → Cultural Context Adaptation → 
Architecture Adaptation Assessment → Community Validation (if needed) → 
Semantic Analysis Methodology
```

### 4. Multi-Paradigm Validation
```
Analysis Results → Traditional Validation Design → Scientific Validation Design → 
Computational Validation Design → Convergence Score Calculation → 
Comprehensive Validation Framework
```

## Performance Considerations

- Transformer model caching for repeated analyses
- Batch processing for multiple linguistic roots
- Async operations throughout the Python backend
- Connection pooling for database access