# SIRAJ MCP Server Setup Guide

## Revolutionary Methodology-First Approach

**SIRAJ provides AI with analytical methodologies, not pre-stored data.**

Instead of returning cached linguistic analyses, SIRAJ teaches AI how to perform scholarly analysis by providing step-by-step methodological frameworks. Each response is dynamically generated based on the specific query, ensuring culturally sensitive and academically rigorous analysis.

## Quick Start

### For Claude Desktop (Recommended)

1. **Copy Configuration**: Add the following to your Claude Desktop config file:
   - Windows: `%AppData%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "siraj-computational-hermeneutics": {
      "command": "python",
      "args": [
        "C:\\Users\\Admin\\Documents\\RST\\Siraj-MCP\\src\\server\\main_mcp_server.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

2. **Restart Claude Desktop**

3. **Verify**: Look for the tools icon in Claude Desktop. You should see:
   - `analyze_linguistic_root`
   - `semantic_mapping`

### For HTTPS/Web Clients

1. **Generate SSL Certificate** (if not already done):
```bash
python scripts/generate-ssl-cert.py
```

2. **Start HTTPS Server**:
```bash
python src/server/main_mcp_server.py --transport https --port 3443 --ssl-cert certs/server.crt --ssl-key certs/server.key
```

3. **Access Endpoints**:
   - SSE Stream: `https://localhost:3443/sse`
   - Messages: `https://localhost:3443/messages/`

## Available Transports

| Transport | Use Case | URL/Connection |
|-----------|----------|----------------|
| stdio | Claude Desktop | Direct process communication |
| sse | Web apps (HTTP) | http://localhost:3001 |
| https | Secure web apps | https://localhost:3443 |

## Available Methodology Tools

### analyze_linguistic_root
**Provides methodology for systematic Semitic root analysis**

Returns a dynamic analytical framework customized for the specific root and context. Guides AI through:
- Structural deconstruction methodology
- Semantic domain mapping process
- Historical-comparative analysis steps  
- Cultural sovereignty protocols
- 72-node archetypal mapping framework

**Parameters:**
- `root` (string, required): The linguistic root to analyze
- `language_family` (string, required): arabic, hebrew, aramaic, or semitic
- `cultural_context` (string, optional): Cultural context (e.g., biblical, quranic, classical)

### semantic_mapping
**Provides 72-node archetypal semantic mapping methodology**

Returns a systematic framework for analyzing texts through archetypal semantic categories. Adapts methodology based on depth level and text characteristics.

**Parameters:**
- `text` (string, required): Text to analyze
- `depth` (number, optional): Analysis depth level (1-3)

### cross_textual_evolution  
**Provides methodology for tracing linguistic evolution between texts**

Returns a comparative analysis framework for studying how concepts evolve between textual traditions while respecting cultural sovereignty.

**Parameters:**
- `source_text` (string, required): hebrew_bible, aramaic_targum, septuagint, arabic_quran, classical_arabic, other
- `target_text` (string, required): hebrew_bible, aramaic_targum, septuagint, arabic_quran, classical_arabic, other  
- `root` (string, required): Root or concept to trace

## Testing

### Test stdio transport:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python src/server/main_mcp_server.py --transport stdio
```

### Test HTTPS endpoint:
```bash
curl -k https://localhost:3443/sse
```

## Example: Methodology-First Approach in Action

**Before (Data-Centric):** AI would get the same canned response for different roots:
```
Root Analysis: k-t-b
Semantic Core: Writing, Recording, Inscription
Archetypal Mappings: 1. Creation through inscription...
```

**Now (Methodology-Centric):** AI receives dynamic analytical framework:
```
SIRAJ ROOT ANALYSIS METHODOLOGY for k-t-b (arabic)

STEP 1: STRUCTURAL DECONSTRUCTION
• Extract consonantal skeleton from 'k-t-b'
• Identify trilateral/quadrilateral pattern
• Map vowel positions and morphological variants
...

STEP 2: SEMANTIC DOMAIN MAPPING  
Apply Arabic morphological patterns and Quranic usage analysis
• Collect usage examples from primary sources
• Identify core semantic field and primary meaning
...

[Complete step-by-step methodology continues]
```

This enables AI to:
- Apply different analytical approaches for different contexts
- Understand the analytical process, not just get answers
- Respect cultural sovereignty through built-in protocols
- Generate unique insights based on methodological application

## Security Notes

- The provided SSL certificates are self-signed and for development only
- For production, use certificates from a trusted Certificate Authority
- The HTTPS transport implements proper SSL/TLS encryption
- Cultural sovereignty protection is built into all methodological frameworks
- Future integration with community validation website for cultural leader oversight