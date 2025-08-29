# SIRAJ v6.1 - Claude Desktop Integration Guide

## Status: FULLY OPERATIONAL ✅

The SIRAJ v6.1 Computational Hermeneutics MCP Server has been successfully rewritten and tested with proper stdio transport protocol for Claude Desktop integration.

## What Was Fixed

1. **✅ Transport Protocol**: Changed from SSE/HTTPS to stdio transport (required for Claude Desktop)
2. **✅ JSON-RPC Compliance**: Implemented proper JSON-RPC 2.0 protocol
3. **✅ Logging Issues**: Moved all logging to stderr to avoid corrupting stdio communication
4. **✅ Initialization**: Fixed server initialization with proper MCP protocol handshake
5. **✅ Tool Registration**: All 4 SIRAJ methodology tools are properly registered and functional

## Available Tools

The server provides 4 comprehensive methodology-first tools:

### 1. `computational_hermeneutics_methodology`
**Description**: Revolutionary framework integrating Islamic Tafsir, comparative linguistics, and modern NLP
**Parameters**: 
- `root` (required): Linguistic root to analyze (e.g., 'k-t-b', 'q-r-a')
- `language_family`: Target language family (arabic, hebrew, aramaic, semitic, proto_semitic)
- `cultural_context`: Cultural context (islamic, biblical, quranic, classical, modern)

### 2. `adaptive_semantic_architecture`
**Description**: Dynamic 5-tier semantic mapping with cultural adaptation
**Parameters**:
- `text` (required): Text to analyze for semantic architecture
- `cultural_context`: Cultural context for semantic mapping
- `adaptation_triggers`: Triggers for architectural adaptation

### 3. `community_sovereignty_protocols`
**Description**: Cultural sovereignty validation ensuring community authority over cultural knowledge
**Parameters**:
- `source_text` (required): Source text or concept
- `target_text` (required): Target text or concept  
- `root` (required): Linguistic root being analyzed
- `cultural_context`: Cultural context for validation

### 4. `multi_paradigm_validation`
**Description**: Convergence validation across Traditional (40%), Scientific (30%), and Computational (30%) paradigms
**Parameters**:
- `analysis_results` (required): Analysis results to validate
- `validation_context`: Context for validation (comprehensive, focused, rapid)

## Claude Desktop Configuration

### Step 1: Locate Claude Desktop Config File

On Windows, the configuration file is located at:
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Step 2: Add SIRAJ Server Configuration

Copy the configuration from `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "siraj-computational-hermeneutics": {
      "command": "python",
      "args": [
        "C:/Users/Admin/Documents/RST/Siraj-MCP/src/server/main_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "C:/Users/Admin/Documents/RST/Siraj-MCP"
      }
    }
  }
}
```

**Note**: Adjust the path `C:/Users/Admin/Documents/RST/Siraj-MCP` to match your actual project location.

### Step 3: Restart Claude Desktop

1. Completely quit Claude Desktop
2. Restart Claude Desktop
3. The SIRAJ tools should now appear in Claude's interface

## Server Testing Commands

### Test Server Initialization
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "claude-desktop", "version": "1.0"}}}' | python src/server/main_mcp_server.py
```

### Test Tools Listing
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "claude-desktop", "version": "1.0"}}}
{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}' | python src/server/main_mcp_server.py
```

## Expected Behavior

When properly configured:

1. **Tools Appear**: SIRAJ tools will appear in Claude Desktop's tool selection
2. **Methodology Response**: Tools return comprehensive step-by-step methodologies (not analysis results)
3. **Cultural Sovereignty**: All tools respect community authority over cultural knowledge
4. **Error Handling**: Tools provide helpful methodology guidance even when errors occur

## Troubleshooting

### Tools Don't Appear
- Verify Claude Desktop configuration file path and syntax
- Check that Python path in config matches your Python installation
- Ensure project path is absolute and correct
- Restart Claude Desktop completely

### Server Connection Issues
- Test server manually with the testing commands above
- Check that all dependencies are installed (`pip install -r requirements.txt`)
- Verify no other process is using stdio

### Component Loading Errors
- SIRAJ components are loaded lazily on first use
- Check stderr logs for component initialization errors
- Ensure all required dependencies are available

## Technical Architecture

- **Transport**: stdio (standard input/output) for Claude Desktop compatibility
- **Protocol**: JSON-RPC 2.0 over stdio streams
- **Logging**: All logs go to stderr to avoid corrupting JSON-RPC communication
- **Components**: Lazy loading to avoid blocking server startup
- **Error Handling**: Graceful degradation with methodology guidance

## Success Criteria

✅ Server starts without errors  
✅ Responds to MCP protocol initialization  
✅ Lists all 4 tools correctly  
✅ Tools execute and return methodologies  
✅ Logging works without interfering with communication  
✅ Compatible with Claude Desktop integration  

**Status: READY FOR PRODUCTION USE**