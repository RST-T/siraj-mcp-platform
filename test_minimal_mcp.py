#!/usr/bin/env python3
"""
Minimal MCP server test to isolate issues
"""

import asyncio
import sys
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Create server
app = Server("test-server")

@app.list_tools()
async def list_tools():
    """List available tools"""
    return [
        Tool(
            name="test_tool",
            description="A simple test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Test message"
                    }
                },
                "required": ["message"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls"""
    if name == "test_tool":
        message = arguments.get("message", "Hello")
        return [TextContent(type="text", text=f"Test response: {message}")]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point"""
    logger.info("Starting minimal MCP server test...")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("stdio_server context established")
            await app.run(read_stream, write_stream, app.create_initialization_options())
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())