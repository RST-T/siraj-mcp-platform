#!/usr/bin/env python3
"""
SIRAJ v6.1 MCP Server Launcher
Convenience script to start the SIRAJ MCP server for Claude Desktop integration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the SIRAJ MCP server"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    server_script = project_root / "src" / "server" / "main_mcp_server.py"
    
    # Check if server script exists
    if not server_script.exists():
        print(f"Error: Server script not found at {server_script}", file=sys.stderr)
        sys.exit(1)
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    print("Starting SIRAJ v6.1 Computational Hermeneutics MCP Server...", file=sys.stderr)
    print("Press Ctrl+C to stop the server", file=sys.stderr)
    print("Server ready for Claude Desktop connection.", file=sys.stderr)
    
    try:
        # Start the server
        subprocess.run([sys.executable, str(server_script)], env=env, check=True)
    except KeyboardInterrupt:
        print("\nSIRAJ MCP Server stopped by user.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()