#!/bin/bash

# SIRAJ MCP Server Installation Script for Linux/macOS
# Shell script to install and configure SIRAJ MCP Server

echo "🧠 SIRAJ v6.1 Computational Hermeneutics MCP Server Installation"
echo "   Advanced linguistic analysis with cultural sovereignty protection"
echo ""

# Check Node.js
if command -v node >/dev/null 2>&1; then
    node_version=$(node --version)
    echo "✅ Node.js: $node_version"
else
    echo "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check Python
if command -v python >/dev/null 2>&1; then
    python_version=$(python --version)
    echo "✅ Python: $python_version"
elif command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version)
    echo "✅ Python: $python_version"
else
    echo "❌ Python not found. Please install Python 3.10+ from https://python.org"
    exit 1
fi

# Install NPM package globally
echo "📦 Installing SIRAJ MCP Server package..."
if npm install -g @siraj-team/mcp-server-computational-hermeneutics; then
    echo "✅ Package installed successfully"
else
    echo "❌ Failed to install NPM package"
    exit 1
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if siraj-mcp-server install; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Health check
echo "🏥 Running health check..."
siraj-mcp-server health

echo ""
echo "🎉 SIRAJ MCP Server installation completed!"
echo ""
echo "To add to Claude Desktop, run:"
echo "  claude mcp add @siraj-team/mcp-server-computational-hermeneutics"
echo ""
echo "Or manually add to your Claude Desktop config:"
echo "{"
echo '  "mcpServers": {'
echo '    "siraj-computational-hermeneutics": {'
echo '      "command": "siraj-mcp-server",'
echo '      "args": ["start"]'
echo "    }"
echo "  }"
echo "}"