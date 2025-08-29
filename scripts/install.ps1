# SIRAJ MCP Server Installation Script for Windows
# PowerShell script to install and configure SIRAJ MCP Server

Write-Host "🧠 SIRAJ v6.1 Computational Hermeneutics MCP Server Installation" -ForegroundColor Blue
Write-Host "   Advanced linguistic analysis with cultural sovereignty protection" -ForegroundColor Gray
Write-Host ""

# Check Node.js
$nodeVersion = $null
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org" -ForegroundColor Red
    exit 1
}

# Check Python
$pythonVersion = $null
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Install NPM package globally
Write-Host "📦 Installing SIRAJ MCP Server package..." -ForegroundColor Blue
try {
    npm install -g @siraj-team/mcp-server-computational-hermeneutics
    Write-Host "✅ Package installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to install NPM package" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "🐍 Installing Python dependencies..." -ForegroundColor Blue
try {
    siraj-mcp-server install
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}

# Health check
Write-Host "🏥 Running health check..." -ForegroundColor Blue
siraj-mcp-server health

Write-Host ""
Write-Host "🎉 SIRAJ MCP Server installation completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To add to Claude Desktop, run:" -ForegroundColor Yellow
Write-Host "  claude mcp add @siraj-team/mcp-server-computational-hermeneutics" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or manually add to your Claude Desktop config:" -ForegroundColor Yellow
Write-Host '{' -ForegroundColor Gray
Write-Host '  "mcpServers": {' -ForegroundColor Gray
Write-Host '    "siraj-computational-hermeneutics": {' -ForegroundColor Gray
Write-Host '      "command": "siraj-mcp-server",' -ForegroundColor Gray
Write-Host '      "args": ["start"]' -ForegroundColor Gray
Write-Host '    }' -ForegroundColor Gray
Write-Host '  }' -ForegroundColor Gray
Write-Host '}' -ForegroundColor Gray