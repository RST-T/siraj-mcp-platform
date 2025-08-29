# SIRAJ MCP Server Installation Scripts

This directory contains installation and setup scripts for the SIRAJ v6.1 Computational Hermeneutics MCP Server.

## Scripts

### `install.ps1` (Windows PowerShell)
Complete installation script for Windows users.

**Usage:**
```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install.ps1
```

**What it does:**
- Checks Node.js and Python prerequisites
- Installs the SIRAJ MCP Server NPM package globally
- Installs Python dependencies
- Runs health check
- Provides Claude Desktop configuration instructions

### `install.sh` (Linux/macOS)
Complete installation script for Unix-based systems.

**Usage:**
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

**What it does:**
- Checks Node.js and Python prerequisites
- Installs the SIRAJ MCP Server NPM package globally
- Installs Python dependencies
- Runs health check
- Provides Claude Desktop configuration instructions

### `setup-dev.js` (Development Setup)
Development environment setup script for contributors and developers.

**Usage:**
```bash
node scripts/setup-dev.js
```

**What it does:**
- Checks prerequisites (Node.js, Python)
- Installs NPM dependencies locally
- Builds TypeScript
- Installs Python dependencies
- Runs health check
- Shows available development commands

## Quick Start

### For End Users

**Windows:**
```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install.ps1
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/siraj-team/siraj-mcp/main/scripts/install.sh | bash
```

### For Developers

```bash
git clone https://github.com/siraj-team/siraj-mcp.git
cd siraj-mcp
node scripts/setup-dev.js
```

## Prerequisites

### Required
- **Node.js 18+**: Download from [nodejs.org](https://nodejs.org)
- **Python 3.10+**: Download from [python.org](https://python.org)

### Python Dependencies
The following Python packages will be installed automatically:
- `mcp` - Model Context Protocol framework
- `pydantic` - Data validation
- `fastapi` - Web framework (for SSE transport)
- `transformers` - Hugging Face transformers
- `torch` - PyTorch for ML models
- And other dependencies listed in `requirements.txt`

## Claude Desktop Integration

After installation, you can integrate with Claude Desktop in two ways:

### Option 1: Automatic (Recommended)
```bash
claude mcp add @siraj-team/mcp-server-computational-hermeneutics
```

### Option 2: Manual Configuration
1. Generate configuration:
   ```bash
   siraj-mcp-server generate-config
   ```

2. Add to your Claude Desktop config file:
   - **Windows**: `%APPDATA%/Claude/config.json`
   - **macOS**: `~/Library/Application Support/Claude/config.json` 
   - **Linux**: `~/.config/claude-desktop/config.json`

## Troubleshooting

### Common Issues

**Python not found:**
- Make sure Python 3.10+ is installed and in your PATH
- On Windows, try using `python3` instead of `python`

**Permission denied (Linux/macOS):**
```bash
chmod +x scripts/install.sh
```

**NPM install fails:**
- Check you have sufficient permissions
- Try running with admin/sudo if needed
- Ensure Node.js 18+ is installed

**Health check fails:**
```bash
siraj-mcp-server health
```
This will show you which dependencies are missing.

### Getting Help

If you encounter issues:
1. Run the health check: `siraj-mcp-server health`
2. Check the installation logs
3. Report issues at: https://github.com/siraj-team/siraj-mcp/issues

## License

MIT License - see LICENSE file for details.