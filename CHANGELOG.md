# Changelog

All notable changes to the SIRAJ v6.1 Computational Hermeneutics MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.1.0] - 2025-08-28

### Added - MCP Server Integration

#### 🚀 Model Context Protocol (MCP) Support
- **Full MCP Server Implementation**: Complete integration with Claude Desktop via MCP protocol
- **Node.js CLI Wrapper**: TypeScript-based CLI that wraps the Python SIRAJ engine
- **Multiple Transport Protocols**: Support for stdio (default) and SSE transports
- **Claude Desktop Compatibility**: Ready for `claude mcp add` integration

#### 🔧 Command Line Interface
- **`siraj-mcp-server start`**: Start the MCP server with configurable options
- **`siraj-mcp-server health`**: Comprehensive health check for dependencies
- **`siraj-mcp-server install`**: Python dependency installation
- **`siraj-mcp-server generate-config`**: Claude Desktop configuration generator
- **Configuration Options**: Transport, debug mode, logging levels, and more

#### 📦 Package Distribution
- **NPM Package**: `@siraj-team/mcp-server-computational-hermeneutics`
- **Global Installation**: Support for global NPM installation
- **Binary Executable**: `siraj-mcp-server` command available after installation
- **MCP Metadata**: Complete metadata for Claude Desktop integration

#### ⚙️ Configuration System
- **Zod Schema Validation**: Type-safe configuration with validation
- **Multiple Configuration Formats**: Basic, debug, SSE, and performance presets
- **Environment Variables**: Support for environment-based configuration
- **Cultural Sovereignty Settings**: Configurable protection thresholds

#### 🛠️ Installation & Setup
- **Cross-Platform Scripts**: PowerShell (Windows) and Bash (Linux/macOS) installers
- **Development Setup**: Automated development environment configuration
- **Health Verification**: Automated dependency and configuration checking
- **Post-Install Guidance**: Automatic setup instructions and examples

### Enhanced - Existing Features

#### 🔍 Python Engine Integration
- **Import Path Resolution**: Fixed relative import issues for MCP execution
- **Module Execution**: Support for `python -m` execution mode
- **Environment Isolation**: Proper PYTHONPATH configuration for MCP context
- **Error Handling**: Improved error reporting and debugging capabilities

#### 🛡️ Cultural Sovereignty
- **Configuration Integration**: Cultural sovereignty settings via configuration schema
- **MCP Context Awareness**: Cultural validation within MCP request/response flow
- **Community Validation**: Integration with MCP tool execution pipeline
- **Sensitivity Thresholds**: Configurable via MCP server configuration

#### 📊 Performance Optimization
- **Startup Optimization**: Faster initialization for MCP server startup
- **Caching Integration**: Performance cache configuration via MCP settings
- **Resource Management**: Optimized resource usage for MCP environment
- **Batch Processing**: Enhanced batch processing for MCP tool requests

### Technical - Infrastructure

#### 🏗️ Architecture
- **Dual-Layer Design**: Node.js wrapper + Python engine architecture
- **Process Management**: Proper subprocess management and cleanup
- **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM
- **Transport Abstraction**: Clean separation of stdio vs SSE protocols

#### 🧪 Testing & Quality
- **Integration Tests**: Comprehensive MCP integration testing
- **Health Checks**: Automated dependency and configuration validation
- **CLI Testing**: Complete command-line interface validation
- **Configuration Testing**: Validation of all configuration scenarios

#### 📚 Documentation
- **Comprehensive README**: Complete usage guide with examples
- **Installation Scripts**: Documented installation procedures
- **Configuration Reference**: Detailed configuration options
- **Integration Examples**: Claude Desktop integration examples

### Fixed

#### 🐛 Import Issues
- **Relative Imports**: Fixed relative import issues in MCP execution context
- **PYTHONPATH Configuration**: Correct path setup for module resolution
- **Working Directory**: Proper CWD configuration for Python execution

#### 🔧 Configuration Issues
- **Transport Mapping**: Fixed transport protocol mapping (stdio/sse)
- **Argument Passing**: Correct argument passing to Python server
- **Environment Variables**: Proper environment variable handling

#### 📦 Package Issues
- **Binary Distribution**: Fixed binary executable generation and installation
- **Dependency Management**: Proper NPM and Python dependency handling
- **Path Resolution**: Fixed cross-platform path resolution issues

### Changed

#### 🔄 Python Server Updates
- **MCP Argument Support**: Updated argument parsing for MCP integration
- **Transport Protocols**: Standardized stdio and SSE protocol support
- **Logging Integration**: Enhanced logging for MCP debugging

#### 📋 Configuration Schema
- **Structured Configuration**: Migrated from ad-hoc to structured configuration
- **Type Safety**: Full TypeScript type safety for all configuration options
- **Validation**: Comprehensive configuration validation with helpful errors

### Security

#### 🔒 Process Security
- **Subprocess Isolation**: Proper process isolation and resource management
- **Signal Handling**: Secure signal handling and graceful shutdown
- **Environment Protection**: Protected environment variable handling

#### 🛡️ Cultural Protection
- **MCP Integration**: Cultural sovereignty protection within MCP context
- **Request Validation**: Validation of MCP requests for cultural sensitivity
- **Community Oversight**: Integration of community validation workflows

---

## Migration Guide

### For Existing Users

If you were using SIRAJ v6.1 before this MCP integration:

1. **Install the MCP Package**:
   ```bash
   npm install -g @siraj-team/mcp-server-computational-hermeneutics
   ```

2. **Verify Installation**:
   ```bash
   siraj-mcp-server health
   ```

3. **Integrate with Claude Desktop**:
   ```bash
   claude mcp add @siraj-team/mcp-server-computational-hermeneutics
   ```

### For New Users

1. **Quick Installation**:
   ```bash
   claude mcp add @siraj-team/mcp-server-computational-hermeneutics
   ```

2. **Manual Setup** (if needed):
   ```bash
   npm install -g @siraj-team/mcp-server-computational-hermeneutics
   siraj-mcp-server generate-config
   ```

## Breaking Changes

### None

This release maintains full backward compatibility with the existing Python SIRAJ engine while adding new MCP integration capabilities.

## Contributors

- **Claude Code AI Assistant**: Complete MCP server implementation and integration
- **SIRAJ Team**: Original computational hermeneutics framework

## Links

- **GitHub Repository**: https://github.com/siraj-team/siraj-mcp
- **NPM Package**: https://www.npmjs.com/package/@siraj-team/mcp-server-computational-hermeneutics
- **Claude Desktop**: https://claude.ai
- **Model Context Protocol**: https://modelcontextprotocol.io