#!/usr/bin/env node

/**
 * SIRAJ v6.1 Computational Hermeneutics MCP Server
 * 
 * A Node.js wrapper that launches the Python-based SIRAJ computational hermeneutics
 * engine via MCP protocol. This server provides advanced linguistic analysis with
 * cultural sovereignty protection for Arabic and Semitic languages.
 * 
 * @author SIRAJ Team
 * @version 6.1.0
 * @license MIT
 */

import { Command } from 'commander';
import { spawn, ChildProcess } from 'child_process';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';
import chalk from 'chalk';
import dotenv from 'dotenv';

// ES module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
dotenv.config();

import { ServerConfig, ConfigValidator, defaultServerConfig } from './config-schema.js';

class SirajMCPServer {
  private pythonProcess: ChildProcess | null = null;
  private config: ServerConfig;

  constructor(config: Partial<ServerConfig> = {}) {
    this.config = ConfigValidator.validateServerConfig({
      ...defaultServerConfig,
      ...config
    });
  }

  /**
   * Find the Python server executable
   */
  private findPythonServer(): string {
    // Look for Python server in several locations
    const possiblePaths = [
      join(__dirname, '../src/server/main_mcp_server.py'),
      join(__dirname, '../src/server/main.py'),
      join(process.cwd(), 'src/server/main_mcp_server.py'),
      join(process.cwd(), 'src/server/main.py'),
    ];

    for (const path of possiblePaths) {
      if (existsSync(path)) {
        return path;
      }
    }

    throw new Error(`Could not find Python server. Searched paths: ${possiblePaths.join(', ')}`);
  }

  /**
   * Validate Python environment
   */
  private async validatePython(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonCheck = spawn(this.config.pythonPath!, ['--version']);
      
      pythonCheck.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Python not found at ${this.config.pythonPath}. Please install Python 3.10+ or specify --python-path`));
        }
      });

      pythonCheck.on('error', (error) => {
        reject(new Error(`Failed to execute Python: ${error.message}`));
      });
    });
  }

  /**
   * Start the SIRAJ MCP server
   */
  async start(): Promise<void> {
    try {
      // Validate Python environment
      await this.validatePython();
      
      // Find Python server
      const serverScript = this.findPythonServer();
      
      if (this.config.debug) {
        console.log(chalk.blue('🔍 SIRAJ MCP Server Debug Info:'));
        console.log(chalk.gray(`  Python: ${this.config.pythonPath}`));
        console.log(chalk.gray(`  Server: ${serverScript}`));
        console.log(chalk.gray(`  Transport: ${this.config.transport}`));
        console.log(chalk.gray(`  Config: ${JSON.stringify(this.config, null, 2)}`));
        console.log('');
      }

      // Prepare arguments for Python server  
      const args = [
        '-m', 'server.main_mcp_server',
        '--transport', this.config.transport!,
      ];

      if (this.config.transport !== 'stdio') {
        args.push('--host', this.config.host!);
        args.push('--port', this.config.port!.toString());
      }

      if (this.config.configFile) {
        args.push('--config', this.config.configFile);
      }

      // Add SSL arguments for HTTPS transport
      if (this.config.transport === 'https' && this.config.ssl.enabled) {
        if (this.config.ssl.cert_file) {
          args.push('--ssl-cert', this.config.ssl.cert_file);
        }
        if (this.config.ssl.key_file) {
          args.push('--ssl-key', this.config.ssl.key_file);
        }
        if (this.config.ssl.ca_file) {
          args.push('--ssl-ca', this.config.ssl.ca_file);
        }
        args.push('--ssl-verify', this.config.ssl.verify_mode);
      }
      
      // Note: Python server doesn't currently support --debug flag

      // Launch Python server
      this.pythonProcess = spawn(this.config.pythonPath!, args, {
        stdio: this.config.transport === 'stdio' ? 'inherit' : 'pipe',
        cwd: join(__dirname, '../src'),
        env: {
          ...process.env,
          PYTHONPATH: `${join(__dirname, '..')}${process.platform === 'win32' ? ';' : ':'}${join(__dirname, '../src')}`,
          SIRAJ_DEBUG_MODE: this.config.debug ? 'true' : 'false',
          ...(this.config.transport === 'https' && this.config.ssl.enabled ? {
            SIRAJ_SSL_ENABLED: 'true',
            SIRAJ_SSL_CERT_FILE: this.config.ssl.cert_file || '',
            SIRAJ_SSL_KEY_FILE: this.config.ssl.key_file || '',
            SIRAJ_SSL_CA_FILE: this.config.ssl.ca_file || '',
            SIRAJ_SSL_VERIFY_MODE: this.config.ssl.verify_mode,
            SIRAJ_SSL_PROTOCOL: this.config.ssl.protocol_version,
          } : {})
        }
      });

      // Handle process events
      this.pythonProcess.on('error', (error) => {
        console.error(chalk.red(`❌ SIRAJ server error: ${error.message}`));
        process.exit(1);
      });

      this.pythonProcess.on('close', (code) => {
        if (code !== 0 && code !== null) {
          console.error(chalk.red(`❌ SIRAJ server exited with code ${code}`));
          process.exit(code);
        }
      });

      // For non-stdio transports, log server info
      if (this.config.transport !== 'stdio') {
        console.log(chalk.green('✅ SIRAJ v6.1 Computational Hermeneutics MCP Server'));
        const protocol = this.config.transport === 'https' ? 'https' : 'http';
        console.log(chalk.cyan(`🌐 Running on ${protocol}://${this.config.host}:${this.config.port}`));
        
        if (this.config.transport === 'https' && this.config.ssl.enabled) {
          console.log(chalk.green(`🔒 SSL/TLS enabled (${this.config.ssl.protocol_version})`));
          if (this.config.ssl.cert_file) {
            console.log(chalk.gray(`   Certificate: ${this.config.ssl.cert_file}`));
          }
          if (this.config.ssl.verify_mode !== 'none') {
            console.log(chalk.gray(`   Verification: ${this.config.ssl.verify_mode}`));
          }
        }
        
        console.log(chalk.yellow('📚 Advanced linguistic analysis with cultural sovereignty protection'));
        console.log('');
      }

      // Handle graceful shutdown
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

    } catch (error) {
      console.error(chalk.red(`❌ Failed to start SIRAJ server: ${(error as Error).message}`));
      process.exit(1);
    }
  }

  /**
   * Shutdown the server gracefully
   */
  private shutdown(): void {
    if (this.pythonProcess) {
      console.log(chalk.yellow('\n🛑 Shutting down SIRAJ server...'));
      this.pythonProcess.kill('SIGTERM');
      
      // Force kill after 5 seconds
      setTimeout(() => {
        if (this.pythonProcess && !this.pythonProcess.killed) {
          this.pythonProcess.kill('SIGKILL');
        }
        process.exit(0);
      }, 5000);
    } else {
      process.exit(0);
    }
  }
}

/**
 * CLI Implementation
 */
function setupCLI(): Command {
  const program = new Command();

  program
    .name('siraj-mcp-server')
    .description('SIRAJ v6.1 Computational Hermeneutics MCP Server')
    .version('6.1.0');

  // Main server command
  program
    .command('start', { isDefault: true })
    .description('Start the SIRAJ MCP server')
    .option('-t, --transport <type>', 'Transport protocol (stdio|sse|https)', 'stdio')
    .option('-p, --port <number>', 'Server port (for sse/https)', process.env.NODE_ENV === 'production' && process.platform !== 'win32' ? '3443' : '3001')
    .option('-h, --host <address>', 'Server host (for sse/https)', 'localhost')
    .option('--python-path <path>', 'Path to Python executable', 'python')
    .option('-c, --config <file>', 'Configuration file path')
    .option('-d, --debug', 'Enable debug mode', false)
    .option('--log-level <level>', 'Log level (DEBUG|INFO|WARNING|ERROR)', 'INFO')
    .option('--ssl-cert <file>', 'SSL certificate file (for https transport)')
    .option('--ssl-key <file>', 'SSL private key file (for https transport)')
    .option('--ssl-ca <file>', 'SSL CA certificate file (for https transport)')
    .option('--ssl-verify <mode>', 'SSL verification mode (none|optional|required)', 'required')
    .action(async (options) => {
      const config = ConfigValidator.validateServerConfig({
        transport: options.transport,
        port: parseInt(options.port),
        host: options.host,
        pythonPath: options.pythonPath,
        configFile: options.config,
        debug: options.debug,
        logLevel: options.logLevel,
        ssl: {
          enabled: options.transport === 'https',
          cert_file: options.sslCert,
          key_file: options.sslKey,
          ca_file: options.sslCa,
          verify_mode: options.sslVerify
        }
      });

      const server = new SirajMCPServer(config);
      await server.start();
    });

  // Install command for dependencies
  program
    .command('install')
    .description('Install Python dependencies for SIRAJ server')
    .option('--python-path <path>', 'Path to Python executable', 'python')
    .action(async (options) => {
      console.log(chalk.blue('📦 Installing SIRAJ Python dependencies...'));
      
      const requirementsPath = join(__dirname, '../requirements.txt');
      if (!existsSync(requirementsPath)) {
        console.error(chalk.red('❌ requirements.txt not found'));
        process.exit(1);
      }

      const installProcess = spawn(options.pythonPath, ['-m', 'pip', 'install', '-r', requirementsPath], {
        stdio: 'inherit'
      });

      installProcess.on('close', (code) => {
        if (code === 0) {
          console.log(chalk.green('✅ Dependencies installed successfully'));
        } else {
          console.error(chalk.red(`❌ Installation failed with code ${code}`));
          process.exit(code);
        }
      });
    });

  // Health check command
  program
    .command('health')
    .description('Check SIRAJ server health and dependencies')
    .option('--python-path <path>', 'Path to Python executable', 'python')
    .action(async (options) => {
      console.log(chalk.blue('🏥 SIRAJ Health Check'));
      console.log('');

      // Check Python
      try {
        const pythonCheck = spawn(options.pythonPath, ['--version'], { stdio: 'pipe' });
        pythonCheck.stdout.on('data', (data) => {
          console.log(chalk.green(`✅ Python: ${data.toString().trim()}`));
        });
        
        pythonCheck.on('close', (code) => {
          if (code !== 0) {
            console.log(chalk.red('❌ Python: Not found or invalid'));
          }
        });
      } catch (error) {
        console.log(chalk.red('❌ Python: Not found'));
      }

      // Check Python server
      const serverScript = join(__dirname, '../src/server/main_mcp_server.py');
      if (existsSync(serverScript)) {
        console.log(chalk.green(`✅ Server: Found at ${serverScript}`));
      } else {
        console.log(chalk.red('❌ Server: Python server not found'));
      }

      // Check key dependencies
      const deps = ['mcp', 'transformers', 'torch', 'fastapi', 'pydantic'];
      console.log(chalk.blue('\n📋 Checking Python dependencies:'));
      
      for (const dep of deps) {
        const checkProcess = spawn(options.pythonPath, ['-c', `import ${dep}; print("${dep}: OK")`], { stdio: 'pipe' });
        
        checkProcess.stdout.on('data', (data) => {
          console.log(chalk.green(`  ✅ ${data.toString().trim()}`));
        });
        
        checkProcess.stderr.on('data', () => {
          console.log(chalk.red(`  ❌ ${dep}: Missing or invalid`));
        });
      }
    });

  // Configuration generator command
  program
    .command('generate-config')
    .description('Generate Claude Desktop configuration examples')
    .option('-o, --output <file>', 'Output file path')
    .option('--format <type>', 'Configuration format (basic|debug|sse|performance)', 'basic')
    .action(async (options) => {
      const configs = ConfigValidator.generateExampleConfigs();
      const selectedConfig = configs[options.format as keyof typeof configs] || configs.basic;
      
      const output = JSON.stringify(selectedConfig, null, 2);
      
      if (options.output) {
        const fs = await import('fs/promises');
        await fs.writeFile(options.output, output, 'utf8');
        console.log(chalk.green(`✅ Configuration written to ${options.output}`));
      } else {
        console.log(chalk.blue('📋 Claude Desktop Configuration:'));
        console.log(output);
      }
      
      console.log(chalk.yellow('\n💡 Usage:'));
      console.log(chalk.gray('  Add this configuration to your Claude Desktop config file'));
      console.log(chalk.gray('  Usually located at: ~/.config/claude-desktop/config.json'));
    });

  return program;
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  try {
    console.log(chalk.bold.blue('🧠 SIRAJ v6.1 Computational Hermeneutics MCP Server'));
    console.log(chalk.gray('   Advanced linguistic analysis with cultural sovereignty protection'));
    console.log('');

    const program = setupCLI();
    await program.parseAsync(process.argv);
  } catch (error) {
    console.error(chalk.red(`❌ Error: ${(error as Error).message}`));
    process.exit(1);
  }
}

// Run if this is the main module
if (process.argv[1] && process.argv[1].endsWith('index.js')) {
  main().catch(console.error);
}

export { SirajMCPServer, main, setupCLI };
export default SirajMCPServer;