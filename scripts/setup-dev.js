#!/usr/bin/env node

/**
 * Development Setup Script for SIRAJ MCP Server
 * 
 * This script helps developers set up the SIRAJ MCP server for development
 * and testing purposes.
 */

import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

/**
 * Execute a command and return a promise
 */
function runCommand(command, args = [], options = {}) {
  return new Promise((resolve, reject) => {
    console.log(chalk.blue(`🔧 Running: ${command} ${args.join(' ')}`));
    
    const child = spawn(command, args, {
      stdio: 'inherit',
      cwd: rootDir,
      shell: process.platform === 'win32',
      ...options
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });
    
    child.on('error', reject);
  });
}

/**
 * Check if a command exists
 */
function checkCommand(command) {
  return new Promise((resolve) => {
    const child = spawn(command, ['--version'], {
      stdio: 'pipe',
      shell: process.platform === 'win32'
    });
    
    child.on('close', (code) => resolve(code === 0));
    child.on('error', () => resolve(false));
  });
}

/**
 * Main setup function
 */
async function setupDevelopment() {
  console.log(chalk.bold.blue('🧠 SIRAJ v6.1 MCP Server Development Setup'));
  console.log(chalk.gray('   Setting up development environment...'));
  console.log('');
  
  try {
    // Check prerequisites
    console.log(chalk.yellow('📋 Checking prerequisites...'));
    
    const nodeExists = await checkCommand('node');
    if (!nodeExists) {
      throw new Error('Node.js not found. Please install Node.js 18+ from https://nodejs.org');
    }
    console.log(chalk.green('✅ Node.js found'));
    
    const pythonExists = await checkCommand('python');
    if (!pythonExists) {
      throw new Error('Python not found. Please install Python 3.10+ from https://python.org');
    }
    console.log(chalk.green('✅ Python found'));
    
    // Install NPM dependencies
    console.log(chalk.yellow('📦 Installing NPM dependencies...'));
    await runCommand('npm', ['install']);
    console.log(chalk.green('✅ NPM dependencies installed'));
    
    // Build TypeScript
    console.log(chalk.yellow('🏗️ Building TypeScript...'));
    await runCommand('npm', ['run', 'build']);
    console.log(chalk.green('✅ TypeScript built'));
    
    // Install Python dependencies
    console.log(chalk.yellow('🐍 Installing Python dependencies...'));
    const requirementsPath = join(rootDir, 'requirements.txt');
    if (existsSync(requirementsPath)) {
      await runCommand('python', ['-m', 'pip', 'install', '-r', 'requirements.txt']);
      console.log(chalk.green('✅ Python dependencies installed'));
    } else {
      console.log(chalk.yellow('⚠️ requirements.txt not found, skipping Python dependencies'));
    }
    
    // Run health check
    console.log(chalk.yellow('🏥 Running health check...'));
    await runCommand('node', ['dist/index.js', 'health']);
    
    console.log('');
    console.log(chalk.green('🎉 Development setup completed successfully!'));
    console.log('');
    console.log(chalk.yellow('📋 Available commands:'));
    console.log(chalk.cyan('  node dist/index.js --help') + chalk.gray(' - Show all available commands'));
    console.log(chalk.cyan('  node dist/index.js start') + chalk.gray(' - Start the MCP server'));
    console.log(chalk.cyan('  node dist/index.js health') + chalk.gray(' - Check server health'));
    console.log(chalk.cyan('  node dist/index.js generate-config') + chalk.gray(' - Generate Claude Desktop config'));
    console.log('');
    console.log(chalk.yellow('🔧 Development commands:'));
    console.log(chalk.cyan('  npm run dev') + chalk.gray(' - Start in development mode'));
    console.log(chalk.cyan('  npm run build') + chalk.gray(' - Build TypeScript'));
    console.log(chalk.cyan('  npm run lint') + chalk.gray(' - Lint code'));
    console.log(chalk.cyan('  npm test') + chalk.gray(' - Run tests'));
    
  } catch (error) {
    console.error(chalk.red(`❌ Setup failed: ${error.message}`));
    process.exit(1);
  }
}

// Run setup if this script is executed directly
if (import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}`) {
  setupDevelopment().catch(console.error);
}

export default setupDevelopment;