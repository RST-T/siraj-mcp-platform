#!/usr/bin/env node

/**
 * MCP Integration Test Script
 * 
 * This script tests the SIRAJ MCP server integration to ensure it works
 * correctly with Claude Desktop's `claude mcp add` functionality.
 */

import { spawn } from 'child_process';
import { existsSync, writeFileSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Test MCP server can be spawned correctly
 */
async function testMCPSpawn() {
  console.log(chalk.blue('🔧 Testing MCP server spawn...'));
  
  return new Promise((resolve, reject) => {
    const serverProcess = spawn('node', ['dist/index.js', 'start'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    let hasInitialized = false;
    
    const timeout = setTimeout(() => {
      serverProcess.kill('SIGTERM');
      if (hasInitialized) {
        resolve(true);
      } else {
        reject(new Error('Server did not initialize within timeout'));
      }
    }, 10000); // 10 second timeout
    
    serverProcess.stdout.on('data', (data) => {
      output += data.toString();
      if (output.includes('SIRAJ v6.1 Computational Hermeneutics Engine initialized')) {
        hasInitialized = true;
        clearTimeout(timeout);
        serverProcess.kill('SIGTERM');
        resolve(true);
      }
    });
    
    serverProcess.stderr.on('data', (data) => {
      output += data.toString();
      if (output.includes('SIRAJ v6.1 Computational Hermeneutics Engine initialized')) {
        hasInitialized = true;
        clearTimeout(timeout);
        serverProcess.kill('SIGTERM');
        resolve(true);
      }
    });
    
    serverProcess.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    
    serverProcess.on('close', (code) => {
      clearTimeout(timeout);
      if (hasInitialized) {
        resolve(true);
      } else {
        reject(new Error(`Server exited with code ${code} before initialization`));
      }
    });
  });
}

/**
 * Test configuration generation
 */
async function testConfigGeneration() {
  console.log(chalk.blue('📋 Testing configuration generation...'));
  
  return new Promise((resolve, reject) => {
    const configProcess = spawn('node', ['dist/index.js', 'generate-config'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    
    configProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    configProcess.stderr.on('data', (data) => {
      output += data.toString();
    });
    
    configProcess.on('close', (code) => {
      if (code === 0 && output.includes('mcpServers') && output.includes('siraj-computational-hermeneutics')) {
        resolve(output);
      } else {
        reject(new Error(`Config generation failed with code ${code}`));
      }
    });
    
    configProcess.on('error', reject);
  });
}

/**
 * Test health check
 */
async function testHealthCheck() {
  console.log(chalk.blue('🏥 Testing health check...'));
  
  return new Promise((resolve, reject) => {
    const healthProcess = spawn('node', ['dist/index.js', 'health'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    
    healthProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    healthProcess.stderr.on('data', (data) => {
      output += data.toString();
    });
    
    healthProcess.on('close', (code) => {
      if (code === 0 && output.includes('✅')) {
        resolve(output);
      } else {
        reject(new Error(`Health check failed with code ${code}`));
      }
    });
    
    healthProcess.on('error', reject);
  });
}

/**
 * Test that package.json has correct MCP metadata
 */
function testPackageMetadata() {
  console.log(chalk.blue('📦 Testing package metadata...'));
  
  const packagePath = join(__dirname, 'package.json');
  if (!existsSync(packagePath)) {
    throw new Error('package.json not found');
  }
  
  const packageData = JSON.parse(readFileSync(packagePath, 'utf8'));
  
  // Check required fields for MCP
  const requiredFields = ['name', 'version', 'bin', 'main'];
  for (const field of requiredFields) {
    if (!packageData[field]) {
      throw new Error(`Missing required field: ${field}`);
    }
  }
  
  // Check MCP-specific metadata
  if (!packageData.mcp || !packageData.mcp.server) {
    throw new Error('Missing MCP server metadata');
  }
  
  const mcpServer = packageData.mcp.server;
  if (!mcpServer.name || !mcpServer.capabilities || !mcpServer.transport) {
    throw new Error('Incomplete MCP server metadata');
  }
  
  console.log(chalk.green(`✅ Package metadata valid`));
  console.log(chalk.gray(`   Name: ${packageData.name}`));
  console.log(chalk.gray(`   Version: ${packageData.version}`));
  console.log(chalk.gray(`   MCP Server: ${mcpServer.name}`));
  console.log(chalk.gray(`   Capabilities: ${mcpServer.capabilities.join(', ')}`));
  console.log(chalk.gray(`   Transports: ${mcpServer.transport.join(', ')}`));
  
  return true;
}

/**
 * Simulate claude mcp add by testing the expected workflow
 */
async function testClaudeMCPAddWorkflow() {
  console.log(chalk.blue('🔌 Testing Claude MCP add workflow simulation...'));
  
  // Step 1: Generate configuration that claude mcp add would generate
  const configOutput = await testConfigGeneration();
  
  // Step 2: Parse the generated configuration
  const configMatch = configOutput.match(/{[\s\S]*}/);
  if (!configMatch) {
    throw new Error('Could not parse generated configuration');
  }
  
  const config = JSON.parse(configMatch[0]);
  
  // Step 3: Validate the configuration structure
  if (!config.mcpServers || !config.mcpServers['siraj-computational-hermeneutics']) {
    throw new Error('Invalid configuration structure');
  }
  
  const serverConfig = config.mcpServers['siraj-computational-hermeneutics'];
  
  // Step 4: Test that the command and args would work
  if (serverConfig.command !== 'siraj-mcp-server') {
    throw new Error('Incorrect command in configuration');
  }
  
  if (!serverConfig.args || !serverConfig.args.includes('start')) {
    throw new Error('Missing or incorrect args in configuration');
  }
  
  console.log(chalk.green('✅ Claude MCP add workflow simulation passed'));
  console.log(chalk.gray(`   Command: ${serverConfig.command}`));
  console.log(chalk.gray(`   Args: ${serverConfig.args.join(' ')}`));
  
  return config;
}

/**
 * Main test function
 */
async function runTests() {
  console.log(chalk.bold.blue('🧪 SIRAJ MCP Server Integration Tests'));
  console.log(chalk.gray('   Testing claude mcp add compatibility...'));
  console.log('');
  
  try {
    // Test 1: Package metadata
    testPackageMetadata();
    
    // Test 2: Health check
    await testHealthCheck();
    console.log(chalk.green('✅ Health check passed'));
    
    // Test 3: Configuration generation
    await testConfigGeneration();
    console.log(chalk.green('✅ Configuration generation passed'));
    
    // Test 4: MCP server spawn
    await testMCPSpawn();
    console.log(chalk.green('✅ MCP server spawn passed'));
    
    // Test 5: Claude MCP add workflow simulation
    const config = await testClaudeMCPAddWorkflow();
    
    console.log('');
    console.log(chalk.green('🎉 All integration tests passed!'));
    console.log('');
    console.log(chalk.yellow('📋 Integration Summary:'));
    console.log(chalk.cyan('✅ Package structure compatible with claude mcp add'));
    console.log(chalk.cyan('✅ MCP server starts and initializes correctly'));  
    console.log(chalk.cyan('✅ Configuration generation works'));
    console.log(chalk.cyan('✅ Health checks pass'));
    console.log(chalk.cyan('✅ All required metadata present'));
    console.log('');
    console.log(chalk.yellow('🚀 Ready for Claude Desktop integration!'));
    console.log('');
    console.log(chalk.gray('To integrate with Claude Desktop:'));
    console.log(chalk.white('1. Publish package: npm publish'));
    console.log(chalk.white('2. Install globally: npm install -g @siraj-team/mcp-server-computational-hermeneutics'));
    console.log(chalk.white('3. Add to Claude: claude mcp add @siraj-team/mcp-server-computational-hermeneutics'));
    
  } catch (error) {
    console.error(chalk.red(`❌ Test failed: ${error.message}`));
    process.exit(1);
  }
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}`) {
  runTests().catch(console.error);
}

export { runTests };