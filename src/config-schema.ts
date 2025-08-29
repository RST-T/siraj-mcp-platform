/**
 * SIRAJ MCP Server Configuration Schema
 * 
 * Defines the configuration options for the SIRAJ v6.1 Computational Hermeneutics MCP Server
 * This schema is used for validation and type safety when configuring the server
 */

import { z } from 'zod';

/**
 * Transport protocol options
 */
export const TransportSchema = z.enum(['stdio', 'sse', 'https']);

/**
 * Log level options
 */
export const LogLevelSchema = z.enum(['DEBUG', 'INFO', 'WARNING', 'ERROR']);

/**
 * Cultural sovereignty configuration
 */
export const CulturalSovereigntyConfigSchema = z.object({
  enable_cultural_boundaries: z.boolean().default(true),
  require_community_validation: z.boolean().default(true),
  cultural_sensitivity_threshold: z.number().min(0).max(1).default(0.8),
  protected_cultural_contexts: z.array(z.string()).default([
    'sacred_texts',
    'religious_content',
    'cultural_heritage',
    'indigenous_knowledge'
  ]),
}).strict();

/**
 * Performance optimization configuration
 */
export const PerformanceConfigSchema = z.object({
  cache_enabled: z.boolean().default(true),
  cache_size_limit: z.number().positive().default(1000),
  similarity_threshold: z.number().min(0).max(1).default(0.85),
  batch_processing_enabled: z.boolean().default(true),
  max_batch_size: z.number().positive().default(50),
}).strict();

/**
 * Database configuration
 */
export const DatabaseConfigSchema = z.object({
  corpus_db_path: z.string().default('data/corpora'),
  lexicon_db_path: z.string().default('data/lexicons.db'),
  enable_caching: z.boolean().default(true),
  connection_pool_size: z.number().positive().default(10),
}).strict();

/**
 * SSL/TLS configuration
 */
export const SSLConfigSchema = z.object({
  enabled: z.boolean().default(false),
  cert_file: z.string().optional(),
  key_file: z.string().optional(),
  ca_file: z.string().optional(),
  verify_mode: z.enum(['none', 'optional', 'required']).default('required'),
  protocol_version: z.enum(['TLSv1.2', 'TLSv1.3']).default('TLSv1.3'),
  cipher_suites: z.array(z.string()).optional(),
  dhparam_file: z.string().optional(),
}).strict();

/**
 * Server configuration schema
 */
export const ServerConfigSchema = z.object({
  // Basic server settings
  transport: TransportSchema.default('stdio'),
  host: z.string().default('localhost'),
  port: z.number().int().positive().max(65535).default(3001),
  
  // Python environment
  pythonPath: z.string().default('python'),
  
  // Logging
  logLevel: LogLevelSchema.default('INFO'),
  debug: z.boolean().default(false),
  
  // Configuration file
  configFile: z.string().optional(),
  
  // SIRAJ specific settings
  cultural_sovereignty: CulturalSovereigntyConfigSchema.default({}),
  performance: PerformanceConfigSchema.default({}),
  database: DatabaseConfigSchema.default({}),
  ssl: SSLConfigSchema.default({}),
  
  // Advanced settings
  enable_community_validation: z.boolean().default(true),
  max_request_size: z.number().positive().default(1048576), // 1MB
  request_timeout: z.number().positive().default(30000), // 30 seconds
  
}).strict();

/**
 * Claude Desktop MCP server configuration schema
 */
export const ClaudeDesktopConfigSchema = z.object({
  mcpServers: z.record(z.object({
    command: z.string(),
    args: z.array(z.string()).optional(),
    env: z.record(z.string()).optional(),
  }))
});

/**
 * Type definitions
 */
export type ServerConfig = z.infer<typeof ServerConfigSchema>;
export type TransportType = z.infer<typeof TransportSchema>;
export type LogLevel = z.infer<typeof LogLevelSchema>;
export type CulturalSovereigntyConfig = z.infer<typeof CulturalSovereigntyConfigSchema>;
export type PerformanceConfig = z.infer<typeof PerformanceConfigSchema>;
export type DatabaseConfig = z.infer<typeof DatabaseConfigSchema>;
export type SSLConfig = z.infer<typeof SSLConfigSchema>;
export type ClaudeDesktopConfig = z.infer<typeof ClaudeDesktopConfigSchema>;

/**
 * Default configuration for SIRAJ MCP Server
 */
export const defaultServerConfig: ServerConfig = ServerConfigSchema.parse({});

/**
 * Configuration validation utilities
 */
export class ConfigValidator {
  /**
   * Validate server configuration
   */
  static validateServerConfig(config: unknown): ServerConfig {
    return ServerConfigSchema.parse(config);
  }
  
  /**
   * Validate Claude Desktop configuration
   */
  static validateClaudeDesktopConfig(config: unknown): ClaudeDesktopConfig {
    return ClaudeDesktopConfigSchema.parse(config);
  }
  
  /**
   * Generate Claude Desktop configuration for SIRAJ MCP Server
   */
  static generateClaudeDesktopConfig(
    serverConfig: Partial<ServerConfig> = {}
  ): ClaudeDesktopConfig {
    const config = { ...defaultServerConfig, ...serverConfig };
    
    return {
      mcpServers: {
        'siraj-computational-hermeneutics': {
          command: 'siraj-mcp-server',
          args: [
            'start',
            '--transport', config.transport,
            '--log-level', config.logLevel,
            ...(config.debug ? ['--debug'] : []),
            ...(config.configFile ? ['--config', config.configFile] : []),
            ...(config.transport !== 'stdio' ? [
              '--host', config.host,
              '--port', config.port.toString()
            ] : []),
            ...(config.transport === 'https' && config.ssl.enabled ? [
              '--ssl-cert', config.ssl.cert_file || 'certs/server.crt',
              '--ssl-key', config.ssl.key_file || 'certs/server.key'
            ] : []),
          ],
          env: {
            SIRAJ_DEBUG_MODE: config.debug ? 'true' : 'false',
            ...(config.transport !== 'stdio' ? {
              SIRAJ_HOST: config.host,
              SIRAJ_PORT: config.port.toString(),
            } : {}),
            ...(config.transport === 'https' && config.ssl.enabled ? {
              SIRAJ_SSL_ENABLED: 'true',
              SIRAJ_SSL_CERT_FILE: config.ssl.cert_file || 'certs/server.crt',
              SIRAJ_SSL_KEY_FILE: config.ssl.key_file || 'certs/server.key',
              SIRAJ_SSL_VERIFY_MODE: config.ssl.verify_mode,
              SIRAJ_SSL_PROTOCOL: config.ssl.protocol_version,
            } : {}),
          }
        }
      }
    };
  }
  
  /**
   * Generate example configuration files
   */
  static generateExampleConfigs() {
    return {
      // Basic stdio configuration (recommended)
      basic: this.generateClaudeDesktopConfig(),
      
      // Debug configuration
      debug: this.generateClaudeDesktopConfig({ debug: true, logLevel: 'DEBUG' }),
      
      // SSE configuration
      sse: this.generateClaudeDesktopConfig({ 
        transport: 'sse', 
        host: 'localhost', 
        port: 3001 
      }),
      
      // HTTPS configuration
      https: this.generateClaudeDesktopConfig({ 
        transport: 'https', 
        host: 'localhost', 
        port: 3443,
        ssl: {
          enabled: true,
          cert_file: 'certs/server.crt',
          key_file: 'certs/server.key',
          verify_mode: 'required',
          protocol_version: 'TLSv1.3'
        }
      }),
      
      // High-performance configuration
      performance: this.generateClaudeDesktopConfig({
        performance: {
          cache_enabled: true,
          cache_size_limit: 2000,
          similarity_threshold: 0.8,
          batch_processing_enabled: true,
          max_batch_size: 100,
        }
      }),
    };
  }
}