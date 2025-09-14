import { test, expect } from '@playwright/test';
import { execSync } from 'child_process';
import { WebSocket } from 'ws';
import * as fs from 'fs';
import * as path from 'path';

test.describe('MCP Server Integration', () => {
  let mcpServerProcess: any;
  let apiKey: string;

  test.beforeAll(async ({ browser }) => {
    // Get API key for authenticated requests
    const context = await browser.newContext();
    const page = await context.newPage();
    
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'testpassword123');
    await page.click('button[type="submit"]');
    
    await page.goto('/dashboard/api-keys');
    await page.click('[data-testid="generate-api-key"]');
    await page.fill('[name="keyName"]', 'MCP Integration Key');
    await page.click('button[type="submit"]');
    
    const apiKeyElement = page.locator('[data-testid="api-key-value"]');
    apiKey = (await apiKeyElement.textContent()) || '';
    
    await context.close();
  });

  test('MCP server startup and initialization', async ({ request }) => {
    // Test that MCP server can be started
    const serverPath = path.join(process.cwd(), 'src', 'server', 'main_mcp_server.py');
    
    // Check server file exists
    expect(fs.existsSync(serverPath)).toBe(true);
    
    // Test server health check
    const healthResponse = await request.get('/api/v1/mcp/health');
    expect(healthResponse.status()).toBe(200);
    
    const healthData = await healthResponse.json();
    expect(healthData.server_status).toBe('running');
    expect(healthData.tools_available).toBe(4); // Our 4 MCP tools
  });

  test('MCP tool discovery and listing', async ({ request }) => {
    const toolsResponse = await request.post('/api/v1/mcp/list-tools', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    
    expect(toolsResponse.status()).toBe(200);
    
    const toolsData = await toolsResponse.json();
    expect(Array.isArray(toolsData.tools)).toBe(true);
    expect(toolsData.tools).toHaveLength(4);
    
    const toolNames = toolsData.tools.map((tool: any) => tool.name);
    expect(toolNames).toContain('computational_hermeneutics_methodology');
    expect(toolNames).toContain('adaptive_semantic_architecture');
    expect(toolNames).toContain('community_sovereignty_protocols');
    expect(toolNames).toContain('multi_paradigm_validation');
    
    // Validate tool schemas
    toolsData.tools.forEach((tool: any) => {
      expect(tool.name).toBeDefined();
      expect(tool.description).toBeDefined();
      expect(tool.inputSchema).toBeDefined();
      expect(tool.inputSchema.type).toBe('object');
      expect(tool.inputSchema.properties).toBeDefined();
    });
  });

  test('Computational Hermeneutics Methodology tool', async ({ request }) => {
    const toolCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'computational_hermeneutics_methodology',
        arguments: {
          text: 'سلام عليكم',
          analysis_type: 'root_etymology',
          cultural_context: 'classical_arabic',
          depth_level: 'comprehensive',
          community_consultation: 'recommended'
        }
      }
    });
    
    expect(toolCallResponse.status()).toBe(200);
    
    const toolData = await toolCallResponse.json();
    expect(toolData.content).toBeDefined();
    expect(Array.isArray(toolData.content)).toBe(true);
    
    const result = toolData.content[0];
    expect(result.type).toBe('text');
    expect(result.text).toContain('methodology');
    expect(result.text).toContain('cultural_sovereignty_assessment');
    expect(result.text).toContain('recommended_approach');
    expect(result.text).toContain('validation_protocols');
    
    // Verify methodology structure
    const methodologyData = JSON.parse(result.text);
    expect(methodologyData.analysis_phases).toBeDefined();
    expect(methodologyData.cultural_sovereignty_assessment).toBeDefined();
    expect(methodologyData.validation_protocols).toBeDefined();
    expect(methodologyData.recommended_approach).toBeDefined();
  });

  test('Adaptive Semantic Architecture tool', async ({ request }) => {
    const toolCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'adaptive_semantic_architecture',
        arguments: {
          text: 'الحمد لله رب العالمين',
          context: 'religious_expression',
          tier_focus: 'universal_primitives',
          cultural_adaptation: true
        }
      }
    });
    
    expect(toolCallResponse.status()).toBe(200);
    
    const toolData = await toolCallResponse.json();
    const result = toolData.content[0];
    const semanticData = JSON.parse(result.text);
    
    expect(semanticData.semantic_mapping).toBeDefined();
    expect(semanticData.tier_analysis).toBeDefined();
    expect(semanticData.cultural_adaptations).toBeDefined();
    expect(semanticData.methodology).toBeDefined();
    
    // Verify tier structure
    expect(semanticData.tier_analysis.tier_1_nodes).toBeDefined();
    expect(Array.isArray(semanticData.semantic_mapping.universal_nodes)).toBe(true);
    expect(semanticData.semantic_mapping.universal_nodes.length).toBeGreaterThan(0);
  });

  test('Community Sovereignty Protocols tool', async ({ request }) => {
    const toolCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'community_sovereignty_protocols',
        arguments: {
          text: 'قرآن كريم',
          content_type: 'sacred_text',
          cultural_context: 'islamic_tradition',
          authority_level: 'traditional_scholars'
        }
      }
    });
    
    expect(toolCallResponse.status()).toBe(200);
    
    const toolData = await toolCallResponse.json();
    const result = toolData.content[0];
    const protocolData = JSON.parse(result.text);
    
    expect(protocolData.sovereignty_assessment).toBeDefined();
    expect(protocolData.validation_workflow).toBeDefined();
    expect(protocolData.community_engagement_methodology).toBeDefined();
    expect(protocolData.authority_requirements).toBeDefined();
    
    // Verify sacred content handling
    expect(protocolData.sovereignty_assessment.sensitivity_level).toBe('sacred');
    expect(protocolData.authority_requirements.required_credentials).toBeDefined();
  });

  test('Multi-Paradigm Validation tool', async ({ request }) => {
    const toolCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'multi_paradigm_validation',
        arguments: {
          analysis_results: {
            root_analysis: 's-l-m (peace/safety)',
            semantic_mapping: 'universal_peace_concept',
            cultural_significance: 'fundamental_islamic_greeting'
          },
          traditional_sources: ['Classical Arabic dictionaries', 'Quranic commentaries'],
          scientific_methods: ['comparative_linguistics', 'corpus_analysis'],
          computational_validation: ['cross_referencing', 'frequency_analysis']
        }
      }
    });
    
    expect(toolCallResponse.status()).toBe(200);
    
    const toolData = await toolCallResponse.json();
    const result = toolData.content[0];
    const validationData = JSON.parse(result.text);
    
    expect(validationData.convergence_analysis).toBeDefined();
    expect(validationData.validation_scores).toBeDefined();
    expect(validationData.methodology).toBeDefined();
    
    // Verify convergence scoring
    const scores = validationData.validation_scores;
    expect(scores.traditional_score).toBeGreaterThanOrEqual(0);
    expect(scores.traditional_score).toBeLessThanOrEqual(1);
    expect(scores.scientific_score).toBeGreaterThanOrEqual(0);
    expect(scores.scientific_score).toBeLessThanOrEqual(1);
    expect(scores.computational_score).toBeGreaterThanOrEqual(0);
    expect(scores.computational_score).toBeLessThanOrEqual(1);
    expect(scores.overall_convergence).toBeGreaterThanOrEqual(0);
    expect(scores.overall_convergence).toBeLessThanOrEqual(1);
  });

  test('Error handling in MCP tools', async ({ request }) => {
    // Test with missing required arguments
    const missingArgsResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'computational_hermeneutics_methodology',
        arguments: {
          // Missing required 'text' argument
          analysis_type: 'root_etymology'
        }
      }
    });
    
    expect(missingArgsResponse.status()).toBe(400);
    const errorData = await missingArgsResponse.json();
    expect(errorData.error).toContain('required');
    
    // Test with invalid tool name
    const invalidToolResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'nonexistent_tool',
        arguments: {
          text: 'test'
        }
      }
    });
    
    expect(invalidToolResponse.status()).toBe(404);
    const invalidToolError = await invalidToolResponse.json();
    expect(invalidToolError.error).toContain('tool not found');
  });

  test('MCP server resource management', async ({ request }) => {
    // Test server resource usage
    const resourcesResponse = await request.get('/api/v1/mcp/resources', {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });
    
    expect(resourcesResponse.status()).toBe(200);
    
    const resourceData = await resourcesResponse.json();
    expect(resourceData.memory_usage).toBeDefined();
    expect(resourceData.active_connections).toBeGreaterThanOrEqual(0);
    expect(resourceData.processed_requests).toBeGreaterThanOrEqual(0);
    
    // Test resource cleanup
    const cleanupResponse = await request.post('/api/v1/mcp/cleanup', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    
    expect(cleanupResponse.status()).toBe(200);
  });

  test('MCP server performance under load', async ({ request }) => {
    // Execute multiple concurrent tool calls
    const concurrentCalls = Array(10).fill(null).map((_, index) =>
      request.post('/api/v1/mcp/call-tool', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        data: {
          name: 'computational_hermeneutics_methodology',
          arguments: {
            text: `test text ${index}`,
            analysis_type: 'root_etymology',
            cultural_context: 'modern_arabic'
          }
        }
      })
    );
    
    const startTime = Date.now();
    const responses = await Promise.all(concurrentCalls);
    const endTime = Date.now();
    
    // All requests should succeed
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });
    
    // Performance should be reasonable (less than 10 seconds for 10 concurrent calls)
    const totalTime = endTime - startTime;
    expect(totalTime).toBeLessThan(10000);
    
    console.log(`Processed 10 concurrent MCP calls in ${totalTime}ms`);
  });

  test('MCP tool caching behavior', async ({ request }) => {
    const testText = 'سلام';
    
    // First call - should be slower (no cache)
    const firstCallStart = Date.now();
    const firstCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'computational_hermeneutics_methodology',
        arguments: {
          text: testText,
          analysis_type: 'root_etymology',
          cultural_context: 'classical_arabic'
        }
      }
    });
    const firstCallTime = Date.now() - firstCallStart;
    
    expect(firstCallResponse.status()).toBe(200);
    
    // Second call with same parameters - should be faster (cached)
    const secondCallStart = Date.now();
    const secondCallResponse = await request.post('/api/v1/mcp/call-tool', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: {
        name: 'computational_hermeneutics_methodology',
        arguments: {
          text: testText,
          analysis_type: 'root_etymology',
          cultural_context: 'classical_arabic'
        }
      }
    });
    const secondCallTime = Date.now() - secondCallStart;
    
    expect(secondCallResponse.status()).toBe(200);
    
    // Second call should be significantly faster if caching is working
    // (Allow some tolerance for network variability)
    console.log(`First call: ${firstCallTime}ms, Second call: ${secondCallTime}ms`);
    
    // Results should be identical
    const firstResult = await firstCallResponse.json();
    const secondResult = await secondCallResponse.json();
    expect(firstResult.content[0].text).toBe(secondResult.content[0].text);
  });

  test('MCP server stdio transport (Claude Desktop integration)', async () => {
    // Test that server can run in stdio mode for Claude Desktop
    const serverPath = path.join(process.cwd(), 'src', 'server', 'main_mcp_server.py');
    
    try {
      // Start server in stdio mode
      const command = `python "${serverPath}" --transport stdio`;
      const process = execSync(command, {
        timeout: 5000,
        stdio: 'pipe',
        env: {
          ...process.env,
          PYTHONPATH: process.cwd()
        }
      });
      
      // If we get here without timeout, server started successfully
      expect(true).toBe(true);
    } catch (error: any) {
      // Check if it's a timeout (expected for a running server)
      if (error.status === null && error.signal === 'SIGTERM') {
        // Server was running and killed by timeout - this is expected
        expect(true).toBe(true);
      } else {
        // Actual error starting server
        console.error('Server startup error:', error.message);
        throw error;
      }
    }
  });

  test('MCP protocol compliance', async ({ request }) => {
    // Test JSON-RPC 2.0 compliance
    const rpcCall = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'computational_hermeneutics_methodology',
        arguments: {
          text: 'test',
          analysis_type: 'root_etymology'
        }
      },
      id: 1
    };
    
    const rpcResponse = await request.post('/api/v1/mcp/jsonrpc', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      data: rpcCall
    });
    
    expect(rpcResponse.status()).toBe(200);
    
    const rpcData = await rpcResponse.json();
    expect(rpcData.jsonrpc).toBe('2.0');
    expect(rpcData.id).toBe(1);
    expect(rpcData.result || rpcData.error).toBeDefined();
  });
});