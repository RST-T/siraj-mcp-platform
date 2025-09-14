#!/bin/bash
export SIRAJ_CORPUS_DATABASE_URL="postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-little-hill-a8to8sbf-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
cd ./src/server
python main_mcp_server.py --transport stdio