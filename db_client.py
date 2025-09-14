#!/usr/bin/env python3
import psycopg2
import sys

# Connection string
CONNECTION_STRING = "postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-icy-wave-a8h14f5w-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"

def execute_query(query):
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Check if it's a SELECT query
        if query.strip().upper().startswith('SELECT') or query.strip().upper().startswith('SHOW') or query.strip().upper().startswith('\\'):
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            if column_names:
                print(" | ".join(column_names))
                print("-" * (sum(len(name) for name in column_names) + 3 * (len(column_names) - 1)))
            
            for row in results:
                print(" | ".join(str(col) for col in row))
        else:
            conn.commit()
            print(f"Query executed successfully. Rows affected: {cursor.rowcount}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        execute_query(query)
    else:
        print("Usage: python db_client.py 'SQL_QUERY'")
        print("Example: python db_client.py 'SELECT * FROM information_schema.tables LIMIT 5;'")