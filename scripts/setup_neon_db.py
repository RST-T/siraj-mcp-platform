#!/usr/bin/env python3
"""
Set up SIRAJ database schema in Neon using direct PostgreSQL connection
"""

import psycopg2
import json
from pathlib import Path
import sys

# Connection string
CONNECTION_STRING = "postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-little-hill-a8to8sbf-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"

def test_connection():
    """Test database connection"""
    try:
        print("Testing connection to Neon database...")
        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"[+] Connected successfully!")
        print(f"[+] PostgreSQL version: {version[:100]}...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"[-] Connection failed: {e}")
        return False

def create_siraj_schema():
    """Create SIRAJ database schema"""
    try:
        print("Creating SIRAJ database schema...")
        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        
        # SIRAJ v6.1 Schema
        schema_sql = """
        -- SIRAJ v6.1 Computational Hermeneutics Database Schema
        
        -- Drop tables if they exist (for clean setup)
        DROP TABLE IF EXISTS quranic_verses CASCADE;
        DROP TABLE IF EXISTS hadith_collection CASCADE;
        DROP TABLE IF EXISTS classical_texts CASCADE;
        DROP TABLE IF EXISTS root_etymologies CASCADE;
        
        -- Quranic verses table
        CREATE TABLE quranic_verses (
            id SERIAL PRIMARY KEY,
            surah_number INTEGER NOT NULL,
            verse_number INTEGER NOT NULL,
            arabic_text TEXT NOT NULL,
            transliteration TEXT,
            translation_en TEXT,
            revelation_context TEXT,
            linguistic_features JSONB DEFAULT '{}',
            root_analysis JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(surah_number, verse_number)
        );
        
        -- Hadith collection table
        CREATE TABLE hadith_collection (
            id SERIAL PRIMARY KEY,
            collection_name VARCHAR(100) NOT NULL,
            book_number INTEGER,
            hadith_number INTEGER,
            arabic_text TEXT NOT NULL,
            english_text TEXT,
            narrator_chain TEXT,
            authenticity_grade VARCHAR(50),
            subject_tags TEXT[],
            linguistic_analysis JSONB DEFAULT '{}',
            root_occurrences JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Classical Arabic literature table
        CREATE TABLE classical_texts (
            id SERIAL PRIMARY KEY,
            work_title VARCHAR(255) NOT NULL,
            author_name VARCHAR(255) NOT NULL,
            period VARCHAR(100),
            genre VARCHAR(100),
            text_excerpt TEXT NOT NULL,
            linguistic_features JSONB DEFAULT '{}',
            cultural_context JSONB DEFAULT '{}',
            root_analysis JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Root etymology table
        CREATE TABLE root_etymologies (
            id SERIAL PRIMARY KEY,
            root_form VARCHAR(20) NOT NULL UNIQUE,
            language_family VARCHAR(50) NOT NULL,
            proto_form VARCHAR(50),
            semantic_field VARCHAR(100),
            core_meaning TEXT NOT NULL,
            derived_meanings JSONB DEFAULT '{}',
            cognates JSONB DEFAULT '{}',
            historical_development JSONB DEFAULT '{}',
            scholarly_consensus JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create performance indexes
        CREATE INDEX idx_quranic_verses_surah ON quranic_verses (surah_number);
        CREATE INDEX idx_quranic_verses_roots ON quranic_verses USING GIN (root_analysis);
        CREATE INDEX idx_hadith_collection ON hadith_collection (collection_name);
        CREATE INDEX idx_hadith_roots ON hadith_collection USING GIN (root_occurrences);
        CREATE INDEX idx_classical_texts_period ON classical_texts (period);
        CREATE INDEX idx_classical_texts_roots ON classical_texts USING GIN (root_analysis);
        CREATE INDEX idx_root_etymologies_family ON root_etymologies (language_family);
        CREATE INDEX idx_root_etymologies_field ON root_etymologies (semantic_field);
        """
        
        print("Executing schema creation...")
        cursor.execute(schema_sql)
        conn.commit()
        print("[+] Schema created successfully!")
        
        # Insert sample data
        print("Inserting sample data...")
        sample_data_sql = """
        -- Insert sample root etymologies for testing
        INSERT INTO root_etymologies (root_form, language_family, semantic_field, core_meaning, derived_meanings, cognates) VALUES
        ('ك-ت-ب', 'semitic', 'knowledge', 'writing, recording', 
         '{"book": "كتاب", "library": "مكتبة", "writer": "كاتب", "office": "مكتب"}', 
         '{"hebrew": "כתב", "aramaic": "כתב", "syriac": "ܟܬܒ"}'),
        
        ('د-ر-س', 'semitic', 'education', 'studying, learning', 
         '{"school": "مدرسة", "lesson": "درس", "teacher": "مدرس", "student": "دارس"}', 
         '{"hebrew": "דרש", "aramaic": "דרש"}'),
         
        ('ع-ل-م', 'semitic', 'knowledge', 'knowing, learning', 
         '{"science": "علم", "scholar": "عالم", "world": "عالم", "flag": "علم"}', 
         '{"hebrew": "עלם", "aramaic": "עלמא"}'),
         
        ('ق-ر-أ', 'semitic', 'communication', 'reading, reciting', 
         '{"quran": "قرآن", "reader": "قارئ", "reading": "قراءة"}', 
         '{"hebrew": "קרא", "aramaic": "קרא"}'),
         
        ('س-م-ع', 'semitic', 'perception', 'hearing, listening', 
         '{"hearing": "سمع", "listener": "سامع", "reputation": "سمعة"}', 
         '{"hebrew": "שמע", "aramaic": "שמע"}')
        ON CONFLICT (root_form) DO NOTHING;
        
        -- Insert sample Quranic verse
        INSERT INTO quranic_verses (surah_number, verse_number, arabic_text, transliteration, translation_en, root_analysis) VALUES
        (96, 1, 'اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ', 
         'Iqra bi-ismi rabbika alladhi khalaq', 
         'Read in the name of your Lord who created',
         '{"ق-ر-أ": {"forms": ["اقْرَأْ"], "meaning": "read"}, "ر-ب-ب": {"forms": ["رَبِّكَ"], "meaning": "lord"}, "خ-ل-ق": {"forms": ["خَلَقَ"], "meaning": "created"}}'
        ) ON CONFLICT (surah_number, verse_number) DO NOTHING;
        
        -- Insert sample Hadith
        INSERT INTO hadith_collection (collection_name, book_number, hadith_number, arabic_text, english_text, authenticity_grade, subject_tags) VALUES
        ('Bukhari', 1, 1, 
         'إنما الأعمال بالنيات وإنما لكل امرئ ما نوى', 
         'Actions are but by intention and every man shall have only that which he intended',
         'sahih', 
         ARRAY['intentions', 'actions', 'ethics']
        ) ON CONFLICT DO NOTHING;
        
        -- Insert sample classical text
        INSERT INTO classical_texts (work_title, author_name, period, genre, text_excerpt, cultural_context) VALUES
        ('Mu''allaqat', 'Imru al-Qays', 'pre_islamic', 'poetry',
         'قفا نبك من ذكرى حبيب ومنزل بسقط اللوى بين الدخول فحومل',
         '{"period": "Jahiliyyah", "tribe": "Kinda", "theme": "nasib"}'
        ) ON CONFLICT DO NOTHING;
        """
        
        cursor.execute(sample_data_sql)
        conn.commit()
        print("[+] Sample data inserted!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"[+] Tables created: {', '.join([table[0] for table in tables])}")
        
        # Get row counts
        for table_name, in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} rows")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"[-] Schema creation failed: {e}")
        return False

def update_siraj_config():
    """Update SIRAJ configuration with Neon connection"""
    try:
        print("Updating SIRAJ configuration...")
        
        # Create .env file
        env_path = Path("C:/Users/Admin/Documents/RST/Siraj-MCP/.env")
        
        env_content = f"""# SIRAJ v6.1 MCP Server Configuration
# Neon Database Connection

# Database Configuration
SIRAJ_CORPUS_DATABASE_URL={CONNECTION_STRING}
SIRAJ_LEXICON_DATABASE_URL=sqlite:///./data/lexicons.db
SIRAJ_CACHE_DATABASE_URL=redis://localhost:6379/0

# Server Configuration  
SIRAJ_HOST=localhost
SIRAJ_PORT=8000
SIRAJ_DEBUG_MODE=true
SIRAJ_LOG_LEVEL=INFO

# Security Configuration
SIRAJ_API_KEY_REQUIRED=false
SIRAJ_ENABLE_AUTHENTICATION=true

# Performance Configuration
SIRAJ_MAX_MEMORY_USAGE=2147483648
SIRAJ_REQUEST_TIMEOUT=120
SIRAJ_CONCURRENT_REQUESTS=10
"""
        
        env_path.write_text(env_content, encoding='utf-8')
        print(f"[+] Configuration saved to: {env_path}")
        
        # Create connection test script
        test_script_path = Path("C:/Users/Admin/Documents/RST/Siraj-MCP/test_neon.py")
        test_script = f'''#!/usr/bin/env python3
"""
Test SIRAJ v6.1 Neon Database Connection
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database.connection_manager import ConnectionManager

async def test_siraj_neon():
    """Test SIRAJ with Neon database"""
    print("Testing SIRAJ v6.1 with Neon database...")
    
    try:
        conn_mgr = ConnectionManager()
        await conn_mgr.initialize()
        
        health = await conn_mgr.health_check()
        print("Database Health Check:")
        
        for db_type, status in health["status"].items():
            emoji = "[+]" if status else "[-]"
            detail = health["details"].get(db_type, "Unknown")
            print(f"  {{emoji}} {{db_type.capitalize()}}: {{detail}}")
        
        if health["status"]["corpus"]:
            print("\\n[SUCCESS] SIRAJ v6.1 is ready with Neon database!")
            return True
        else:
            print("\\n[WARNING] Some database connections failed")
            return False
            
    except Exception as e:
        print(f"\\n[-] Test failed: {{e}}")
        return False
    finally:
        try:
            await conn_mgr.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = asyncio.run(test_siraj_neon())
    sys.exit(0 if success else 1)
'''
        
        test_script_path.write_text(test_script, encoding='utf-8')
        print(f"[+] Test script created: {test_script_path}")
        
        return True
        
    except Exception as e:
        print(f"[-] Configuration update failed: {e}")
        return False

def main():
    """Main setup function"""
    print("SIRAJ v6.1 Neon Database Setup")
    print("=" * 50)
    
    # Test connection
    if not test_connection():
        print("Setup failed - cannot connect to database")
        return False
    
    # Create schema  
    if not create_siraj_schema():
        print("Setup failed - schema creation failed")
        return False
    
    # Update configuration
    if not update_siraj_config():
        print("Setup failed - configuration update failed")
        return False
    
    print("\n" + "=" * 50)
    print("SIRAJ v6.1 NEON SETUP COMPLETE!")
    print("=" * 50)
    print("[+] Database connected and schema created")
    print("[+] Sample data inserted for testing")
    print("[+] Configuration files updated")
    print("[+] Test script created")
    print("\nNext Steps:")
    print("1. Test connection: python test_neon.py")
    print("2. Start server: python src/server/main.py --protocol both")
    print("3. Test API: curl http://localhost:8000/health")
    print("4. Try analysis: curl -X POST http://localhost:8000/analysis -H 'Content-Type: application/json' -d '{\"root\":\"ك-ت-ب\",\"contexts\":{\"test\":\"writing and books\"}}'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)