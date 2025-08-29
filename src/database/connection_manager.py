"""
Database Connection Manager for SIRAJ v6.1
Handles connections to PostgreSQL (corpus), SQLite (lexicons), and Redis (cache)
"""

import asyncio
import asyncpg
import aiosqlite
import redis.asyncio as redis
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

from config.settings import settings
from src.utils.exceptions import SirajProcessingError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration container"""
    corpus_db_url: str
    lexicon_db_url: str  
    cache_db_url: str
    pool_min_size: int = 5
    pool_max_size: int = 20
    connection_timeout: int = 30
    query_timeout: int = 120


class ConnectionManager:
    """
    Centralized database connection manager for all SIRAJ data sources
    
    Manages connections to:
    - PostgreSQL: Corpus data (Quranic verses, Hadith, classical texts)
    - SQLite: Lexicon data (dictionaries, etymologies)  
    - Redis: Performance cache and session storage
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig(
            corpus_db_url=settings.corpus_database_url,
            lexicon_db_url=settings.lexicon_database_url,
            cache_db_url=settings.cache_database_url
        )
        
        # Connection pools
        self._corpus_pool: Optional[asyncpg.Pool] = None
        self._cache_client: Optional[redis.Redis] = None
        self._lexicon_connections: Dict[str, aiosqlite.Connection] = {}
        
        # Connection status tracking
        self._connection_status = {
            "corpus": False,
            "lexicon": False, 
            "cache": False
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all database connections"""
        if self._initialized:
            return
            
        logger.info("Initializing database connections...")
        
        try:
            # Initialize corpus database (PostgreSQL)
            await self._init_corpus_db()
            
            # Initialize cache database (Redis)  
            await self._init_cache_db()
            
            # Initialize lexicon database (SQLite)
            await self._init_lexicon_db()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise SirajProcessingError(f"Failed to initialize database connections: {str(e)}")
    
    async def _init_corpus_db(self) -> None:
        """Initialize PostgreSQL corpus database connection"""
        try:
            logger.info("Connecting to corpus database (PostgreSQL)...")
            
            self._corpus_pool = await asyncpg.create_pool(
                self.config.corpus_db_url,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.query_timeout,
                server_settings={
                    'application_name': 'siraj-mcp-v6.1',
                    'timezone': 'UTC'
                }
            )
            
            # Test connection with a simple query
            async with self._corpus_pool.acquire() as conn:
                result = await conn.fetchrow("SELECT version() as version")
                logger.info(f"Corpus database connected: {result['version']}")
            
            self._connection_status["corpus"] = True
            
        except Exception as e:
            logger.error(f"Corpus database connection failed: {str(e)}")
            # For development, create tables if database exists but is empty
            if "does not exist" not in str(e).lower():
                await self._create_corpus_tables()
            raise
    
    async def _init_cache_db(self) -> None:
        """Initialize Redis cache database connection"""
        try:
            logger.info("Connecting to cache database (Redis)...")
            
            self._cache_client = redis.from_url(
                self.config.cache_db_url,
                decode_responses=True,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.query_timeout
            )
            
            # Test connection
            await self._cache_client.ping()
            info = await self._cache_client.info()
            logger.info(f"Cache database connected: Redis {info.get('redis_version')}")
            
            self._connection_status["cache"] = True
            
        except Exception as e:
            logger.error(f"Cache database connection failed: {str(e)}")
            # Continue without cache in development
            logger.warning("Continuing without cache - performance may be degraded")
    
    async def _init_lexicon_db(self) -> None:
        """Initialize SQLite lexicon database connection"""
        try:
            logger.info("Connecting to lexicon database (SQLite)...")
            
            # Extract database path from URL
            db_path = self.config.lexicon_db_url.replace("sqlite:///", "")
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create connection
            conn = await aiosqlite.connect(
                db_path,
                timeout=self.config.connection_timeout
            )
            
            # Enable foreign keys and WAL mode for better performance
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL") 
            await conn.commit()
            
            self._lexicon_connections["main"] = conn
            
            # Test connection
            async with conn.execute("SELECT sqlite_version()") as cursor:
                result = await cursor.fetchone()
                logger.info(f"Lexicon database connected: SQLite {result[0]}")
            
            # Initialize lexicon tables
            await self._create_lexicon_tables()
            
            self._connection_status["lexicon"] = True
            
        except Exception as e:
            logger.error(f"Lexicon database connection failed: {str(e)}")
            raise
    
    async def _create_corpus_tables(self) -> None:
        """Create corpus database tables if they don't exist"""
        if not self._corpus_pool:
            return
            
        corpus_schema = """
        -- Quranic verses table
        CREATE TABLE IF NOT EXISTS quranic_verses (
            id SERIAL PRIMARY KEY,
            surah_number INTEGER NOT NULL,
            verse_number INTEGER NOT NULL,
            arabic_text TEXT NOT NULL,
            transliteration TEXT,
            translation_en TEXT,
            revelation_context TEXT,
            linguistic_features JSONB,
            root_analysis JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(surah_number, verse_number)
        );
        
        -- Hadith collection table
        CREATE TABLE IF NOT EXISTS hadith_collection (
            id SERIAL PRIMARY KEY,
            collection_name VARCHAR(100) NOT NULL, -- Bukhari, Muslim, etc.
            book_number INTEGER,
            hadith_number INTEGER,
            arabic_text TEXT NOT NULL,
            english_text TEXT,
            narrator_chain TEXT,
            authenticity_grade VARCHAR(50),
            subject_tags TEXT[],
            linguistic_analysis JSONB,
            root_occurrences JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(collection_name, book_number, hadith_number)
        );
        
        -- Classical Arabic literature table
        CREATE TABLE IF NOT EXISTS classical_texts (
            id SERIAL PRIMARY KEY,
            work_title VARCHAR(255) NOT NULL,
            author_name VARCHAR(255) NOT NULL,
            period VARCHAR(100), -- Jahili, Umayyad, Abbasid, etc.
            genre VARCHAR(100), -- Poetry, Prose, etc.
            text_excerpt TEXT NOT NULL,
            linguistic_features JSONB,
            cultural_context JSONB,
            root_analysis JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Root etymology table
        CREATE TABLE IF NOT EXISTS root_etymologies (
            id SERIAL PRIMARY KEY,
            root_form VARCHAR(20) NOT NULL UNIQUE,
            language_family VARCHAR(50) NOT NULL,
            proto_form VARCHAR(50),
            semantic_field VARCHAR(100),
            core_meaning TEXT NOT NULL,
            derived_meanings JSONB,
            cognates JSONB, -- Related forms in other languages
            historical_development JSONB,
            scholarly_consensus JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_quranic_verses_roots ON quranic_verses USING GIN (root_analysis);
        CREATE INDEX IF NOT EXISTS idx_hadith_roots ON hadith_collection USING GIN (root_occurrences);
        CREATE INDEX IF NOT EXISTS idx_classical_texts_roots ON classical_texts USING GIN (root_analysis);
        CREATE INDEX IF NOT EXISTS idx_root_etymologies_family ON root_etymologies (language_family);
        """
        
        try:
            async with self._corpus_pool.acquire() as conn:
                await conn.execute(corpus_schema)
                logger.info("Corpus database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create corpus tables: {str(e)}")
            raise
    
    async def _create_lexicon_tables(self) -> None:
        """Create lexicon database tables if they don't exist"""
        conn = self._lexicon_connections.get("main")
        if not conn:
            return
            
        lexicon_schema = """
        -- Dictionary entries table
        CREATE TABLE IF NOT EXISTS dictionary_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headword TEXT NOT NULL,
            language_code TEXT NOT NULL,
            part_of_speech TEXT,
            definition TEXT NOT NULL,
            etymology TEXT,
            root_form TEXT,
            semantic_field TEXT,
            usage_examples TEXT,
            frequency_rank INTEGER,
            source_dictionary TEXT,
            confidence_score REAL DEFAULT 1.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Morphological analysis cache
        CREATE TABLE IF NOT EXISTS morphological_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_form TEXT NOT NULL,
            language_code TEXT NOT NULL,
            analysis_result TEXT NOT NULL, -- JSON string
            algorithm_version TEXT,
            confidence_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(word_form, language_code, algorithm_version)
        );
        
        -- Semantic similarity cache
        CREATE TABLE IF NOT EXISTS similarity_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word1 TEXT NOT NULL,
            word2 TEXT NOT NULL,
            language_code TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            algorithm_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(word1, word2, language_code, algorithm_version)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_dictionary_headword ON dictionary_entries (headword);
        CREATE INDEX IF NOT EXISTS idx_dictionary_root ON dictionary_entries (root_form);
        CREATE INDEX IF NOT EXISTS idx_dictionary_lang ON dictionary_entries (language_code);
        CREATE INDEX IF NOT EXISTS idx_morph_cache_word ON morphological_cache (word_form, language_code);
        CREATE INDEX IF NOT EXISTS idx_similarity_words ON similarity_cache (word1, word2, language_code);
        """
        
        try:
            await conn.executescript(lexicon_schema)
            await conn.commit()
            logger.info("Lexicon database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create lexicon tables: {str(e)}")
            raise
    
    @asynccontextmanager
    async def get_corpus_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a corpus database connection from the pool"""
        if not self._corpus_pool:
            raise SirajProcessingError("Corpus database not initialized")
        
        async with self._corpus_pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager  
    async def get_lexicon_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a lexicon database connection"""
        conn = self._lexicon_connections.get("main")
        if not conn:
            raise SirajProcessingError("Lexicon database not initialized")
        
        yield conn
    
    def get_cache_client(self) -> Optional[redis.Redis]:
        """Get the Redis cache client"""
        return self._cache_client
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections"""
        status = {
            "corpus": False,
            "lexicon": False,
            "cache": False,
            "overall": False
        }
        
        details = {}
        
        # Check corpus database
        if self._corpus_pool:
            try:
                async with self._corpus_pool.acquire() as conn:
                    result = await conn.fetchrow("SELECT 1 as test")
                    if result and result["test"] == 1:
                        status["corpus"] = True
                        details["corpus"] = "Connected"
                    else:
                        details["corpus"] = "Query failed"
            except Exception as e:
                details["corpus"] = f"Error: {str(e)}"
        else:
            details["corpus"] = "Not initialized"
        
        # Check lexicon database
        conn = self._lexicon_connections.get("main")
        if conn:
            try:
                async with conn.execute("SELECT 1 as test") as cursor:
                    result = await cursor.fetchone()
                    if result and result[0] == 1:
                        status["lexicon"] = True
                        details["lexicon"] = "Connected"
                    else:
                        details["lexicon"] = "Query failed"
            except Exception as e:
                details["lexicon"] = f"Error: {str(e)}"
        else:
            details["lexicon"] = "Not initialized"
        
        # Check cache database
        if self._cache_client:
            try:
                await self._cache_client.ping()
                status["cache"] = True
                details["cache"] = "Connected"
            except Exception as e:
                details["cache"] = f"Error: {str(e)}"
        else:
            details["cache"] = "Not initialized"
        
        # Overall status
        status["overall"] = status["corpus"] and status["lexicon"]  # Cache is optional
        
        return {
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Clean up all database connections"""
        logger.info("Cleaning up database connections...")
        
        # Close corpus pool
        if self._corpus_pool:
            await self._corpus_pool.close()
            logger.info("Corpus database pool closed")
        
        # Close cache client
        if self._cache_client:
            await self._cache_client.close()
            logger.info("Cache database connection closed")
        
        # Close lexicon connections
        for name, conn in self._lexicon_connections.items():
            await conn.close()
            logger.info(f"Lexicon database connection '{name}' closed")
        
        self._initialized = False
        logger.info("Database cleanup complete")


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Dependency injection function for FastAPI"""
    if not connection_manager._initialized:
        await connection_manager.initialize()
    return connection_manager