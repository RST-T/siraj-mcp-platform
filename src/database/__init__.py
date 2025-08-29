"""
SIRAJ v6.1 Database Module
Provides database connection management and corpus data access
"""

from .connection_manager import ConnectionManager, connection_manager, get_connection_manager
from .corpus_access import CorpusDataAccess

__all__ = [
    'ConnectionManager',
    'connection_manager', 
    'get_connection_manager',
    'CorpusDataAccess'
]