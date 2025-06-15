"""
PostgreSQL-based search results storage with embeddings.
"""

import logging
import os
import uuid
from typing import List, Optional, Dict, Any
from . import extract_urls_from_text

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class PostgresStorage:
    """
    PostgreSQL storage for search results with embeddings.
    
    Requires the search_results table to be created with the migration script:
    create_search_results_table.py
    
    This will create the necessary table structure and indexes for optimal performance.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL storage. Install with: pip install psycopg2-binary")
        
        self.session_id = session_id or str(uuid.uuid4())
        
        # Database configuration (same as original implementation)
        self.db_config = {
            'host': '127.0.0.1',
            'port': 5433,
            'database': 'my-deep-research',
            'user': 'postgres',
            'password': 'secret'
        }
        
        # Initialize OpenAI client for embeddings
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized for embedding generation")
                else:
                    logger.warning("OPENAI_API_KEY not found - embeddings will be skipped")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e} - embeddings will be skipped")
        else:
            logger.warning("OpenAI library not available - embeddings will be skipped")
        
        logger.info(f"Created PostgreSQL storage with session_id: {self.session_id}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API."""
        if not self.openai_client:
            return None
            
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for text (length: {len(text)}): {e}")
            return None
    
    def _connect_to_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def collect_search_results(self, results: str) -> None:
        """Store search results in PostgreSQL with embeddings."""
        try:
            # Connect to database
            conn = self._connect_to_db()
            if not conn:
                logger.error("Failed to connect to database - search results not stored")
                return
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get embedding for the search results
                    embedding = self._get_embedding(results)
                    
                    # Insert search results (with or without embedding)
                    cur.execute("""
                        INSERT INTO search_results (session_id, text, embedding)
                        VALUES (%s, %s, %s)
                        RETURNING id, created_at;
                    """, (self.session_id, results, embedding))
                    
                    result = cur.fetchone()
                    conn.commit()
                    
                    embedding_status = "with embedding" if embedding else "without embedding"
                    logger.info(
                        f"Stored search results to database (ID: {result['id']}, "
                        f"Session: {self.session_id[:8]}..., {embedding_status})"
                    )
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error storing search results to database: {e}")
    
    def get_citations(self) -> List[str]:
        """Extract and return unique URLs from stored search results by session ID."""
        try:
            # Connect to database
            conn = self._connect_to_db()
            if not conn:
                logger.error("Failed to connect to database - returning empty citations")
                return []
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Query all search results for this session
                    cur.execute("""
                        SELECT text FROM search_results 
                        WHERE session_id = %s 
                        ORDER BY created_at;
                    """, (self.session_id,))
                    
                    rows = cur.fetchall()
                    
                    # Combine all text from results
                    all_text = ""
                    for row in rows:
                        all_text += row['text'] + "\n"
                    
                    # Extract citations
                    citations = extract_urls_from_text(all_text)
                    logger.info(
                        f"Extracted {len(citations)} unique citations from {len(rows)} database records "
                        f"(Session: {self.session_id[:8]}...)"
                    )
                    return citations
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting citations from database: {e}")
            return []
    
    def has_similarity_search(self) -> bool:
        """PostgreSQL storage supports similarity search with embeddings."""
        return True
    
    def similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity on embeddings.
        Searches within the current session only.
        
        Uses the ivfflat index created by the migration script for optimal performance.
        """
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                logger.warning("Could not generate embedding for query - returning empty results")
                return []
            
            # Connect to database
            conn = self._connect_to_db()
            if not conn:
                logger.error("Failed to connect to database - returning empty results")
                return []
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Simple similarity search using cosine distance
                    # Lower distance = more similar
                    cur.execute("""
                        SELECT 
                            text, 
                            created_at,
                            embedding <=> %s AS distance,
                            id
                        FROM search_results 
                        WHERE session_id = %s 
                            AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s 
                        LIMIT %s;
                    """, (query_embedding, self.session_id, query_embedding, limit))
                    
                    rows = cur.fetchall()
                    
                    # Convert to standard format
                    results = []
                    for row in rows:
                        # Convert distance to similarity score (0-1, higher is more similar)
                        distance = float(row['distance']) if row['distance'] is not None else 1.0
                        similarity_score = max(0.0, min(1.0, 1.0 - distance))
                        
                        result = {
                            "text": row['text'],
                            "similarity_score": similarity_score,
                            "timestamp": row['created_at'].isoformat() if row['created_at'] else "",
                            "metadata": {
                                "database_id": row['id'],
                                "session_id": self.session_id,
                                "cosine_distance": distance
                            }
                        }
                        results.append(result)
                    
                    logger.info(
                        f"Found {len(results)} similar results for query (Session: {self.session_id[:8]}...)"
                    )
                    return results
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []


def create_postgres_storage(session_id: Optional[str] = None) -> tuple[PostgresStorage, str]:
    """
    Create a PostgreSQL-based search results storage with embeddings.
    
    Parameters
    ----------
    session_id : str, optional
        Session ID for grouping results. If not provided, a new UUID will be generated.
        
    Returns
    -------
    tuple
        (PostgresStorage, session_id) - Storage object and the session ID used
        
    Raises
    ------
    ImportError
        If required dependencies (psycopg2) are not available
    """
    storage = PostgresStorage(session_id)
    return storage, storage.session_id 