"""
Memory-based search results storage.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from . import extract_urls_from_text

logger = logging.getLogger(__name__)


class MemoryStorage:
    """In-memory storage for search results."""
    
    def __init__(self):
        self.storage: List[Dict] = []
    
    def collect_search_results(self, results: str) -> None:
        """Store search results in memory."""
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "search_results": results
        }
        self.storage.append(record)
        logger.info(f"Collected search results in memory (total: {len(self.storage)} records)")
    
    def get_citations(self) -> List[str]:
        """Extract and return unique URLs from all stored search results."""
        all_text = ""
        for record in self.storage:
            all_text += record["search_results"] + "\n"
        
        citations = extract_urls_from_text(all_text)
        logger.info(f"Extracted {len(citations)} unique citations from memory storage")
        return citations
    
    def get_all_results(self) -> List[Dict]:
        """Retrieve all stored results (for backward compatibility)."""
        return self.storage.copy()
    
    def has_similarity_search(self) -> bool:
        """Memory storage does not support similarity search."""
        return False
    
    def similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Memory storage does not support similarity search.
        Returns empty list.
        """
        logger.warning("Similarity search not supported for memory storage")
        return []


def create_memory_storage() -> MemoryStorage:
    """
    Create a memory-based search results storage.
    
    Returns
    -------
    MemoryStorage
        Storage object that implements SearchResultsStorage protocol
    """
    return MemoryStorage() 