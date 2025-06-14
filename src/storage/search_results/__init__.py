"""
Search Results Storage System

Provides a unified interface for storing and retrieving search results
with different backend storage mechanisms.
"""

from typing import Protocol, List, Dict, Any
import re
from abc import ABC, abstractmethod


class SearchResultsStorage(Protocol):
    """Common interface for search results storage backends."""
    
    def collect_search_results(self, results: str) -> None:
        """Store search results."""
        ...
    
    def get_citations(self) -> List[str]:
        """Extract and return unique URLs from stored search results."""
        ...


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text and return unique list.
    
    Parameters
    ----------
    text : str
        Text containing URLs
        
    Returns
    -------
    List[str]
        List of unique URLs found in the text
    """
    # Regex pattern to match URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?)]'
    
    urls = re.findall(url_pattern, text)
    
    # Return unique URLs while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


# Import storage factories
from .memory import create_memory_storage
from .textfile import create_textfile_storage
from .postgres import create_postgres_storage

__all__ = [
    'SearchResultsStorage',
    'extract_urls_from_text',
    'create_memory_storage',
    'create_textfile_storage', 
    'create_postgres_storage'
] 