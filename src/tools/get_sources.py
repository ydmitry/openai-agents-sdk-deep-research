#!/usr/bin/env python3
"""
Get Sources Tool Factory

Shared tool factory for creating get_sources functions that extract URLs from stored search results.
This allows multiple agents to use the same citation retrieval functionality.

Usage:
    from tools.get_sources import create_get_sources_tool
    from storage.search_results import SearchResultsStorage
    
    storage = SearchResultsStorage()
    get_sources_tool = create_get_sources_tool(storage)
"""

import logging
from agents import function_tool
from storage.search_results import SearchResultsStorage

logger = logging.getLogger(__name__)


def create_get_sources_tool(storage: SearchResultsStorage):
    """
    Create a get sources function tool that extracts URLs from stored search results.
    
    Args:
        storage: SearchResultsStorage instance to retrieve citations from
        
    Returns:
        function_tool: A configured function tool that can retrieve citations
    """
    @function_tool
    async def get_sources() -> str:
        """
        Get all unique sources (URLs) from previously stored search results.
        Use this when users ask for sources, citations, references, or URLs from the research.

        Returns:
            A formatted list of unique URLs found in the search results
        """
        try:
            logger.info("Getting sources from stored search results")
            
            citations = storage.get_citations()
            
            if not citations:
                return "No sources found in stored search results."
            
            # Format citations nicely
            formatted_citations = "\n".join([f"• {url}" for url in citations])
            result = f"Sources from search results ({len(citations)} unique URLs):\n\n{formatted_citations}"
            
            logger.info(f"Retrieved {len(citations)} unique sources")
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving sources: {str(e)}"
            logger.error(error_msg)
            return f"Failed to get sources: {error_msg}"

    return get_sources 