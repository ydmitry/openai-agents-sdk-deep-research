"""
Text file-based search results storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List
from . import extract_urls_from_text

logger = logging.getLogger(__name__)


class TextFileStorage:
    """Text file storage for search results."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
    
    def collect_search_results(self, results: str) -> None:
        """Store search results in text file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure parent directory exists
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"SEARCH RESULTS - {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{results}\n")
            f.write(f"{'='*80}\n\n")

        logger.info(f"Collected search results to {self.output_file}")
    
    def get_citations(self) -> List[str]:
        """Extract and return unique URLs from stored search results in the file."""
        try:
            if not Path(self.output_file).exists():
                logger.info(f"File {self.output_file} does not exist, returning empty citations")
                return []
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            citations = extract_urls_from_text(content)
            logger.info(f"Extracted {len(citations)} unique citations from {self.output_file}")
            return citations
            
        except Exception as e:
            logger.error(f"Error reading citations from {self.output_file}: {e}")
            return []


def create_textfile_storage(output_file: str) -> TextFileStorage:
    """
    Create a text file-based search results storage.
    
    Parameters
    ----------
    output_file : str
        Path to the output text file
        
    Returns
    -------
    TextFileStorage
        Storage object that implements SearchResultsStorage protocol
    """
    return TextFileStorage(output_file) 