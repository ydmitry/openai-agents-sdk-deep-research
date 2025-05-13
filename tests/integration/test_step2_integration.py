"""
Integration tests for the deep_research.step2 module.
"""

import asyncio
import os
import tempfile
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from deep_research.step1 import ResearchPlan, SubTask
from deep_research.step2 import (
    Document, 
    build_corpus,
    async_build_corpus,
    Scraper
)
from tests.unit.test_step2 import MockSearchClient

# Sample research plan for testing
SAMPLE_PLAN = ResearchPlan(
    objective="Test research objective",
    sub_tasks=[
        SubTask(id=1, task="First research task", priority=1),
        SubTask(id=2, task="Second research task", priority=2),
    ]
)

# -----------------------------
# Mock Scraper for testing
# -----------------------------

class MockScraper(Scraper):
    """Mock scraper that returns predefined documents without needing to actually scrape URLs."""
    
    def __init__(self):
        super().__init__()
        # Map task_id to list of documents
        self.docs_by_task = {}
    
    def add_mock_documents(self, task_id: int, urls: list[str]):
        """
        Pre-populate mock documents for a specific task.
        
        Args:
            task_id: ID of the task.
            urls: List of URLs to generate mock documents for.
        """
        docs = []
        for url in urls:
            query = url.split("?q=")[1].replace("+", " ") if "?q=" in url else "unknown"
            docs.append(Document(
                source_task_id=task_id,
                url=url,
                title=f"Mock document for {query}",
                text=f"This is mock content about {query}. " * 20  # Make it long enough to pass filters
            ))
        self.docs_by_task[task_id] = docs
    
    async def scrape_urls(self, task_id: int, urls: list[str]) -> list[Document]:
        """
        Return pre-populated mock documents without actually scraping.
        
        Args:
            task_id: ID of the task.
            urls: List of URLs (ignored, we use pre-populated docs).
            
        Returns:
            List of pre-populated Document objects or empty list if none found.
        """
        return self.docs_by_task.get(task_id, [])

# -----------------------------
# Integration tests
# -----------------------------

@pytest.mark.asyncio
async def test_async_build_corpus_with_mocks():
    """Test the async_build_corpus function with mocked components."""
    # Arrange
    mock_urls = {
        "First research task": [
            "https://example.com/1?q=First+research+task",
            "https://example.com/2?q=First+research+task"
        ],
        "Second research task": [
            "https://example.com/3?q=Second+research+task",
            "https://example.com/4?q=Second+research+task"
        ]
    }
    
    # Create a mock search client with fixed URLs
    mock_search = MockSearchClient(mock_urls)
    
    # Create a mock scraper with pre-populated documents
    mock_scraper = MockScraper()
    mock_scraper.add_mock_documents(1, mock_urls["First research task"])
    mock_scraper.add_mock_documents(2, mock_urls["Second research task"])
    
    # Act
    docs = await async_build_corpus(
        plan=SAMPLE_PLAN,
        search_client=mock_search,
        scraper=mock_scraper,
        concurrency=2
    )
    
    # Assert
    assert len(docs) == 4  # 2 docs per subtask * 2 subtasks
    
    # Check that docs have correct task IDs
    assert sum(1 for d in docs if d.source_task_id == 1) == 2
    assert sum(1 for d in docs if d.source_task_id == 2) == 2
    
    # Check that content was successfully extracted
    for doc in docs:
        assert f"This is mock content about" in doc.text

def test_build_corpus_to_jsonl():
    """
    Test the end-to-end process of building a corpus and writing to a JSONL file.
    Uses mocked components to avoid actual web requests.
    """
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "corpus.jsonl"
        
        # Create a plan.json file for CLI testing
        plan_path = Path(tmpdir) / "plan.json"
        with plan_path.open("w") as f:
            f.write(SAMPLE_PLAN.to_json())
        
        # Prepare mock components
        mock_urls = {
            "First research task": [f"https://example.com/{i}?q=First+research+task" for i in range(3)],
            "Second research task": [f"https://example.com/{i}?q=Second+research+task" for i in range(3)]
        }
        
        mock_search = MockSearchClient(mock_urls)
        mock_scraper = MockScraper()
        mock_scraper.add_mock_documents(1, mock_urls["First research task"])
        mock_scraper.add_mock_documents(2, mock_urls["Second research task"])
        
        # Act - build corpus with mocked components
        docs = build_corpus(
            SAMPLE_PLAN, 
            search_client=mock_search,
            scraper=mock_scraper
        )
        
        # Write docs to JSONL file
        with output_file.open("w", encoding="utf-8") as out_f:
            for d in docs:
                out_f.write(d.to_json() + "\n")
        
        # Assert
        # Check that the JSONL file was created and contains documents
        assert output_file.exists()
        
        # Read the file and check content
        docs_from_file = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                docs_from_file.append(json.loads(line))
        
        assert len(docs_from_file) > 0
        
        # Check that each doc has the expected fields
        for doc in docs_from_file:
            assert "source_task_id" in doc
            assert "url" in doc
            assert "title" in doc
            assert "text" in doc 