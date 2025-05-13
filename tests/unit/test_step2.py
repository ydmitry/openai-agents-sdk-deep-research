"""
Unit tests for the deep_research.step2 module.
"""

import asyncio
import json
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import pytest

from deep_research.step1 import ResearchPlan, SubTask
from deep_research.step2 import (
    Document, 
    SearchClient, 
    AgentSearchClient, 
    Scraper, 
    ResearchCorpusBuilder,
    build_corpus
)

# -----------------------------
# Test Document class
# -----------------------------

def test_document_to_json():
    """Test Document.to_json() serializes Document correctly."""
    # Arrange
    doc = Document(source_task_id=1, url="https://example.com", title="Example", text="Some content")
    
    # Act
    json_str = doc.to_json()
    parsed = json.loads(json_str)
    
    # Assert
    assert parsed["source_task_id"] == 1
    assert parsed["url"] == "https://example.com"
    assert parsed["title"] == "Example"
    assert parsed["text"] == "Some content"

# -----------------------------
# Mock search client for testing
# -----------------------------

class MockSearchClient:
    """Implementation of SearchClient for testing."""
    
    def __init__(self, mock_urls=None):
        """
        Initialize with optional predefined URLs to return.
        
        Args:
            mock_urls: Dictionary mapping queries to lists of URLs.
        """
        self.mock_urls = mock_urls or {}
        self.search_calls = []
    
    async def search(self, query: str, max_results: int = 5) -> list[str]:
        """
        Mock search that returns predefined URLs or generates fake ones.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of mock URLs.
        """
        self.search_calls.append((query, max_results))
        
        if query in self.mock_urls:
            return self.mock_urls[query][:max_results]
        
        # Generate fake URLs if none provided
        return [f"https://example.com/result/{i}?q={query.replace(' ', '+')}" 
                for i in range(max_results)]

# -----------------------------
# Test AgentSearchClient
# -----------------------------

@pytest.mark.asyncio
async def test_agent_search_client():
    """Test the AgentSearchClient with mocked Runner."""
    # Arrange
    query = "test query"
    mock_result = Mock()
    mock_result.final_output = '["https://example.com/1", "https://example.com/2"]'
    
    # Act
    with patch("deep_research.step2.Runner.run", new=AsyncMock(return_value=mock_result)):
        client = AgentSearchClient(model="gpt-4o")
        urls = await client.search(query)
    
    # Assert
    assert urls == ["https://example.com/1", "https://example.com/2"]

# -----------------------------
# Test Scraper
# -----------------------------

@pytest.mark.asyncio
async def test_html_to_text():
    """Test the Scraper.html_to_text method."""
    # Arrange
    scraper = Scraper()
    html = """
    <html>
        <head><title>Test</title></head>
        <body>
            <header>Site header</header>
            <script>console.log("hello");</script>
            <style>body { color: red; }</style>
            <div>Hello <span>World</span></div>
            <footer>Footer content</footer>
        </body>
    </html>
    """
    
    # Act
    text = scraper.html_to_text(html)
    
    # Assert
    assert "Hello World" in text
    assert "console.log" not in text
    assert "Site header" not in text
    assert "Footer content" not in text

@pytest.mark.asyncio
async def test_fetch_html_success():
    """Test the Scraper.fetch_html method with successful response."""
    # Arrange
    scraper = Scraper()
    
    # Create a context manager mock for the session.get result
    mock_response = MagicMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.raise_for_status = Mock()
    mock_response.text = AsyncMock(return_value="<html>content</html>")
    
    mock_session = MagicMock()
    mock_session.get = Mock(return_value=mock_response)
    
    # Act
    html = await scraper.fetch_html(mock_session, "https://example.com")
    
    # Assert
    assert html == "<html>content</html>"
    mock_session.get.assert_called_once_with("https://example.com", timeout=scraper.timeout)

@pytest.mark.asyncio
async def test_fetch_html_failure():
    """Test the Scraper.fetch_html method with a failed request."""
    # Arrange
    scraper = Scraper()
    mock_session = MagicMock()
    mock_session.get = Mock(side_effect=Exception("Connection error"))
    
    # Act
    html = await scraper.fetch_html(mock_session, "https://example.com")
    
    # Assert
    assert html == ""  # Should return empty string on error

@pytest.mark.asyncio
async def test_scrape_urls():
    """Test the Scraper.scrape_urls method."""
    # Create a mock scraper class with controlled behavior
    class MockScraper(Scraper):
        async def fetch_html(self, session, url):
            # Return different HTML based on URL
            if url == "https://example.com/1":
                return "<html><body>First document content</body></html>"
            else:
                return "<html><body>Second document content</body></html>"
        
        def html_to_text(self, html):
            # Extract text based on the content
            if "First" in html:
                return "First document content with enough text to pass minimum length check " * 10
            else:
                return "Second document content with enough text to pass minimum length check " * 10
    
    # Use our mock scraper
    scraper = MockScraper()
    task_id = 1
    urls = ["https://example.com/1", "https://example.com/2"]
    
    # Mock ClientSession to avoid real HTTP requests
    mock_session = MagicMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    
    # Patch ClientSession constructor to return our mock
    with patch("deep_research.step2.aiohttp.ClientSession", return_value=mock_session):
        # Act
        docs = await scraper.scrape_urls(task_id, urls)
    
    # Assert
    assert len(docs) == 2
    assert docs[0].source_task_id == task_id
    assert docs[0].url == urls[0]
    assert "First document content" in docs[0].text
    assert docs[1].source_task_id == task_id
    assert docs[1].url == urls[1]
    assert "Second document content" in docs[1].text

# -----------------------------
# Test ResearchCorpusBuilder
# -----------------------------

@pytest.mark.asyncio
async def test_process_task():
    """Test the ResearchCorpusBuilder.process_task method."""
    # Arrange
    mock_search = MockSearchClient({
        "task 1": ["https://example.com/1", "https://example.com/2"]
    })
    
    mock_scraper = Mock(spec=Scraper)
    mock_docs = [
        Document(source_task_id=1, url="https://example.com/1", title="Doc 1", text="Content 1"),
        Document(source_task_id=1, url="https://example.com/2", title="Doc 2", text="Content 2")
    ]
    mock_scraper.scrape_urls = AsyncMock(return_value=mock_docs)
    
    task = SubTask(id=1, task="task 1", priority=1)
    builder = ResearchCorpusBuilder(search_client=mock_search, scraper=mock_scraper)
    
    # Act
    docs = await builder.process_task(task)
    
    # Assert
    assert len(mock_search.search_calls) == 1
    assert mock_search.search_calls[0][0] == "task 1"
    mock_scraper.scrape_urls.assert_called_once_with(1, ["https://example.com/1", "https://example.com/2"])
    assert docs == mock_docs

@pytest.mark.asyncio
async def test_build_corpus():
    """Test the ResearchCorpusBuilder.build_corpus method."""
    # Arrange
    plan = ResearchPlan(
        objective="Test objective",
        sub_tasks=[
            SubTask(id=1, task="task 1", priority=1),
            SubTask(id=2, task="task 2", priority=2),
        ]
    )
    
    # Mock documents to return for each task
    mock_docs_task1 = [Document(source_task_id=1, url="https://example.com/1", title="Doc 1", text="Content 1")]
    mock_docs_task2 = [Document(source_task_id=2, url="https://example.com/2", title="Doc 2", text="Content 2")]
    
    # Create a builder with a mock search client that returns predictable results
    mock_search = MockSearchClient()
    
    # Create a mock scraper that returns our predefined documents
    mock_scraper = MagicMock(spec=Scraper)
    mock_scraper.scrape_urls = AsyncMock()
    mock_scraper.scrape_urls.side_effect = [mock_docs_task1, mock_docs_task2]
    
    # Create builder with our mocks
    builder = ResearchCorpusBuilder(
        search_client=mock_search,
        scraper=mock_scraper,
        concurrency=2
    )
    
    # Act
    docs = await builder.build_corpus(plan)
    
    # Assert
    assert len(docs) == 2
    assert any(d.source_task_id == 1 for d in docs)
    assert any(d.source_task_id == 2 for d in docs)

# -----------------------------
# Test public API functions
# -----------------------------

def test_build_corpus_api():
    """Test the public build_corpus API function."""
    # Arrange
    plan = ResearchPlan(
        objective="Test objective",
        sub_tasks=[SubTask(id=1, task="task 1", priority=1)]
    )
    
    mock_docs = [Document(source_task_id=1, url="https://example.com", title="Doc", text="Content")]
    
    # Act
    with patch("deep_research.step2.async_build_corpus", new=AsyncMock(return_value=mock_docs)):
        docs = build_corpus(plan)
    
    # Assert
    assert docs == mock_docs 