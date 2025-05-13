"""
Step 2 of an AutoGPT‑style deep‑research pipeline
=================================================
Web Search & Scraping Layer
--------------------------
Takes the `ResearchPlan` produced by *Step 1* and, for each sub‑task,
performs a breadth‑limited web search plus lightweight scraping of the
returned URLs. Yields a structured corpus of *Documents* ready for the
summarisation layer (Step 3).

Highlights
~~~~~~~~~~
* **Asynchronous parallelism** – tasks and URL fetches execute concurrently.
* **Pluggable search client** – default uses the Agents SDK `WebSearchTool`,
  but a fake client can be injected for unit tests.
* **Basic HTML → Text extraction** using `BeautifulSoup` (no JS).  Swap in a
  headless‑browser scraper if needed.
* **Observable trace** – every agent run is streamed so it can be debugged
  alongside Step 1's planner trace.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Sequence, Optional, Protocol, Any

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

from agents import Agent, Runner, WebSearchTool, RunConfig, ModelSettings
from deep_research.step1 import ResearchPlan, SubTask  # Re‑use dataclasses
from deep_research.utils import load_dotenv_files, get_model_settings

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Document:
    """Represents a document scraped from the web with metadata."""
    source_task_id: int
    url: str
    title: str
    text: str

    def to_json(self) -> str:
        """Serialize document to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# Protocol for search clients
# -----------------------------

class SearchClient(Protocol):
    """Protocol for search clients to allow dependency injection and easier testing."""

    async def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Execute a search and return a list of URLs.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of URLs.
        """
        ...

# -----------------------------
# Default search client implementation
# -----------------------------

class AgentSearchClient:
    """Implementation of SearchClient using the OpenAI Agents WebSearchTool."""

    def __init__(self, model: str = "gpt-4.1"):
        """
        Initialize the agent search client.

        Args:
            model: OpenAI model to use.
        """
        self.model = model
        self.agent = Agent(
            name="Searcher",
            instructions="""
            You are *SearchExecutor*, an agent that performs a single web search query
            and returns up to **5 high‑quality result URLs** as a JSON array. Prioritise
            scholarly, governmental, or reputable industry sources, avoid ads. Respond
            with **ONLY** the JSON list, e.g. ["https://example.com", ...].
            """,
            tools=[WebSearchTool()],
        )

    async def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Execute a search using the WebSearchTool and return a list of URLs.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of URLs.
        """
        # Create run configuration with conditional reasoning effort
        run_config = RunConfig(
            model=self.model,
            model_settings=get_model_settings(
                model_name=self.model,
                temperature=0.0
            ),
            tracing_disabled=True,
            workflow_name="Web Search"
        )

        print(f"Searching for: {query}")

        result = await Runner.run(
            self.agent,
            query,
            run_config=run_config,
            max_turns=3,  # Limit to 3 turns to avoid infinite loops
        )

        print(f"Search response: {result.final_output}")

        # Parse the JSON response
        try:
            # Try to parse the response directly
            urls = json.loads(result.final_output)

            # Ensure we have a list of strings
            if not isinstance(urls, list):
                urls = []

            # Filter out non-string elements and take up to max_results
            urls = [url for url in urls if isinstance(url, str)][:max_results]
            print(f"Found {len(urls)} URLs for query: {query}")

            return urls
        except (json.JSONDecodeError, TypeError) as e:
            # In case parsing fails, return an empty list
            print(f"Error parsing search results: {e}")
            return []

# -----------------------------
# Scraper implementation
# -----------------------------

class Scraper:
    """HTML scraper that extracts plaintext content from web pages."""

    def __init__(self, timeout: int = 30):
        """
        Initialize the scraper.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        Fetch HTML content from a URL.

        Args:
            session: aiohttp ClientSession.
            url: URL to fetch.

        Returns:
            HTML content as string or empty string if fetch fails.
        """
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return await resp.text()
        except Exception:
            return ""

    def html_to_text(self, html: str) -> str:
        """
        Convert HTML to plaintext with enhanced structure preservation.

        Args:
            html: HTML content.

        Returns:
            Extracted plaintext with preserved document structure.
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title with higher priority
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Extract meta description for context
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Remove script/style and navigation elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        
        # Process headings and paragraphs to maintain structure
        headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3']) if len(h.get_text().strip()) > 10]
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 40]
        
        # Combine with structure markers
        structured_text = title_text
        if description:
            structured_text += f"\n\n{description}"
        
        if headings:
            structured_text += f"\n\n" + "\n\n".join(headings)
        
        if paragraphs:
            structured_text += f"\n\n" + "\n\n".join(paragraphs)
        
        # If we couldn't extract anything meaningful, fall back to basic extraction
        if len(structured_text.strip()) < 200:
            text = soup.get_text(" ", strip=True)
            return " ".join(text.split())
            
        return structured_text

    async def scrape_urls(self, task_id: int, urls: Sequence[str]) -> List[Document]:
        """
        Scrape multiple URLs and convert to Document objects.

        Args:
            task_id: ID of the source subtask.
            urls: List of URLs to scrape.

        Returns:
            List of Document objects.
        """
        docs: List[Document] = []
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for url in urls:
                html = await self.fetch_html(session, url)
                if not html:
                    continue
                text = self.html_to_text(html)[:8000]  # truncate to keep size reasonable
                if len(text) < 200:  # skip thin pages
                    continue
                title = url.split("/")[-1][:120]
                docs.append(Document(source_task_id=task_id, url=url, title=title, text=text))
        return docs

# -----------------------------
# Core pipeline
# -----------------------------

class ResearchCorpusBuilder:
    """
    Orchestrates the process of building a research corpus by searching and scraping.
    """

    def __init__(
        self,
        search_client: Optional[SearchClient] = None,
        scraper: Optional[Scraper] = None,
        model: str = "gpt-4.1",
        concurrency: int = 4
    ):
        """
        Initialize the corpus builder.

        Args:
            search_client: Client for web searches. If None, use AgentSearchClient.
            scraper: Web scraper. If None, use default Scraper.
            model: OpenAI model to use if creating a default search client.
            concurrency: Maximum number of concurrent tasks.
        """
        self.search_client = search_client or AgentSearchClient(model=model)
        self.scraper = scraper or Scraper()
        self.concurrency = concurrency

    async def process_task(self, task: SubTask) -> List[Document]:
        """
        Process a single research subtask: search and scrape.

        Args:
            task: SubTask to process.

        Returns:
            List of Document objects.
        """
        urls = await self.search_client.search(task.task)
        return await self.scraper.scrape_urls(task.id, urls)

    async def build_corpus(self, plan: ResearchPlan) -> List[Document]:
        """
        Execute searches & scraping for every sub‑task in plan concurrently.

        Args:
            plan: Research plan with subtasks.

        Returns:
            List of Document objects forming the research corpus.
        """
        sem = asyncio.Semaphore(self.concurrency)

        async def _worker(st: SubTask):
            async with sem:
                return await self.process_task(st)

        coros = [_worker(st) for st in plan.sub_tasks]
        documents_nested = await tqdm_asyncio.gather(*coros)
        # Flatten the list of lists
        docs: List[Document] = [doc for lst in documents_nested for doc in lst]
        return docs

# -----------------------------
# Public API
# -----------------------------

async def async_build_corpus(
    plan: ResearchPlan,
    search_client: Optional[SearchClient] = None,
    scraper: Optional[Scraper] = None,
    model: str = "gpt-4.1",
    concurrency: int = 4
) -> List[Document]:
    """
    Async function to build a research corpus from a plan.

    Args:
        plan: Research plan with subtasks.
        search_client: Client for web searches. If None, use AgentSearchClient.
        scraper: Web scraper. If None, use default Scraper.
        model: OpenAI model to use if creating a default search client.
        concurrency: Maximum number of concurrent tasks.

    Returns:
        List of Document objects forming the research corpus.
    """
    builder = ResearchCorpusBuilder(
        search_client=search_client,
        scraper=scraper,
        model=model,
        concurrency=concurrency
    )
    return await builder.build_corpus(plan)

def build_corpus(
    plan: ResearchPlan,
    search_client: Optional[SearchClient] = None,
    scraper: Optional[Scraper] = None,
    model: str = "gpt-4.1",
    concurrency: int = 4
) -> List[Document]:
    """
    Blocking function to build a research corpus from a plan.

    Args:
        plan: Research plan with subtasks.
        search_client: Client for web searches. If None, use AgentSearchClient.
        scraper: Web scraper. If None, use default Scraper.
        model: OpenAI model to use if creating a default search client.
        concurrency: Maximum number of concurrent tasks.

    Returns:
        List of Document objects forming the research corpus.
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We are already in an event loop, so just create and return the coroutine
            return async_build_corpus(
                plan=plan,
                search_client=search_client,
                scraper=scraper,
                model=model,
                concurrency=concurrency
            )
    except RuntimeError:
        # No event loop running, so use asyncio.run to create one
        return asyncio.run(async_build_corpus(
            plan=plan,
            search_client=search_client,
            scraper=scraper,
            model=model,
            concurrency=concurrency
        ))

# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    # Load environment variables from .env files
    loaded_files = load_dotenv_files()

    parser = argparse.ArgumentParser(description="Step 2 – run web searches & scrape pages for each planner sub‑task.")
    parser.add_argument("plan_json", help="Path to plan JSON produced by Step 1")
    parser.add_argument("--out", default="corpus.jsonl", help="Output JSON‑Lines file (default: corpus.jsonl)")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel tasks (default: 4)")
    args = parser.parse_args()

    plan_path = pathlib.Path(args.plan_json)
    if not plan_path.exists():
        sys.exit(f"Plan file not found: {plan_path}")

    with plan_path.open() as f:
        plan_dict = json.load(f)

    plan = ResearchPlan(
        objective=plan_dict["objective"],
        sub_tasks=[SubTask(**st) for st in plan_dict["sub_tasks"]],
    )

    docs = build_corpus(
        plan,
        model=args.model,
        concurrency=args.concurrency
    )

    with open(args.out, "w", encoding="utf-8") as out_f:
        for d in docs:
            out_f.write(d.to_json() + "\n")

    print(f"Wrote {len(docs)} docs → {args.out}")
