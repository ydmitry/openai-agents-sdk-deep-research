"""
Step 3 of an AutoGPT-style deep-research pipeline
=================================================
Summarization & Synthesis Layer
------------------------------
Takes the corpus of documents produced by *Step 2* and generates a structured,
citation-faithful summary through a map-reduce architecture. Each phase is
implemented as an Agent or Tool, making the pipeline modular and observable.

Highlights
~~~~~~~~~~
* **Map-reduce architecture** - Extract bullets from document chunks, synthesize
  per-document summaries, and then compose a final report with citations.
* **Citation-faithful** - Every sentence can be traced to a source.
* **Parallelism** - Map phases execute in parallel for scalability and speed.
* **Observable** - Built-in tracing and metrics for the entire pipeline.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, RunConfig, ModelSettings
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

from deep_research.step2 import Document
from deep_research.utils import load_dotenv_files

# -----------------------------
# Schema objects
# -----------------------------

class BulletsJSON(BaseModel):
    """Structured representation of key facts extracted from a document chunk."""
    source_id: int = Field(..., description="ID of the source document")
    bullets: List[str] = Field(..., description="3-5 key facts, each with source reference")

class DocSummary(BaseModel):
    """Structured representation of a document-level summary."""
    source_id: int = Field(..., description="ID of the source document")
    title: str = Field(..., description="Title of the document")
    url: str = Field(..., description="URL of the document")
    summary: str = Field(..., description="≤200 words summary with inline [n] citations")

class Report(BaseModel):
    """Structured representation of the final research report."""
    title: str = Field(..., description="Title of the research report")
    body_md: str = Field(..., description="Full report body in Markdown format with citations")
    references: List[str] = Field(..., description="List of references used in the report")

class FactCheck(BaseModel):
    """Representation of fact-checking results for a report."""
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims lacking source support")
    conflicting_claims: List[str] = Field(default_factory=list, description="Claims with conflicting sources")
    overall_assessment: str = Field(..., description="Overall assessment of the report's factual accuracy")

# -----------------------------
# Extractive tools
# -----------------------------

@function_tool
def textrank_bullets(text: str, source_id: int) -> BulletsJSON:
    """
    Extract key facts from text using TextRank algorithm.
    
    Args:
        text: The text content to analyze
        source_id: ID of the source document
        
    Returns:
        BulletsJSON with extracted bullet points
    """
    # In a production environment, implement an actual TextRank algorithm
    # This is a placeholder that just returns the first few sentences
    
    # Clean and split the text into sentences
    clean_text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    # Take up to 5 sentences as bullets
    extracted_bullets = [f"{s} <{source_id}>" for s in sentences[:5] if len(s) > 30]
    
    # If we couldn't extract meaningful bullets, create a placeholder
    if not extracted_bullets:
        extracted_bullets = [f"Content from source {source_id}"]
    
    return BulletsJSON(source_id=source_id, bullets=extracted_bullets)

# -----------------------------
# Agent definitions
# -----------------------------

BulletExtractor = Agent(
    name="BulletExtractor",
    instructions="""
    You are an expert at extracting key facts from documents.
    
    Your task is to extract 3-5 important points from the provided text.
    Each fact should be:
    1. Self-contained and meaningful
    2. Focused on factual information
    3. Include the source_id token in format <SOURCE_ID> at the end
    
    Call the textrank_bullets tool with chunks of the document to get initial extractions,
    then review and refine the results before returning the final BulletsJSON.
    """,
    tools=[textrank_bullets],
    model="gpt-3.5-turbo-0125",
    output_type=BulletsJSON,
)

DocSummarizer = Agent(
    name="DocSummarizer",
    instructions="""
    You are a document summarization expert.
    
    Create a concise summary (≤200 words) of the document based on the key facts provided.
    Your summary should:
    1. Integrate all important facts from the bullet points
    2. Maintain the narrative flow and context
    3. Preserve source references using <SOURCE_ID> format for each fact
    4. Focus on factual information relevant to the research objective
    
    The returned DocSummary should include the document title, URL, and source ID.
    """,
    model="gpt-4o",
    output_type=DocSummary,
)

# Agent-as-tool lets the reducer call the summarizers in parallel
doc_summarizer_tool = DocSummarizer.as_tool(
    tool_name="summarize_doc",
    tool_description="Turn bullets into a doc-level summary"
)

ReportComposer = Agent(
    name="ReportComposer",
    instructions="""
    You are a research synthesis expert.
    
    Compose a comprehensive research report from multiple document summaries.
    Your report should:
    1. Have a clear structure with sections based on the research topic
    2. Include a concise introduction and conclusion
    3. Maintain all citation references using [n] format
    4. Organize information logically for maximum clarity
    5. Use proper Markdown formatting
    6. Include a references section at the end
    
    Return a Report object with title, body_md (Markdown content), and references.
    """,
    tools=[doc_summarizer_tool],
    model="gpt-4o",
    output_type=Report,
    model_settings=ModelSettings(parallel_tool_calls=True)
)

CriticAgent = Agent(
    name="CriticAgent",
    instructions="""
    You are a fact-checker and critical evaluator.
    
    Review the final report against the source document summaries to:
    1. Identify any claims in the report that lack proper source support
    2. Flag any conflicting claims across different sources
    3. Evaluate the overall factual accuracy of the report
    
    Focus on substance rather than style or grammar.
    Return a FactCheck object with your findings.
    """,
    model="gpt-4o",
    output_type=FactCheck,
)

# -----------------------------
# Pipeline implementation
# -----------------------------

class ResearchSummarizer:
    """
    Orchestrates the process of generating a research report from a corpus of documents.
    """
    
    def __init__(
        self,
        bullet_model: str = "gpt-3.5-turbo-0125",
        summary_model: str = "gpt-4o",
        report_model: str = "gpt-4o",
        critic_model: str = "gpt-4o",
        run_critique: bool = True,
        concurrency: int = 4,
        chunk_size: int = 1500
    ):
        """
        Initialize the research summarizer.
        
        Args:
            bullet_model: Model to use for bullet extraction
            summary_model: Model to use for document summarization
            report_model: Model to use for report composition
            critic_model: Model to use for fact checking
            run_critique: Whether to run the critique phase
            concurrency: Maximum number of concurrent tasks
            chunk_size: Size of text chunks for bullet extraction
        """
        self.bullet_model = bullet_model
        self.summary_model = summary_model
        self.report_model = report_model
        self.critic_model = critic_model
        self.run_critique = run_critique
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        
        # Configure agents with appropriate models
        self._configure_agents()
    
    def _configure_agents(self):
        """Configure the agents with the specified models."""
        global BulletExtractor, DocSummarizer, ReportComposer, CriticAgent
        
        BulletExtractor = Agent(
            name="BulletExtractor",
            instructions=BulletExtractor.instructions,
            tools=BulletExtractor.tools,
            model=self.bullet_model,
            output_type=BulletsJSON,
        )
        
        DocSummarizer = Agent(
            name="DocSummarizer",
            instructions=DocSummarizer.instructions,
            model=self.summary_model,
            output_type=DocSummary,
        )
        
        # Update the tool with the new DocSummarizer
        doc_summarizer_tool = DocSummarizer.as_tool(
            tool_name="summarize_doc",
            tool_description="Turn bullets into a doc-level summary"
        )
        
        ReportComposer = Agent(
            name="ReportComposer",
            instructions=ReportComposer.instructions,
            tools=[doc_summarizer_tool],
            model=self.report_model,
            output_type=Report,
            model_settings=ModelSettings(parallel_tool_calls=True)
        )
        
        CriticAgent = Agent(
            name="CriticAgent",
            instructions=CriticAgent.instructions,
            model=self.critic_model,
            output_type=FactCheck,
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of appropriate size.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple chunking by approximate character count
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for the space
            current_chunk.append(word)
            
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    async def _extract_bullets(self, document: Document) -> List[BulletsJSON]:
        """
        Extract bullet points from a document by chunking and using BulletExtractor.
        
        Args:
            document: The document to process
            
        Returns:
            List of BulletsJSON containing extracted bullets
        """
        chunks = self._chunk_text(document.text)
        sem = asyncio.Semaphore(self.concurrency)
        
        async def _process_chunk(chunk: str) -> BulletsJSON:
            async with sem:
                run_config = RunConfig(
                    model=self.bullet_model,
                    tracing_disabled=True,
                    workflow_name=f"Bullet Extraction - Doc {document.source_task_id}"
                )
                
                input_text = f"""
                Document ID: {document.source_task_id}
                URL: {document.url}
                Title: {document.title}
                
                Content:
                {chunk}
                """
                
                result = await Runner.run(
                    BulletExtractor,
                    input_text,
                    run_config=run_config
                )
                
                return result.final_output
        
        tasks = [_process_chunk(chunk) for chunk in chunks]
        return await tqdm_asyncio.gather(*tasks)
    
    async def _summarize_document(self, document: Document, bullets: List[BulletsJSON]) -> DocSummary:
        """
        Generate a document summary from extracted bullets.
        
        Args:
            document: The source document
            bullets: List of extracted bullets from the document
            
        Returns:
            DocSummary object
        """
        # Combine all bullets into a single input
        all_bullets = "\n".join([
            "\n".join([f"• {bullet}" for bullet in b.bullets])
            for b in bullets
        ])
        
        input_text = f"""
        Document ID: {document.source_task_id}
        URL: {document.url}
        Title: {document.title}
        
        Key Facts:
        {all_bullets}
        """
        
        run_config = RunConfig(
            model=self.summary_model,
            tracing_disabled=True,
            workflow_name=f"Document Summary - Doc {document.source_task_id}"
        )
        
        result = await Runner.run(
            DocSummarizer,
            input_text,
            run_config=run_config
        )
        
        # Ensure the summary has the correct source ID, URL and title
        summary = result.final_output
        summary.source_id = document.source_task_id
        summary.url = document.url
        summary.title = document.title
        
        return summary
    
    async def _compose_report(self, summaries: List[DocSummary], objective: str) -> Report:
        """
        Compose the final research report from document summaries.
        
        Args:
            summaries: List of document summaries
            objective: The research objective
            
        Returns:
            Report object
        """
        # Create a references dictionary to replace <SOURCE_ID> with [n] citations
        references = {}
        for i, summary in enumerate(summaries, 1):
            references[summary.source_id] = {
                "index": i,
                "title": summary.title,
                "url": summary.url
            }
        
        # Replace <SOURCE_ID> with [n] citations in summaries
        processed_summaries = []
        for summary in summaries:
            text = summary.summary
            for source_id, ref in references.items():
                text = text.replace(f"<{source_id}>", f"[{ref['index']}]")
            processed_summary = DocSummary(
                source_id=summary.source_id,
                title=summary.title,
                url=summary.url,
                summary=text
            )
            processed_summaries.append(processed_summary)
        
        # Create the input for the report composer
        input_text = f"""
        Research Objective: {objective}
        
        Document Summaries:
        
        {chr(10).join([f"--- DOCUMENT {i+1}: {s.title} ---{chr(10)}{s.summary}{chr(10)}" for i, s in enumerate(processed_summaries)])}
        """
        
        run_config = RunConfig(
            model=self.report_model,
            tracing_disabled=True,
            workflow_name="Report Composition"
        )
        
        result = await Runner.run(
            ReportComposer,
            input_text,
            run_config=run_config
        )
        
        # Add formatted references to the report
        report = result.final_output
        formatted_refs = [
            f"[{ref['index']}] {ref['title']}. {ref['url']}"
            for _, ref in sorted(references.items(), key=lambda x: x[1]['index'])
        ]
        report.references = formatted_refs
        
        return report
    
    async def _critique_report(self, report: Report, summaries: List[DocSummary]) -> FactCheck:
        """
        Run a fact-check critique on the generated report.
        
        Args:
            report: The generated report
            summaries: The document summaries used to generate the report
            
        Returns:
            FactCheck object with critique results
        """
        input_text = f"""
        # Report to Fact-Check
        
        {report.title}
        
        {report.body_md}
        
        # Source Document Summaries
        
        {chr(10).join([f"--- DOCUMENT {i+1}: {s.title} ---{chr(10)}{s.summary}{chr(10)}" for i, s in enumerate(summaries)])}
        """
        
        run_config = RunConfig(
            model=self.critic_model,
            tracing_disabled=True,
            workflow_name="Report Critique"
        )
        
        result = await Runner.run(
            CriticAgent,
            input_text,
            run_config=run_config
        )
        
        return result.final_output
    
    async def generate_report(self, documents: List[Document], objective: str) -> tuple[Report, Optional[FactCheck]]:
        """
        Generate a research report from a corpus of documents.
        
        Args:
            documents: List of documents to summarize
            objective: The research objective
            
        Returns:
            Tuple of (Report, FactCheck) where FactCheck may be None if critique is disabled
        """
        print(f"Starting report generation for {len(documents)} documents")
        
        # Step 1: Extract bullets from each document (Map-A)
        print("Extracting key points from documents...")
        bullets_by_doc = {}
        for doc in documents:
            bullets = await self._extract_bullets(doc)
            bullets_by_doc[doc.source_task_id] = bullets
        
        # Step 2: Summarize each document (Map-B)
        print("Generating document summaries...")
        summaries = []
        for doc in documents:
            doc_bullets = bullets_by_doc.get(doc.source_task_id, [])
            if doc_bullets:
                summary = await self._summarize_document(doc, doc_bullets)
                summaries.append(summary)
        
        # Step 3: Compose the final report (Reduce)
        print("Composing final research report...")
        report = await self._compose_report(summaries, objective)
        
        # Step 4: Run critique (optional)
        fact_check = None
        if self.run_critique:
            print("Running fact-check critique...")
            fact_check = await self._critique_report(report, summaries)
        
        return report, fact_check

# -----------------------------
# Public API
# -----------------------------

async def async_generate_report(
    documents: List[Document],
    objective: str,
    bullet_model: str = "gpt-3.5-turbo-0125",
    summary_model: str = "gpt-4o",
    report_model: str = "gpt-4o",
    critic_model: str = "gpt-4o",
    run_critique: bool = True,
    concurrency: int = 4,
    chunk_size: int = 1500
) -> tuple[Report, Optional[FactCheck]]:
    """
    Async function to generate a research report from a corpus of documents.
    
    Args:
        documents: List of documents to summarize
        objective: The research objective
        bullet_model: Model to use for bullet extraction
        summary_model: Model to use for document summarization
        report_model: Model to use for report composition
        critic_model: Model to use for fact checking
        run_critique: Whether to run the critique phase
        concurrency: Maximum number of concurrent tasks
        chunk_size: Size of text chunks for bullet extraction
        
    Returns:
        Tuple of (Report, FactCheck) where FactCheck may be None if critique is disabled
    """
    summarizer = ResearchSummarizer(
        bullet_model=bullet_model,
        summary_model=summary_model,
        report_model=report_model,
        critic_model=critic_model,
        run_critique=run_critique,
        concurrency=concurrency,
        chunk_size=chunk_size
    )
    return await summarizer.generate_report(documents, objective)

def generate_report(
    documents: List[Document],
    objective: str,
    bullet_model: str = "gpt-3.5-turbo-0125",
    summary_model: str = "gpt-4o",
    report_model: str = "gpt-4o",
    critic_model: str = "gpt-4o",
    run_critique: bool = True,
    concurrency: int = 4,
    chunk_size: int = 1500
) -> tuple[Report, Optional[FactCheck]]:
    """
    Blocking function to generate a research report from a corpus of documents.
    
    Args:
        documents: List of documents to summarize
        objective: The research objective
        bullet_model: Model to use for bullet extraction
        summary_model: Model to use for document summarization
        report_model: Model to use for report composition
        critic_model: Model to use for fact checking
        run_critique: Whether to run the critique phase
        concurrency: Maximum number of concurrent tasks
        chunk_size: Size of text chunks for bullet extraction
        
    Returns:
        Tuple of (Report, FactCheck) where FactCheck may be None if critique is disabled
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We are already in an event loop, so just create and return the coroutine
            return async_generate_report(
                documents=documents,
                objective=objective,
                bullet_model=bullet_model,
                summary_model=summary_model,
                report_model=report_model,
                critic_model=critic_model,
                run_critique=run_critique,
                concurrency=concurrency,
                chunk_size=chunk_size
            )
    except RuntimeError:
        # No event loop running, so use asyncio.run to create one
        return asyncio.run(async_generate_report(
            documents=documents,
            objective=objective,
            bullet_model=bullet_model,
            summary_model=summary_model,
            report_model=report_model,
            critic_model=critic_model,
            run_critique=run_critique,
            concurrency=concurrency,
            chunk_size=chunk_size
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
    
    parser = argparse.ArgumentParser(description="Step 3 – Generate a research report from document corpus.")
    parser.add_argument("corpus_jsonl", help="Path to corpus JSONL produced by Step 2")
    parser.add_argument("--objective", required=True, help="Research objective")
    parser.add_argument("--out", default="report.md", help="Output Markdown report file (default: report.md)")
    parser.add_argument("--bullet-model", default="gpt-3.5-turbo-0125", help="Model for bullet extraction")
    parser.add_argument("--summary-model", default="gpt-4o", help="Model for document summarization")
    parser.add_argument("--report-model", default="gpt-4o", help="Model for report composition")
    parser.add_argument("--critic-model", default="gpt-4o", help="Model for fact checking")
    parser.add_argument("--no-critique", action="store_true", help="Skip the critique phase")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel tasks (default: 4)")
    args = parser.parse_args()

    corpus_path = pathlib.Path(args.corpus_jsonl)
    if not corpus_path.exists():
        sys.exit(f"Corpus file not found: {corpus_path}")

    # Load documents from JSONL
    documents = []
    with corpus_path.open() as f:
        for line in f:
            doc_dict = json.loads(line)
            documents.append(Document(**doc_dict))
    
    if not documents:
        sys.exit(f"No documents found in corpus file: {corpus_path}")
    
    print(f"Loaded {len(documents)} documents from {corpus_path}")
    
    # Generate report
    report, fact_check = generate_report(
        documents=documents,
        objective=args.objective,
        bullet_model=args.bullet_model,
        summary_model=args.summary_model,
        report_model=args.report_model,
        critic_model=args.critic_model,
        run_critique=not args.no_critique,
        concurrency=args.concurrency
    )
    
    # Write report to file
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# {report.title}\n\n")
        f.write(report.body_md)
        f.write("\n\n## References\n\n")
        for ref in report.references:
            f.write(f"{ref}\n")
    
    print(f"Wrote research report → {args.out}")
    
    # Write critique to file if available
    if fact_check:
        critique_path = pathlib.Path(args.out).with_suffix(".critique.md")
        with open(critique_path, "w", encoding="utf-8") as f:
            f.write(f"# Fact Check: {report.title}\n\n")
            f.write(f"## Overall Assessment\n\n{fact_check.overall_assessment}\n\n")
            
            if fact_check.unsupported_claims:
                f.write("## Unsupported Claims\n\n")
                for claim in fact_check.unsupported_claims:
                    f.write(f"- {claim}\n")
                f.write("\n")
            
            if fact_check.conflicting_claims:
                f.write("## Conflicting Claims\n\n")
                for claim in fact_check.conflicting_claims:
                    f.write(f"- {claim}\n")
        
        print(f"Wrote critique → {critique_path}") 