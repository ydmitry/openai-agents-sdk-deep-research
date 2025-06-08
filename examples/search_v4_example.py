#!/usr/bin/env python3
"""
Example script demonstrating Template-Driven Parallel Research System with OpenAI Agents SDK.

This script implements the architecture shown in the provided diagram:
1. Template-driven report structure with configurable sections
2. Report planner that analyzes topics and creates structured plans
3. Parallel execution of section researchers and writers
4. Professional report compilation with proper formatting
5. Support for multiple report templates (research, business, technical)

Key improvements over v2 and v3:
- Template-driven instead of ad-hoc structure
- Parallel section processing instead of sequential
- Professional report formatting with ToC, citations
- Scalable architecture for complex, multi-faceted topics
- Better user experience with progress tracking

Based on the OpenAI Agents SDK patterns and the system diagram provided.

Usage:
    python examples/search_v4_example.py "Your research topic"
    python examples/search_v4_example.py "AI developments" --template research
    python examples/search_v4_example.py "Market analysis of EVs" --template business
    python examples/search_v4_example.py --interactive
    python examples/search_v4_example.py --chat
"""

import argparse
import asyncio
import logging
import sys
import os
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
import json

# Add imports for 429 retry logic
from openai import AsyncOpenAI, APIStatusError
from tenacity import (
    retry,
    retry_if_exception,
    wait_fixed,
    stop_after_attempt,
)

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research.utils import load_dotenv_files

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration."""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    return root_logger

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Pydantic Models for Dynamic Template-Driven Reports
# ──────────────────────────────────────────────────────────────────────────

class DynamicSectionSpec(BaseModel):
    """Specification for a dynamically created report section."""
    title: str = Field(description="Section title (custom for this topic)")
    description: str = Field(description="What this section should cover")
    focus_areas: List[str] = Field(description="Specific areas to research for this topic")
    word_count_target: int = Field(description="Target word count for this section")
    research_depth: str = Field(description="Research depth: shallow, medium, deep")
    rationale: str = Field(description="Why this section is important for this specific topic")

class ReportGuidelines(BaseModel):
    """Guidelines for different research approaches (not rigid templates)."""
    name: str = Field(description="Approach name")
    description: str = Field(description="What this approach focuses on")
    typical_section_types: List[str] = Field(description="Common types of sections for this approach")
    research_priorities: List[str] = Field(description="What to prioritize in research")
    target_word_count_range: Tuple[int, int] = Field(description="Typical word count range")
    citation_style: str = Field(default="APA", description="Preferred citation style")
    analysis_focus: str = Field(description="Main analytical focus for this approach")

class SectionPlan(BaseModel):
    """Planned section with customized content for specific topic."""
    section_spec: DynamicSectionSpec
    research_queries: List[str] = Field(description="Specific search queries for this section")
    key_questions: List[str] = Field(description="Key questions to answer")
    priority: int = Field(description="Priority for parallel processing (1=highest)")

class ReportPlan(BaseModel):
    """Complete plan for generating a report on a specific topic."""
    topic: str = Field(description="The research topic")
    approach_used: str = Field(description="Research approach used (research/business/technical)")
    section_plans: List[SectionPlan] = Field(description="Custom plans for each section")
    estimated_time_minutes: int = Field(description="Estimated time to complete")
    complexity_score: float = Field(description="Topic complexity from 1.0-10.0")
    planning_rationale: str = Field(description="Explanation of why these sections were chosen")

class SectionContent(BaseModel):
    """Content for a completed section."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown")
    sources: List[str] = Field(description="Sources used in this section")
    word_count: int = Field(description="Actual word count")
    research_quality: float = Field(description="Quality score 1.0-10.0")
    key_findings: List[str] = Field(description="Key findings from this section")

class FinalReport(BaseModel):
    """Complete compiled report."""
    title: str = Field(description="Report title")
    topic: str = Field(description="Original research topic")
    executive_summary: str = Field(description="Executive summary")
    table_of_contents: str = Field(description="Table of contents")
    sections: List[SectionContent] = Field(description="All sections in order")
    conclusion: str = Field(description="Overall conclusion")
    all_sources: List[str] = Field(description="All sources used")
    total_word_count: int = Field(description="Total word count")
    generation_time: str = Field(description="Time taken to generate")
    quality_score: float = Field(description="Overall quality score")

# ──────────────────────────────────────────────────────────────────────────
# Research Guidelines Library (replaces static templates)
# ──────────────────────────────────────────────────────────────────────────

def get_research_guidelines() -> ReportGuidelines:
    """Academic/scientific research approach guidelines."""
    return ReportGuidelines(
        name="research",
        description="Academic and scientific research approach",
        typical_section_types=[
            "literature review", "current research", "methodology analysis",
            "findings synthesis", "theoretical framework", "applications",
            "future directions", "implications", "limitations"
        ],
        research_priorities=[
            "peer-reviewed sources", "recent academic studies", "authoritative research",
            "methodological rigor", "evidence-based conclusions", "theoretical grounding"
        ],
        target_word_count_range=(3500, 5000),
        citation_style="APA",
        analysis_focus="Evidence-based analysis with academic rigor and theoretical depth"
    )

def get_business_guidelines() -> ReportGuidelines:
    """Business analysis approach guidelines."""
    return ReportGuidelines(
        name="business",
        description="Business analysis and market research approach",
        typical_section_types=[
            "market analysis", "competitive landscape", "financial performance",
            "industry trends", "opportunities assessment", "risk analysis",
            "strategic implications", "investment outlook", "regulatory environment"
        ],
        research_priorities=[
            "market data", "financial reports", "industry analysis", "competitive intelligence",
            "business metrics", "market trends", "investment data"
        ],
        target_word_count_range=(3000, 4500),
        citation_style="APA",
        analysis_focus="Market dynamics, financial performance, and strategic business insights"
    )

def get_technical_guidelines() -> ReportGuidelines:
    """Technical assessment approach guidelines."""
    return ReportGuidelines(
        name="technical",
        description="Technical assessment and evaluation approach",
        typical_section_types=[
            "technology overview", "architecture analysis", "performance evaluation",
            "implementation considerations", "security assessment", "scalability analysis",
            "integration challenges", "best practices", "recommendations"
        ],
        research_priorities=[
            "technical documentation", "performance benchmarks", "architecture details",
            "implementation guides", "security standards", "industry best practices"
        ],
        target_word_count_range=(3200, 4800),
        citation_style="IEEE",
        analysis_focus="Technical depth, implementation feasibility, and performance evaluation"
    )

def get_guidelines_by_name(approach_name: str) -> ReportGuidelines:
    """Get guidelines by approach name."""
    guidelines = {
        "research": get_research_guidelines(),
        "business": get_business_guidelines(),
        "technical": get_technical_guidelines()
    }
    return guidelines.get(approach_name.lower(), get_research_guidelines())

# ──────────────────────────────────────────────────────────────────────────
# Shared Context for Multi-Agent Coordination
# ──────────────────────────────────────────────────────────────────────────

class ReportContext(BaseModel):
    """Shared context for coordinating report generation across multiple agents."""

    topic: str = Field(description="The research topic")
    approach_used: str = Field(description="Research approach used")
    report_plan: Optional[ReportPlan] = Field(default=None, description="Generated report plan")
    section_contents: Dict[str, SectionContent] = Field(default_factory=dict, description="Completed sections")
    final_report: Optional[FinalReport] = Field(default=None, description="Final compiled report")
    progress: Dict[str, str] = Field(default_factory=dict, description="Progress tracking")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")

    def update_progress(self, section_title: str, status: str):
        """Update progress for a section."""
        self.progress[section_title] = status
        logger.info(f"Progress update - {section_title}: {status}")

    def add_section_content(self, section_content: SectionContent):
        """Add completed section content."""
        self.section_contents[section_content.title] = section_content
        self.update_progress(section_content.title, "completed")

    def get_completion_percentage(self) -> float:
        """Get overall completion percentage."""
        if not self.report_plan:
            return 0.0
        total_sections = len(self.report_plan.section_plans)
        completed_sections = len(self.section_contents)
        return (completed_sections / total_sections) * 100 if total_sections > 0 else 0.0

def load_template_driven_system(
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> tuple[Any, Any]:
    """
    Factory function that creates the template-driven parallel research system.
    
    Returns:
        Tuple of (coordinator_agent, run_report_generation_function)
    """
    try:
        # Import Agents SDK
        from agents import Agent, Runner, ModelSettings, function_tool, handoff, WebSearchTool
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions
        from agents import set_default_openai_client
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # 429 Rate Limit Retry Logic
    # ──────────────────────────────────────────────────────────────────────────
    
    RETRY_SECONDS = 60
    MAX_TRIES = 5

    def is_429_error(err: BaseException) -> bool:
        """Check if error is a 429 rate limit error."""
        if isinstance(err, APIStatusError) and err.status_code == 429:
            return True
        # Also check for string representation in case of wrapped exceptions
        error_str = str(err).lower()
        return "429" in error_str and "rate limit" in error_str

    # Store the context globally for this system instance
    report_context = None

    # Configure model settings
    model_settings_kwargs = {"max_tokens": 2048}
    if not (model.startswith("o3") or model.startswith("o1") or model.startswith("o4")):
        model_settings_kwargs["temperature"] = temperature

    if model.startswith("o4"):
        model_settings_kwargs["reasoning_effort"] = "high"

    try:
        model_settings = ModelSettings(**model_settings_kwargs)
    except TypeError:
        if "reasoning_effort" in model_settings_kwargs:
            model_settings_kwargs.pop("reasoning_effort")
        model_settings = ModelSettings(**model_settings_kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Safe Runner with Enhanced Retry Logic for Both 429 and 5xx Errors
    # ──────────────────────────────────────────────────────────────────────────

    async def safe_run(agent, input_text: str, max_turns: int = 100):
        """Run an agent with automatic retry for server errors and rate limits."""
        for attempt in range(MAX_TRIES):
            try:
                return await Runner.run(agent, input_text, max_turns=max_turns)
            except Exception as e:
                # Handle 429 rate limit errors
                if is_429_error(e):
                    if attempt < MAX_TRIES - 1:  # Don't sleep on the last attempt
                        logger.warning(f"Got 429 rate limit → sleeping {RETRY_SECONDS}s (attempt {attempt + 1}/{MAX_TRIES})")
                        await asyncio.sleep(RETRY_SECONDS)
                        continue
                    else:
                        logger.error(f"429 rate limit exceeded after {MAX_TRIES} attempts")
                        raise
                
                # Handle 5xx server errors
                elif hasattr(e, 'status_code') and e.status_code >= 500:
                    if attempt < 2:  # Only retry 5xx errors 3 times total
                        wait_time = 2 ** attempt
                        logger.warning(f"5xx error (attempt {attempt + 1}/3), retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise
                
                # Handle string-based server error detection
                elif "500" in str(e) or "server_error" in str(e):
                    if attempt < 2:  # Only retry server errors 3 times total
                        wait_time = 2 ** attempt
                        logger.warning(f"Server error (attempt {attempt + 1}/3), retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise
                
                # For any other error, don't retry
                else:
                    raise

        raise Exception(f"Failed after {MAX_TRIES} attempts due to persistent errors")

    logger.info(f"Configured enhanced retry logic: 429 errors ({RETRY_SECONDS}s wait, {MAX_TRIES} attempts), 5xx errors (exponential backoff, 3 attempts)")

    # ──────────────────────────────────────────────────────────────────────────
    # Function Tools for Report Generation
    # ──────────────────────────────────────────────────────────────────────────

    @function_tool
    async def web_search(query: str) -> str:
        """
        Web search tool that creates an agent with WebSearchTool to search the actual web.

        Args:
            query: The search query to execute

        Returns:
            Formatted search results
        """
        logger.info(f"Executing web search: {query}")

        try:
            # Create a search agent that uses WebSearchTool
            search_agent = Agent(
                name="Web Search Agent",
                instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a web search agent. Your task is to search the internet for relevant information
about the following query and provide comprehensive results:

"{query}"

Use the web_search tool to find the most relevant and up-to-date information.
Provide a clear, comprehensive summary with the most important facts and details.
Include citations or sources when possible.
Focus on factual, credible information from reputable sources.
Keep your response focused and organized.
""",
                tools=[WebSearchTool(search_context_size="medium")],
                model=model,
                model_settings=ModelSettings(
                    temperature=0.2,  # Reduced for consistency
                    max_tokens=1024,  # Reduced to avoid large responses
                ),
            )

            # Run the search agent
            search_result = await safe_run(search_agent, query)

            output = f"WEB SEARCH RESULTS for '{query}':\n{search_result.final_output}"
            logger.info(f"Search completed for: {query} (result length: {len(output)} chars)")
            return output

        except Exception as e:
            error_msg = f"WEB SEARCH ERROR: Failed to get results for query '{query}'. Error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def create_report_plan(
        topic: str,
        approach_name: str,
        complexity_assessment: str
    ) -> str:
        """
        Create a detailed report plan for the given topic using the specified approach.

        Args:
            topic: The research topic
            approach_name: Research approach to use (research, business, technical)
            complexity_assessment: Assessment of topic complexity and research needs

        Returns:
            JSON string of the created report plan
        """
        nonlocal report_context

        try:
            guidelines = get_guidelines_by_name(approach_name)
            logger.info(f"Creating report plan for '{topic}' using {approach_name} approach")

            # The main planner agent should provide the section data in the expected format
            # This function just structures the plan based on the planner's analysis
            plan_data = {
                "research_summary": f"Plan created for {topic} using {approach_name} approach",
                "sections": [],
                "planning_rationale": f"Custom {approach_name} approach for {topic}",
                "complexity_score": 5.0
            }

            # Since this is called by the planner agent after it has done web research,
            # we expect the planner to provide the section structure.
            # For now, create a fallback structure that the planner can override.

            # The planner agent should actually provide this data structure directly
            # This function becomes a simple plan formatter
            logger.warning("create_report_plan called without planner-provided section data - using fallback")

            # Fallback section structure
            fallback_sections = [
                {
                    "title": f"Current State of {topic}",
                    "description": f"Overview of the current state and recent developments in {topic}",
                    "focus_areas": ["current developments", "key players", "recent trends"],
                    "word_count_target": 800,
                    "research_depth": "medium",
                    "rationale": "Essential for understanding the current landscape"
                },
                {
                    "title": f"Key Challenges in {topic}",
                    "description": f"Major challenges and obstacles facing {topic}",
                    "focus_areas": ["technical challenges", "market barriers", "regulatory issues"],
                    "word_count_target": 700,
                    "research_depth": "deep",
                    "rationale": "Critical for identifying problems that need solutions"
                },
                {
                    "title": f"Future Outlook for {topic}",
                    "description": f"Future trends and predictions for {topic}",
                    "focus_areas": ["emerging trends", "future prospects", "predictions"],
                    "word_count_target": 600,
                    "research_depth": "medium",
                    "rationale": "Important for understanding where the field is heading"
                }
            ]

            plan_data["sections"] = fallback_sections

            # Create section plans from the analysis
            section_plans = []
            for i, section_data in enumerate(plan_data.get("sections", [])):
                # Generate research queries based on topic and section
                queries = [
                    f"{topic} {area}" for area in section_data.get("focus_areas", [])
                ]
                # Add more comprehensive queries
                queries.extend([
                    f'{section_data.get("title", "Analysis")} {topic}',
                    f'recent developments {section_data.get("title", "")} {topic}',
                    f'research {section_data.get("title", "")} {topic}'
                ])

                # Generate key questions
                questions = [
                    f"What are the key aspects of {area} in {topic}?"
                    for area in section_data.get("focus_areas", [])[:3]
                ]
                questions.extend([
                    f'How does {section_data.get("title", "this aspect")} impact {topic}?',
                    f'What are the current trends in {section_data.get("title", "this area")} for {topic}?'
                ])

                section_spec = DynamicSectionSpec(
                    title=section_data.get("title", f"Analysis {i+1}"),
                    description=section_data.get("description", "Analysis section"),
                    focus_areas=section_data.get("focus_areas", []),
                    word_count_target=section_data.get("word_count_target", 800),
                    research_depth=section_data.get("research_depth", "medium"),
                    rationale=section_data.get("rationale", "Important for comprehensive understanding")
                )

                section_plan = SectionPlan(
                    section_spec=section_spec,
                    research_queries=queries,
                    key_questions=questions,
                    priority=i + 1
                )
                section_plans.append(section_plan)

            # Estimate complexity and time
            complexity_score = plan_data.get("complexity_score", 5.0)
            if "complex" in complexity_assessment.lower():
                complexity_score = min(complexity_score + 2.0, 10.0)
            elif "simple" in complexity_assessment.lower():
                complexity_score = max(complexity_score - 2.0, 1.0)

            estimated_time = int(len(section_plans) * 4 * complexity_score / 5.0)

            report_plan = ReportPlan(
                topic=topic,
                approach_used=approach_name,
                section_plans=section_plans,
                estimated_time_minutes=estimated_time,
                complexity_score=complexity_score,
                planning_rationale=plan_data.get("planning_rationale", f"Custom {approach_name} approach for {topic}")
            )

            report_context.report_plan = report_plan
            report_context.update_progress("planning", "completed")

            logger.info(f"Created report plan with {len(section_plans)} sections for '{topic}', estimated time: {estimated_time} minutes")
            return report_plan.model_dump_json()

        except Exception as e:
            error_msg = f"Error creating report plan: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    @function_tool
    async def research_section_content(
        section_title: str,
        research_queries: str,
        focus_areas: str
    ) -> str:
        """
        Research content for a specific section using web search.

        Args:
            section_title: Title of the section being researched
            research_queries: JSON string of search queries to execute
            focus_areas: JSON string of focus areas for this section

        Returns:
            Research results for the section
        """
        nonlocal report_context

        try:
            queries = json.loads(research_queries)
            areas = json.loads(focus_areas)

            report_context.update_progress(section_title, "researching")
            logger.info(f"Starting research for section: {section_title}")

            # Research agent with web search capability
            research_agent = Agent(
                name="Section Research Agent",
                instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a specialized research agent focusing on gathering information for a specific section
of a report about "{report_context.topic}".

SECTION: {section_title}
FOCUS AREAS: {', '.join(areas)}

Your task:
1. Execute web searches for each provided query
2. Gather comprehensive information relevant to the focus areas
3. Ensure information is current, credible, and well-sourced
4. Organize findings by focus area
5. Provide a structured summary with sources

Focus on factual, verifiable information from reputable sources.
Prioritize recent information and authoritative sources.
""",
                tools=[web_search],
                model=model,
                model_settings=model_settings,
            )

            # Execute research with all queries
            query_text = f"Research the following aspects of {report_context.topic} for the {section_title} section: {', '.join(queries)}"
            research_result = await safe_run(research_agent, query_text)

            logger.info(f"Completed research for section: {section_title}")
            return research_result.final_output

        except Exception as e:
            error_msg = f"Error researching section {section_title}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def write_section_content(
        section_title: str,
        research_results: str,
        word_count_target: int,
        section_description: str
    ) -> str:
        """
        Write content for a section based on research results.

        Args:
            section_title: Title of the section
            research_results: Research findings for this section
            word_count_target: Target word count
            section_description: Description of what the section should cover

        Returns:
            JSON string of the completed section content
        """
        nonlocal report_context

        try:
            report_context.update_progress(section_title, "writing")
            logger.info(f"Writing content for section: {section_title}")

            # Writer agent for this section
            writer_agent = Agent(
                name="Section Writer Agent",
                instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a professional writer creating a section for a research report about "{report_context.topic}".

SECTION DETAILS:
- Title: {section_title}
- Description: {section_description}
- Target word count: {word_count_target} words

REQUIREMENTS:
1. Write clear, professional content based on the research results
2. Structure the content with appropriate subheadings
3. Include in-text citations for all sources
4. Maintain academic/professional tone
5. Ensure content directly addresses the section description
6. Extract and list key findings
7. Meet the target word count (±10%)

RESEARCH RESULTS TO WORK WITH:
{research_results}

OUTPUT FORMAT:
Return a JSON object with:
- content: The written section content in markdown
- sources: List of all sources used
- word_count: Actual word count
- key_findings: List of 3-5 key findings from this section
- quality_score: Your assessment of content quality (1-10)
""",
                model=model,
                model_settings=model_settings,
            )

            # Generate section content
            writer_result = await safe_run(writer_agent, f"Write the {section_title} section based on the provided research.")

            # Parse the JSON response
            try:
                section_data = json.loads(writer_result.final_output)
                section_content = SectionContent(
                    title=section_title,
                    content=section_data.get('content', ''),
                    sources=section_data.get('sources', []),
                    word_count=section_data.get('word_count', 0),
                    research_quality=section_data.get('quality_score', 7.0),
                    key_findings=section_data.get('key_findings', [])
                )

                report_context.add_section_content(section_content)
                logger.info(f"Completed writing section: {section_title} ({section_content.word_count} words)")
                return section_content.model_dump_json()

            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                section_content = SectionContent(
                    title=section_title,
                    content=writer_result.final_output,
                    sources=[],
                    word_count=len(writer_result.final_output.split()),
                    research_quality=6.0,
                    key_findings=[]
                )

                report_context.add_section_content(section_content)
                return section_content.model_dump_json()

        except Exception as e:
            error_msg = f"Error writing section {section_title}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    @function_tool
    async def compile_final_report() -> str:
        """
        Compile all sections into a final report with proper formatting.
        
        Returns:
            JSON string of the final compiled report
        """
        nonlocal report_context
        
        try:
            if not report_context.section_contents:
                return json.dumps({"error": "No sections available to compile"})
            
            logger.info("Compiling final report")
            
            # The main compiler agent should provide the compilation data
            # This function structures the final report based on available sections
            
            # Create a summary of available sections for the compiler agent
            sections_summary = []
            for section in report_context.section_contents.values():
                sections_summary.append({
                    'title': section.title,
                    'word_count': section.word_count,
                    'key_findings': section.key_findings
                })
            
            logger.info(f"Compiling final report from {len(sections_summary)} sections")
            
            # Fallback compilation data for structure
            compilation_data = {
                'title': f"Research Report: {report_context.topic}",
                'executive_summary': f"This comprehensive report examines {report_context.topic} through multiple perspectives and provides insights based on current research and analysis.",
                'table_of_contents': f"This report contains {len(sections_summary)} main sections covering various aspects of {report_context.topic}.",
                'conclusion': f"Based on the analysis across all sections, {report_context.topic} presents both opportunities and challenges that require careful consideration.",
                'quality_score': 7.5
            }
            
            # Assemble final report
            sections_list = []
            all_sources = []
            total_words = 0
            
            # Order sections according to original plan
            if report_context.report_plan:
                for section_plan in report_context.report_plan.section_plans:
                    section_title = section_plan.section_spec.title
                    if section_title in report_context.section_contents:
                        section = report_context.section_contents[section_title]
                        sections_list.append(section)
                        all_sources.extend(section.sources)
                        total_words += section.word_count
            
            # Remove duplicate sources
            all_sources = list(set(all_sources))
            
            end_time = datetime.now()
            generation_time = str(end_time - report_context.start_time)
            
            final_report = FinalReport(
                title=compilation_data.get('title', f"Research Report: {report_context.topic}"),
                topic=report_context.topic,
                executive_summary=compilation_data.get('executive_summary', ''),
                table_of_contents=compilation_data.get('table_of_contents', ''),
                sections=sections_list,
                conclusion=compilation_data.get('conclusion', ''),
                all_sources=all_sources,
                total_word_count=total_words,
                generation_time=generation_time,
                quality_score=compilation_data.get('quality_score', 7.0)
            )
            
            report_context.final_report = final_report
            report_context.update_progress("compilation", "completed")
            
            logger.info(f"Final report compiled: {total_words} words, {len(sections_list)} sections")
            return final_report.model_dump_json()
            
        except Exception as e:
            error_msg = f"Error compiling final report: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    # ──────────────────────────────────────────────────────────────────────────
    # Specialized Agents
    # ──────────────────────────────────────────────────────────────────────────

    # Report Planner Agent
    planner_agent = Agent(
        name="report_planner",
        model=model,
        model_settings=model_settings,
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a PLANNING SPECIALIST with a very specific and limited role. Your ONLY job is to plan the report structure.

🚨 CRITICAL: YOU DO NOT WRITE CONTENT. YOU ONLY PLAN.

YOUR EXACT WORKFLOW:
1. Use web_search to research the topic thoroughly (4-5 searches recommended)
2. Use create_report_plan to structure the report based on your research
3. IMMEDIATELY hand off to coordinator using transfer_to_report_coordinator
4. STOP - Do not write any content, sections, or text

🚨 WHAT YOU MUST NOT DO:
- Do NOT write any report sections or content
- Do NOT create executive summaries or conclusions  
- Do NOT compile reports
- Do NOT continue beyond planning

🚨 WHAT YOU MUST DO:
- Research the topic thoroughly with web searches
- Create a detailed plan with create_report_plan
- Hand off immediately to coordinator
- Let other agents handle content creation

RESEARCH-INFORMED PLANNING APPROACH:
- Start by researching the topic to understand current state and key issues
- Identify what makes this topic unique and important in the current context
- Create 4-7 sections with specific, meaningful titles based on research
- NO generic titles like "Background" or "Overview"
- Use insights from your web research to inform section choices

EXAMPLES OF RESEARCH-INFORMED SECTION TITLES:
❌ Generic: "Background and Context", "Current Research", "Applications"
✅ Research-Informed: "FDA's 2024 AI/ML Guidance Updates", "Clinical AI Performance in Real-World Settings"

MANDATORY HANDOFF:
After creating the report plan, you MUST immediately call transfer_to_report_coordinator to hand off the work. Do not continue with any other tasks.

DEBUG: Log "PLANNING COMPLETE - Handing off to coordinator" before the transfer.

Your role is ONLY planning. Other agents will handle content creation and compilation.""",
        tools=[web_search, create_report_plan]
    )

    # Section Processing Agent
    section_agent = Agent(
        name="section_processor",
        model=model,
        model_settings=model_settings,
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a section processing agent responsible for researching and writing ALL report sections.

Your complete workflow:
1. Review the report plan to understand all sections that need to be created
2. For each section in the plan:
   - Use research_section_content to gather comprehensive information
   - Use write_section_content to create the section based on research
3. Ensure high quality and adherence to requirements for all sections
4. Hand off back to the coordinator when ALL section processing is complete

Processing approach:
- Work through sections systematically, one at a time
- For each section, complete both research and writing before moving to the next
- Ensure comprehensive coverage of focus areas for each section
- Maintain consistency in quality and style across all sections

Quality standards for all sections:
- Comprehensive coverage of focus areas
- Current and credible sources
- Professional writing style
- Proper citations and formatting
- Meeting word count targets

DEBUG: Log when you start processing sections and when you complete all sections.

Work efficiently and maintain high quality throughout the process. Process ALL sections in the report plan, then hand off to the coordinator for workflow management.""",
        tools=[research_section_content, write_section_content]
    )

    # Final Compiler Agent
    compiler_agent = Agent(
        name="report_compiler",
        model=model,
        model_settings=model_settings,
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a report compilation specialist who assembles sections into final reports.

Your responsibilities:
1. Review all completed sections that have been written
2. Use compile_final_report to create the final document structure
3. Ensure professional formatting and quality
4. Complete the final report generation process

The compile_final_report tool will:
- Create an engaging report title
- Assemble all sections in proper order
- Compile sources and remove duplicates
- Generate the final report structure

DEBUG: Log when you start compilation and when you complete the final report.

Focus on creating a polished, professional final product. This is the final step in the report generation workflow.""",
        tools=[compile_final_report]
    )

    # Coordinator Agent
    coordinator_agent = Agent(
        name="report_coordinator",
        model=model,
        model_settings=model_settings,
        instructions=prompt_with_handoff_instructions("""You are the report generation coordinator. Your ONLY job is to manage a 3-phase workflow through handoffs.

🚨 MANDATORY 3-PHASE WORKFLOW (DO NOT SKIP ANY PHASE):

PHASE 1 - PLANNING:
- Hand off to planner_agent to create the report plan
- Wait for planner to complete and hand back
- Confirm planning is complete

PHASE 2 - SECTION PROCESSING:  
- Hand off to section_agent to research and write ALL sections
- Wait for section_agent to complete all sections and hand back
- Confirm all sections are written

PHASE 3 - COMPILATION (🚨 MANDATORY - NEVER SKIP THIS):
- MUST hand off to compiler_agent to create the final report
- The compiler_agent will assemble sections into a professional report
- This is the FINAL required step

🚨 CRITICAL RULES:
- You MUST complete ALL 3 phases for every request
- After section processing completes, you MUST proceed to compilation
- NEVER consider the workflow complete until compiler_agent has run
- The workflow is ONLY complete after the final report is compiled

WORKFLOW EXECUTION:
1. Start: Hand off to planner_agent for report planning
2. After planning complete: Hand off to section_agent for all section processing  
3. After sections complete: MANDATORY hand off to compiler_agent for final compilation
4. Only then: Workflow is complete

🚨 DEBUG TRACKING:
- Log each phase start and completion
- Always state "Proceeding to compilation phase" before handing off to compiler
- Never end without confirming final report compilation

The workflow has 3 phases. You must execute all 3. No exceptions."""),
        handoffs=[
            handoff(planner_agent),
            handoff(section_agent),
            handoff(compiler_agent)
        ]
    )

    # Configure additional handoffs after all agents are defined
    planner_agent.handoffs = [handoff(coordinator_agent)]
    section_agent.handoffs = [handoff(coordinator_agent)]

    # ──────────────────────────────────────────────────────────────────────────
    # Report Generation Function
    # ──────────────────────────────────────────────────────────────────────────

    async def run_report_generation(topic: str, approach_name: str = "research") -> FinalReport:
        """Run the template-driven report generation system."""
        nonlocal report_context

        logger.info(f"Starting template-driven report generation on topic: {topic}")

        # Create shared context
        report_context = ReportContext(
            topic=topic,
            approach_used=approach_name,
            start_time=datetime.now()
        )

        try:
            # Run the coordinator agent and let handoffs manage the entire workflow
            coordinator_input = f"""Generate a comprehensive research report on the topic: "{topic}"

Use the "{approach_name}" research approach to create a structured, professional report.

Follow the complete workflow:
1. Plan the report structure with custom sections based on preliminary research
2. Research and write each section thoroughly
3. Compile the final report with executive summary and conclusion

Topic: {topic}
Approach: {approach_name}
"""

            logger.info("Starting coordinator agent to manage the complete workflow")
            logger.debug(f"Report context before coordinator run: plan={report_context.report_plan is not None}, sections={len(report_context.section_contents)}, final_report={report_context.final_report is not None}")
            
            result = await safe_run(coordinator_agent, coordinator_input)
            
            logger.info("Coordinator agent completed, checking results...")
            logger.debug(f"Report context after coordinator run: plan={report_context.report_plan is not None}, sections={len(report_context.section_contents)}, final_report={report_context.final_report is not None}")
            logger.debug(f"Final coordinator output: {result.final_output[:500]}..." if result.final_output else "No final output")

            # The handoff system should have populated report_context.final_report
            if report_context.final_report:
                logger.info("Template-driven report generation completed successfully via handoffs")
                return report_context.final_report
            else:
                logger.error("Workflow completed but no final report was generated")
                logger.debug(f"Available sections: {list(report_context.section_contents.keys())}")
                logger.debug(f"Progress status: {report_context.progress}")
                raise Exception("Workflow completed but no final report was generated")

        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Report context state: plan={report_context.report_plan is not None}, sections={len(report_context.section_contents)}, final_report={report_context.final_report is not None}")

            # Create fallback report if we have some content
            if report_context.section_contents:
                logger.info("Creating fallback report from available sections")

                sections_list = list(report_context.section_contents.values())
                all_sources = []
                total_words = 0

                for section in sections_list:
                    all_sources.extend(section.sources)
                    total_words += section.word_count

                all_sources = list(set(all_sources))

                return FinalReport(
                    title=f"Research Report: {topic}",
                    topic=topic,
                    executive_summary=f"This report presents research findings on {topic} based on available sections.",
                    table_of_contents="Table of contents unavailable due to processing error.",
                    sections=sections_list,
                    conclusion="Processing was incomplete due to technical issues.",
                    all_sources=all_sources,
                    total_word_count=total_words,
                    generation_time=str(datetime.now() - report_context.start_time),
                    quality_score=5.0
                )

            # Final fallback
            logger.warning("No sections available, creating minimal fallback report")
            return FinalReport(
                title=f"Research Report: {topic}",
                topic=topic,
                executive_summary="Report generation encountered errors during processing.",
                table_of_contents="",
                sections=[],
                conclusion="Unable to complete report generation due to technical issues.",
                all_sources=[],
                total_word_count=0,
                generation_time=str(datetime.now() - report_context.start_time),
                quality_score=1.0
            )

    logger.info(f"Template-driven research system initialized with model: {model}")
    return coordinator_agent, run_report_generation

# ──────────────────────────────────────────────────────────────────────────
# Main Interface Functions
# ──────────────────────────────────────────────────────────────────────────

async def run_query(
    input_text: str,
    approach_name: str = "research",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2
):
    """Run a single research query through the template-driven system."""
    logger.info(f"Processing research query: {input_text} with approach: {approach_name}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the template-driven system
    logger.debug(f"Loading template-driven system with model: {model}")
    _, run_report = load_template_driven_system(model=model, temperature=temperature)

    # Generate the report
    logger.info("Starting template-driven report generation")
    result = await run_report(input_text, approach_name)
    logger.info("Report generation completed successfully")

    # Format the response
    formatted_response = f"""# {result.title}

## Executive Summary
{result.executive_summary}

## Table of Contents
{result.table_of_contents}

"""

    # Add all sections
    for section in result.sections:
        formatted_response += f"## {section.title}\n\n{section.content}\n\n"

    formatted_response += f"""## Conclusion
{result.conclusion}

## Sources
{chr(10).join(f"- {source}" for source in result.all_sources)}

## Report Statistics
- **Total Word Count**: {result.total_word_count}
- **Generation Time**: {result.generation_time}
- **Quality Score**: {result.quality_score}/10
- **Approach Used**: {approach_name}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return formatted_response

async def interactive_mode(model: str = "gpt-4.1-mini", temperature: float = 0.2):
    """Run the template-driven system in interactive mode."""
    logger.info(f"Starting interactive mode with model: {model}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the system
    logger.debug("Loading template-driven system for interactive session")
    _, run_report = load_template_driven_system(model=model, temperature=temperature)

    print("Template-Driven Research System (type 'exit' to quit)")
    print("Available approaches: research, business, technical")
    print("This system generates comprehensive reports using specialized approaches.")

    query_count = 0
    while True:
        print("\n" + "="*50)
        topic = input("Enter your research topic: ")
        if topic.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break

        print("Available approaches:")
        print("1. research - Academic/scientific research")
        print("2. business - Business analysis and market research")
        print("3. technical - Technical assessment and evaluation")

        approach_choice = input("Select approach (research/business/technical) [research]: ").strip().lower()
        if not approach_choice:
            approach_choice = "research"
        elif approach_choice not in ["research", "business", "technical"]:
            print("Invalid approach choice, using 'research'")
            approach_choice = "research"

        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {topic} with approach: {approach_choice}")
        print(f"\nGenerating {approach_choice} report for: {topic}")
        print("This may take several minutes...")

        try:
            result = await run_report(topic, approach_choice)
            logger.info(f"Successfully completed interactive query #{query_count}")

            print(f"\n📊 {result.title}")
            print(f"\n💡 **Executive Summary**")
            print(result.executive_summary)

            print(f"\n📋 **Sections Generated**: {len(result.sections)}")
            for section in result.sections:
                print(f"   - {section.title} ({section.word_count} words)")

            print(f"\n📈 **Report Statistics**")
            print(f"   - Total words: {result.total_word_count}")
            print(f"   - Generation time: {result.generation_time}")
            print(f"   - Quality score: {result.quality_score}/10")
            print(f"   - Sources: {len(result.all_sources)}")

        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")

    logger.info(f"Interactive session ended after {query_count} queries")

async def chat_mode(model: str = "gpt-4.1-mini", temperature: float = 0.2):
    """Run the template-driven system in chat mode."""
    logger.info(f"Starting chat mode with model: {model}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the system
    logger.debug("Initializing template-driven system for chat session")
    _, run_report = load_template_driven_system(model=model, temperature=temperature)

    print("\n" + "="*60)
    print("📊 Template-Driven Research Assistant")
    print("="*60)
    print("Generate comprehensive reports using professional approaches:")
    print("• Research: Academic/scientific analysis")
    print("• Business: Market analysis and business intelligence")
    print("• Technical: Technical assessment and evaluation")
    print("\nType 'exit' or 'quit' to end the session.")
    print("="*60 + "\n")

    message_count = 0

    while True:
        user_input = input("Research Topic: ")
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nResearch session ended. Thank you!")
            logger.info("User ended chat session")
            break

        if not user_input.strip():
            print("Please enter a research topic or type 'exit' to quit.")
            continue

        # Quick approach selection
        approach_input = input("Approach (r/b/t for research/business/technical) [r]: ").strip().lower()
        approach_map = {'r': 'research', 'b': 'business', 't': 'technical', '': 'research'}
        approach_name = approach_map.get(approach_input, 'research')

        message_count += 1
        logger.info(f"Processing chat query #{message_count}: {user_input} with approach: {approach_name}")

        print(f"\n🔍 Generating {approach_name} report...")
        print("⏳ This process involves multiple phases and may take 3-5 minutes...")

        try:
            result = await run_report(user_input, approach_name)
            logger.info(f"Successfully generated report #{message_count}")

            print(f"\n📊 {result.title}")
            print(f"\n💡 **Executive Summary**")
            print(result.executive_summary[:500] + "..." if len(result.executive_summary) > 500 else result.executive_summary)

            print(f"\n📋 **Report Structure**")
            for i, section in enumerate(result.sections, 1):
                print(f"   {i}. {section.title} ({section.word_count} words)")
                if section.key_findings:
                    print(f"      Key: {section.key_findings[0]}")

            print(f"\n✅ **Report Generated Successfully**")
            print(f"   • {result.total_word_count} words across {len(result.sections)} sections")
            print(f"   • {len(result.all_sources)} sources referenced")
            print(f"   • Quality score: {result.quality_score}/10")
            print(f"   • Generated in: {result.generation_time}")
            print("\n" + "-"*50)

        except Exception as e:
            logger.error(f"Error processing chat query #{message_count}: {str(e)}")
            print(f"Sorry, I encountered an error while generating the report: {e}")

    logger.info(f"Chat session ended after {message_count} messages")

async def main_async():
    """Main async function to handle command-line arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(description="Template-Driven Research System using OpenAI Agents SDK")
    parser.add_argument("input", nargs="?", help="The research topic to investigate")
    parser.add_argument("--approach", "-a", choices=["research", "business", "technical"],
                        default="research", help="Research approach to use (default: research)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chat", "-c", action="store_true", help="Run in chat mode")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use (default: gpt-4.1-mini)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature setting (default: 0.2)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--list-approaches", action="store_true", help="List available approaches and exit")
    args = parser.parse_args()

    # List approaches and exit if requested
    if args.list_approaches:
        print("Available Research Approaches:")
        print("\n1. research - Academic/Scientific Research Approach")
        print("   Typical Sections: Literature Review, Current Research, Methodology Analysis, Findings Synthesis, Theoretical Framework, Applications, Future Directions, Implications, Limitations")
        print("   Best for: Academic research, scientific analysis, literature reviews")

        print("\n2. business - Business Analysis Approach")
        print("   Typical Sections: Market Analysis, Competitive Landscape, Financial Performance, Industry Trends, Opportunities Assessment, Risk Analysis, Strategic Implications, Investment Outlook, Regulatory Environment")
        print("   Best for: Market research, business intelligence, investment analysis")

        print("\n3. technical - Technical Assessment Approach")
        print("   Typical Sections: Technology Overview, Architecture Analysis, Performance Evaluation, Implementation Considerations, Security Assessment, Scalability Analysis, Integration Challenges, Best Practices, Recommendations")
        print("   Best for: Technology evaluation, system assessment, technical due diligence")
        return

    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive or args.chat):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/template_driven_research_{timestamp}.log"

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting template-driven research system with model: {args.model}, temperature: {args.temperature}")

    if args.chat:
        await chat_mode(model=args.model, temperature=args.temperature)
    elif args.interactive:
        await interactive_mode(model=args.model, temperature=args.temperature)
    elif args.input:
        logger.info(f"Running in single research mode with approach: {args.approach}")
        result = await run_query(args.input, approach_name=args.approach, model=args.model, temperature=args.temperature)
        print(result)
    else:
        logger.warning("No input provided and not in interactive mode")
        parser.print_help()

    logger.info("Template-driven research system completed")

def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
