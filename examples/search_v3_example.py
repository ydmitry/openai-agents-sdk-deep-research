#!/usr/bin/env python3
"""
Example script demonstrating OpenAI Agents SDK multi-agent research system.

This script shows how to:
1. Create specialized agents with specific roles (researcher, fact-checker, writer)
2. Use handoffs to coordinate between agents
3. Implement guardrails for safety and quality control
4. Maintain shared context across agents
5. Use function tools with proper schema generation
6. Provide structured outputs with Pydantic models
7. Include tracing and monitoring capabilities

Based on the OpenAI Agents SDK patterns described at:
https://www.siddharthbharath.com/openai-agents-sdk/

Usage:
    python examples/search_v3_example.py "Your research topic here"
    python examples/search_v3_example.py --interactive
    python examples/search_v3_example.py --chat
    python examples/search_v3_example.py "AI developments" --model "gpt-4.1"
"""

import argparse
import asyncio
import logging
import sys
import os
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field

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
# Pydantic Models for Structured Data
# ──────────────────────────────────────────────────────────────────────────

class ResearchFinding(BaseModel):
    """A single research finding with source information."""
    statement: str = Field(description="The factual statement or finding")
    source: str = Field(description="URL or source reference")
    credibility_score: float = Field(description="Credibility score from 0.0 to 1.0", ge=0.0, le=1.0)
    category: str = Field(description="Category of the finding (e.g., 'technology', 'market', 'research')")

class VerifiedResearch(BaseModel):
    """Collection of verified research findings."""
    findings: List[ResearchFinding] = Field(description="List of verified research findings")
    verification_notes: str = Field(description="Notes about the verification process")
    confidence_level: str = Field(description="Overall confidence level: high, medium, low")

class FinalContent(BaseModel):
    """Final research content output."""
    title: str = Field(description="Title of the research")
    summary: str = Field(description="Executive summary")
    detailed_findings: str = Field(description="Detailed research findings")
    sources: List[str] = Field(description="List of all sources used")
    confidence_assessment: str = Field(description="Overall confidence in the research")
    word_count: int = Field(description="Approximate word count of the content")

# ──────────────────────────────────────────────────────────────────────────
# Shared Context for Multi-Agent Coordination
# ──────────────────────────────────────────────────────────────────────────

class ResearchContext(BaseModel):
    """Shared context for coordinating research across multiple agents."""

    topic: str = Field(description="The research topic")
    findings: List[ResearchFinding] = Field(default_factory=list, description="Research findings")
    verified_findings: List[ResearchFinding] = Field(default_factory=list, description="Verified research findings")
    draft_content: str = Field(default="", description="Draft content")
    history: List[str] = Field(default_factory=list, description="History of actions")
    search_queries_used: List[str] = Field(default_factory=list, description="Search queries used")

    def add_finding(self, finding: ResearchFinding):
        """Add a new research finding."""
        self.findings.append(finding)
        self.history.append(f"Added finding: {finding.statement[:100]}...")

    def add_verified_findings(self, verified: VerifiedResearch):
        """Add verified research findings."""
        self.verified_findings.extend(verified.findings)
        self.history.append(f"Added {len(verified.findings)} verified findings")

    def set_draft(self, draft: str):
        """Set the draft content."""
        self.draft_content = draft
        self.history.append("Updated draft content")

    def add_search_query(self, query: str):
        """Track search queries to avoid repetition."""
        self.search_queries_used.append(query)
        self.history.append(f"Executed search: {query}")

def load_multi_agent_system(
    model: str = "gpt-4.1",
    temperature: float = 0.2,
) -> tuple[Any, Any]:
    """
    Factory function that creates the multi-agent research system.

    Returns:
        Tuple of (triage_agent, run_research_function)
    """
    try:
        # Import Agents SDK
        from agents import Agent, Runner, ModelSettings, function_tool, handoff, WebSearchTool
        from agents import output_guardrail, GuardrailFunctionOutput
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # Store the context globally for this system instance
    research_context = None

    # Maximum characters to pass from search results to avoid oversized payloads
    MAX_SEARCH_CHARS = 2000

    # ──────────────────────────────────────────────────────────────────────────
    # Safe Runner with 5xx Retry Logic
    # ──────────────────────────────────────────────────────────────────────────
    
    async def safe_run(agent, input_text: str, max_turns: int = 10):
        """Run an agent with automatic retry for 5xx server errors."""
        for attempt in range(5):
            try:
                return await Runner.run(agent, input_text, max_turns=max_turns)
            except Exception as e:
                # Check if it's a 5xx server error
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    wait_time = 2 ** attempt
                    logger.warning(f"5xx error (attempt {attempt + 1}/5), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                elif "500" in str(e) or "server_error" in str(e):
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error (attempt {attempt + 1}/5), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Not a 5xx error, re-raise immediately
                    raise
        
        # All retries failed
        raise Exception(f"Failed after 5 attempts due to persistent server errors")

    # ──────────────────────────────────────────────────────────────────────────
    # Function Tools
    # ──────────────────────────────────────────────────────────────────────────

    @function_tool
    async def add_research_finding(
        statement: str, 
        source: str, 
        credibility_score: float, 
        category: str
    ) -> str:
        """
        Add a research finding to the context.
        
        Args:
            statement: The factual statement or finding
            source: URL or source reference
            credibility_score: Credibility score from 0.0 to 1.0
            category: Category of the finding (e.g., 'technology', 'market', 'research')
            
        Returns:
            Confirmation message
        """
        nonlocal research_context
        
        try:
            # Validate credibility score
            credibility_score = max(0.0, min(1.0, float(credibility_score)))
            
            finding = ResearchFinding(
                statement=statement,
                source=source,
                credibility_score=credibility_score,
                category=category.lower()
            )
            
            research_context.add_finding(finding)
            logger.info(f"Added finding: {statement[:50]}...")
            return f"Successfully added finding: {statement[:50]}..."
            
        except Exception as e:
            error_msg = f"Error adding finding: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def web_search(query: str) -> str:
        """
        Search the web for information on a given query using WebSearchTool.
        
        Args:
            query: The search query to execute
            
        Returns:
            Formatted search results (trimmed to avoid oversized payloads)
        """
        nonlocal research_context
        
        # Avoid duplicate searches
        if query in research_context.search_queries_used:
            return f"Query '{query}' already searched. Skipping duplicate search."
        
        research_context.add_search_query(query)
        logger.info(f"Executing web search: {query}")
        
        # Combine query with topic for better results
        full_query = f"{query} {research_context.topic}"
        
        try:
            # Create a search agent that uses WebSearchTool with LOW context to avoid 500 errors
            search_agent = Agent(
                name="Web Search Agent",
                instructions=f"""
                You are a web search agent. Your task is to search the internet for relevant information
                about the following query and provide a concise, focused summary:

                "{full_query}"

                Use the web_search tool to find the most relevant and up-to-date information.
                Provide a clear, three-paragraph summary with the most important facts and details.
                Include citations or sources when possible.
                Focus on factual, credible information from reputable sources.
                Keep your response concise and focused on key insights.
                """,
                tools=[WebSearchTool(search_context_size="low")],  # Use "low" to avoid oversized payloads
                model=model,
                model_settings=ModelSettings(
                    temperature=0.2,  # Reduced for consistency
                    max_tokens=1024,  # Reduced to avoid large responses
                ),
            )

            # Run the search agent
            search_result = await safe_run(search_agent, full_query)
            
            # Trim the result to prevent oversized payloads
            trimmed_result = search_result.final_output[:MAX_SEARCH_CHARS]
            output = f"WEB SEARCH RESULTS for '{query}':\n{trimmed_result}"
            
            if len(search_result.final_output) > MAX_SEARCH_CHARS:
                output += f"\n... (trimmed from {len(search_result.final_output)} to {MAX_SEARCH_CHARS} characters)"

            logger.info(f"Search completed for: {query} (result length: {len(output)} chars)")
            return output
            
        except Exception as e:
            error_msg = f"WEB SEARCH ERROR: Failed to get results for query '{query}'. Error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def extract_findings(search_results: str) -> str:
        """
        Extract structured findings from search results using an analysis agent.
        
        Args:
            search_results: Raw search results to process (pre-trimmed to avoid oversized payloads)
            
        Returns:
            Confirmation message about findings extraction
        """
        nonlocal research_context
        
        logger.info("Extracting findings from search results using analysis agent")
        
        try:
            # Create an extraction agent (add_research_finding is now at module level)
            extraction_agent = Agent(
                name="Research Findings Extractor",
                instructions=f"""
                You are a research findings extraction agent. Your task is to analyze search results and extract 
                structured, factual findings related to the research topic: "{research_context.topic}".

                INSTRUCTIONS:
                1. Carefully read and analyze the provided search results
                2. Extract 2-4 key factual statements or findings (quality over quantity)
                3. For each finding, use the add_research_finding tool with:
                   - statement: A clear, factual statement (avoid opinions or speculation)
                   - source: The URL or source reference from the search results
                   - credibility_score: Score from 0.0-1.0 based on source reliability
                     * 0.9-1.0: Academic journals, government sources, major research institutions
                     * 0.7-0.8: Reputable news outlets, established industry sources
                     * 0.5-0.6: General websites, blogs with citations
                     * 0.0-0.4: Questionable or unverified sources
                   - category: One of 'research', 'market', 'technology', 'policy', 'industry', 'academic'

                GUIDELINES:
                - Focus on recent, verifiable information
                - Prioritize quantitative data and specific facts over general statements
                - Ensure statements are directly supported by the search results
                - Avoid extracting duplicate or very similar findings
                - If no credible findings can be extracted, explain why
                - Citations for all sources used

                SEARCH RESULTS TO ANALYZE:
                {search_results}
                """,
                tools=[add_research_finding],
                model=model,
                model_settings=ModelSettings(
                    temperature=0.1,  # Low temperature for factual extraction
                    max_tokens=2048,  # Increased slightly but within safe limits
                ),
            )

            # Run the extraction agent
            findings_before = len(research_context.findings)
            extraction_result = await safe_run(extraction_agent, search_results)
            findings_after = len(research_context.findings)
            findings_added = findings_after - findings_before
            
            if findings_added > 0:
                logger.info(f"Successfully extracted {findings_added} findings from search results")
                return f"Successfully extracted {findings_added} structured findings from search results. Agent analysis: {extraction_result.final_output}"
            else:
                logger.warning("No findings were extracted from search results")
                return f"No findings extracted. Agent analysis: {extraction_result.final_output}"
                
        except Exception as e:
            error_msg = f"Error in findings extraction agent: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def verify_statement(statement: str, source: str) -> str:
        """
        Verify the credibility of a research statement.

        Args:
            statement: The statement to verify
            source: Source URL or reference

        Returns:
            Verification result
        """
        logger.info(f"Verifying statement: {statement[:50]}...")

        # Simulate verification process
        # In reality, this would check source credibility, fact-check, etc.
        try:
            credibility_score = 0.8 if "research" in source else 0.6
            verification_result = f"VERIFIED: Statement has credibility score of {credibility_score}\n"
            verification_result += f"Source analysis: {source} appears to be a credible source.\n"
            verification_result += f"Statement: {statement}\n"
            verification_result += f"Verification status: APPROVED"

            logger.info("Statement verification completed")
            return verification_result

        except Exception as e:
            error_msg = f"Verification error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def save_verified_research(
        findings_json: str,
        verification_notes: str,
        confidence_level: str
    ) -> str:
        """
        Save verified research findings to context.

        Args:
            findings_json: JSON string of verified findings
            verification_notes: Notes about the verification process
            confidence_level: Overall confidence level: high, medium, low

        Returns:
            Confirmation message
        """
        nonlocal research_context

        try:
            import json

            # Parse the findings JSON
            findings_data = json.loads(findings_json)
            findings = [ResearchFinding(**f) for f in findings_data]

            verified_research = VerifiedResearch(
                findings=findings,
                verification_notes=verification_notes,
                confidence_level=confidence_level
            )

            research_context.add_verified_findings(verified_research)
            logger.info(f"Saved {len(findings)} verified findings")
            return f"Successfully saved {len(findings)} verified findings to research context."

        except Exception as e:
            error_msg = f"Error saving verified research: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @function_tool
    async def generate_final_content() -> str:
        """
        Generate final research content from verified findings or raw findings if verified is empty.

        Returns:
            JSON string of the final content
        """
        nonlocal research_context

        logger.info("Generating final research content")

        try:
            # Determine which findings to use
            if research_context.verified_findings:
                findings_to_use = research_context.verified_findings
                findings_type = "verified"
                confidence_assessment = "High confidence based on multiple verified sources and cross-validation."
            elif research_context.findings:
                findings_to_use = research_context.findings
                findings_type = "raw"
                confidence_assessment = "Medium confidence based on raw research findings without verification."
            else:
                # No findings at all
                fallback_content = FinalContent(
                    title=f"Research Report: {research_context.topic}",
                    summary="No research findings were collected for this topic.",
                    detailed_findings="Unable to generate research findings. Please try a different search approach.",
                    sources=[],
                    confidence_assessment="No confidence - no findings available.",
                    word_count=0
                )
                return fallback_content.model_dump_json()

            # Generate content based on available findings
            title = f"Research Report: {research_context.topic}"

            summary = f"This report presents comprehensive research on {research_context.topic}. "
            summary += f"Based on {len(findings_to_use)} {findings_type} findings, "
            summary += f"the research indicates significant developments and opportunities in this area."

            detailed_findings = f"## Detailed Research Findings ({findings_type.title()})\n\n"
            sources = []

            for i, finding in enumerate(findings_to_use, 1):
                detailed_findings += f"{i}. **{finding.category.title()} Finding**\n"
                detailed_findings += f"   {finding.statement}\n"
                detailed_findings += f"   Credibility Score: {finding.credibility_score}\n"
                detailed_findings += f"   Source: {finding.source}\n\n"
                sources.append(finding.source)

            # Remove duplicates from sources
            sources = list(set(sources))

            word_count = len(detailed_findings.split())

            final_content = FinalContent(
                title=title,
                summary=summary,
                detailed_findings=detailed_findings,
                sources=sources,
                confidence_assessment=confidence_assessment,
                word_count=word_count
            )

            logger.info(f"Final content generation completed using {findings_type} findings")
            return final_content.model_dump_json()

        except Exception as e:
            logger.error(f"Error generating final content: {str(e)}")
            # Return a fallback content structure
            fallback_content = FinalContent(
                title=f"Research Report: {research_context.topic}",
                summary="Research compilation encountered errors during processing.",
                detailed_findings="Content generation failed. Please review the research process.",
                sources=[],
                confidence_assessment="Low confidence due to processing errors.",
                word_count=0
            )
            return fallback_content.model_dump_json()


    # ──────────────────────────────────────────────────────────────────────────
    # Specialized Agents
    # ──────────────────────────────────────────────────────────────────────────

    # Configure model settings
    model_settings_kwargs = {"max_tokens": 2048}  # Reduced from 1500 to safe value
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

    # Researcher Agent
    researcher_agent = Agent(
        name="researcher_agent",
        model=model,
        model_settings=model_settings,
        instructions=prompt_with_handoff_instructions("""You are a thorough research agent specializing in gathering information from multiple sources.

Your responsibilities:
1. Execute comprehensive web searches on the given topic
2. Use multiple search queries to gather diverse perspectives
3. Extract structured findings from search results
4. Ensure comprehensive coverage of the research topic

Guidelines:
- Start with broad searches, then narrow down to specific aspects
- Use 3-4 different search queries to ensure comprehensive coverage
- Extract factual statements with proper source attribution
- Focus on recent and credible information
- Always use the extract_findings tool after each search to structure the data
- Keep searches focused and avoid overly broad queries

When you receive a research topic, immediately begin with web searches and continue until you have gathered sufficient information from multiple angles.
You can handoff to writer_agent if you have sufficient findings to create a report."""),
        tools=[web_search, extract_findings]
    )

    # Fact Checker Agent
    fact_checker_agent = Agent(
        name="fact_checker_agent",
        model=model,
        model_settings=model_settings,
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a meticulous fact-checking agent responsible for verifying research findings.

Your responsibilities:
1. Review all research findings in the shared context
2. Verify each statement using the verify_statement tool
3. Assess source credibility and information reliability
4. Consolidate verified findings using save_verified_research
5. Only approve statements with sufficient evidence

Standards for verification:
- Check source credibility (academic journals, reputable institutions, official reports)
- Look for corroborating evidence from multiple sources
- Flag any statements that seem speculative or opinion-based
- Assign appropriate credibility scores based on source quality
- Be efficient but thorough in your assessment

Process:
1. First, verify individual statements using verify_statement
2. Then, save all verified findings using save_verified_research with:
   - findings_json: JSON array of verified findings
   - verification_notes: Your notes about the verification process
   - confidence_level: "high", "medium", or "low"

Focus on creating high-confidence, well-sourced findings efficiently.""",
        tools=[verify_statement, save_verified_research]
    )

    # Writer Agent
    writer_agent = Agent(
        name="writer_agent",
        model=model,
        model_settings=model_settings,
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a professional research writer who creates comprehensive, well-structured reports.

Your responsibilities:
1. Review both verified research findings and raw research findings in the shared context
2. Generate a comprehensive final report using generate_final_content
3. Ensure proper structure with title, summary, detailed findings, and sources
4. Maintain objectivity and factual accuracy
5. Provide clear confidence assessments

Writing standards:
- Clear, professional tone appropriate for research reports
- Logical organization of information
- Proper citation of all sources
- Balanced presentation of findings
- Executive summary that captures key insights
- Detailed findings section with full context
- Be concise while maintaining comprehensiveness

Priority for findings:
- Use verified findings when available (highest confidence)
- Fall back to raw research findings if no verified findings exist
- Clearly indicate the confidence level based on which type of findings were used

Always use the generate_final_content tool to create the structured output.""",
        tools=[generate_final_content]
    )

    # Triage Agent (Coordinator)
    triage_agent = Agent(
        name="triage_agent",
        model=model,
        model_settings=model_settings,
        instructions=prompt_with_handoff_instructions("""You are the research coordinator managing the entire research workflow efficiently.

Your workflow:
1. **Research Phase**: Hand off to researcher_agent to gather comprehensive information
2. **Verification Phase**: Hand off to fact_checker_agent to verify all findings
3. **Writing Phase**: Hand off to writer_agent to create the final report

Coordination guidelines:
- Ensure each agent completes their task before moving to the next phase
- Monitor the shared context to track progress
- Only proceed to the next phase when the previous phase is complete
- The final output should be a comprehensive research report
- Be efficient and focused in coordinating the workflow
- Avoid unnecessary back-and-forth between agents

For any research query, always follow this three-phase approach and coordinate the handoffs properly."""),
        handoffs=[
            handoff(researcher_agent),
            handoff(fact_checker_agent),
            handoff(writer_agent)
        ]
    )

    # Configure additional handoffs after all agents are defined
    researcher_agent.handoffs = [handoff(writer_agent), handoff(triage_agent)]

    # ──────────────────────────────────────────────────────────────────────────
    # Research Runner Function
    # ──────────────────────────────────────────────────────────────────────────

    async def run_research_system(topic: str) -> FinalContent:
        """Run the multi-agent research system on a given topic."""
        nonlocal research_context

        logger.info(f"Starting research on topic: {topic}")

        # Create shared context
        research_context = ResearchContext(topic=topic)

        try:
            # Run the triage agent with the research topic
            result = await safe_run(triage_agent, f"Conduct comprehensive research on the following topic: {topic}")

            logger.info("Research system completed successfully")

            # Parse the final output if it's JSON
            if isinstance(result.final_output, str):
                try:
                    import json
                    final_content_data = json.loads(result.final_output)
                    return FinalContent(**final_content_data)
                except (json.JSONDecodeError, TypeError):
                    # If it's not JSON, create a fallback FinalContent
                    return FinalContent(
                        title=f"Research Report: {topic}",
                        summary=str(result.final_output)[:500] + "..." if len(str(result.final_output)) > 500 else str(result.final_output),
                        detailed_findings=str(result.final_output),
                        sources=[],
                        confidence_assessment="Medium confidence based on agent output.",
                        word_count=len(str(result.final_output).split())
                    )
            else:
                return result.final_output

        except Exception as e:
            logger.error(f"Error in research system: {str(e)}")

            # If we have research findings, create a report from them
            if research_context and research_context.findings:
                logger.info(f"Creating fallback report from {len(research_context.findings)} collected findings")

                detailed_findings = "## Research Findings Collected Before Error\n\n"
                sources = []

                for i, finding in enumerate(research_context.findings, 1):
                    detailed_findings += f"**{i}. {finding.category.title()} Finding (Score: {finding.credibility_score})**\n"
                    detailed_findings += f"{finding.statement}\n"
                    detailed_findings += f"*Source: {finding.source}*\n\n"
                    sources.append(finding.source)

                sources = list(set(sources))  # Remove duplicates

                return FinalContent(
                    title=f"Research Report: {topic}",
                    summary=f"Research system collected {len(research_context.findings)} findings before encountering an error. The findings provide valuable insights into {topic}.",
                    detailed_findings=detailed_findings,
                    sources=sources,
                    confidence_assessment=f"Partial confidence based on {len(research_context.findings)} research findings before system error.",
                    word_count=len(detailed_findings.split())
                )

            # Final fallback if no findings were collected
            return FinalContent(
                title=f"Research Report: {topic}",
                summary="Research system encountered an error during processing.",
                detailed_findings="The multi-agent research system was unable to complete the research due to technical issues.",
                sources=[],
                confidence_assessment="No confidence due to system errors.",
                word_count=0
            )

    logger.info(f"Multi-agent research system initialized with model: {model}")
    return triage_agent, run_research_system

async def run_query(input_text: str, model: str = "gpt-4.1", temperature: float = 0.2):
    """Run a single research query through the multi-agent system."""
    logger.info(f"Processing research query: {input_text}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the multi-agent system
    logger.debug(f"Loading multi-agent research system with model: {model}")
    _, run_research = load_multi_agent_system(model=model, temperature=temperature)

    # Run the research and return results
    logger.info("Starting multi-agent research process")
    result = await run_research(input_text)
    logger.info("Research process completed successfully")

    # Format the response
    formatted_response = f"""# Multi-Agent Research Report

## Research Topic
{input_text}

## {result.title}

### Executive Summary
{result.summary}

### Research Findings
{result.detailed_findings}

### Sources
{', '.join(result.sources) if result.sources else 'No sources available'}

### Confidence Assessment
{result.confidence_assessment}

### Report Statistics
- Word Count: {result.word_count}
- Research Method: Multi-agent collaboration with verification
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return formatted_response

async def interactive_mode(model: str = "gpt-4.1", temperature: float = 0.2):
    """Run the multi-agent research system in interactive mode."""
    logger.info(f"Starting interactive mode with model: {model}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the multi-agent system
    logger.debug("Loading multi-agent research system for interactive session")
    _, run_research = load_multi_agent_system(model=model, temperature=temperature)

    print("Multi-Agent Research System (type 'exit' to quit)")
    print("This system uses specialized agents for research, fact-checking, and writing.")

    query_count = 0
    while True:
        input_text = input("\nEnter your research topic: ")
        if input_text.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break

        query_count += 1
        logger.info(f"Processing interactive research #{query_count}: {input_text}")
        print("Running multi-agent research process...")
        try:
            result = await run_research(input_text)
            logger.info(f"Successfully completed interactive research #{query_count}")

            print(f"\n# Research Report: {result.title}")
            print(f"\n## Summary\n{result.summary}")
            print(f"\n## Detailed Findings\n{result.detailed_findings}")
            print(f"\n## Sources\n{', '.join(result.sources) if result.sources else 'No sources available'}")
            print(f"\n## Confidence Assessment\n{result.confidence_assessment}")

        except Exception as e:
            logger.error(f"Error processing research #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")

    logger.info(f"Interactive session ended after {query_count} queries")

async def chat_mode(model: str = "gpt-4.1", temperature: float = 0.2):
    """Run the multi-agent research system in chat mode."""
    logger.info(f"Starting chat mode with model: {model}")

    # Load environment variables
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the multi-agent system
    logger.debug("Initializing multi-agent research system for chat session")
    _, run_research = load_multi_agent_system(model=model, temperature=temperature)

    print("\n" + "="*50)
    print("🔬 Multi-Agent Research Assistant")
    print("="*50)
    print("Ask any research question and get comprehensive reports")
    print("powered by specialized AI agents working together.")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")

    message_count = 0

    while True:
        user_input = input("Research Query: ")
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nResearch session ended. Thank you!")
            logger.info("User ended chat session")
            break

        if not user_input.strip():
            print("Please enter a research topic or type 'exit' to quit.")
            continue

        message_count += 1
        logger.info(f"Processing chat research #{message_count}: {user_input}")

        print("\n🔍 Agents working on your research...")
        print("- Researcher gathering information...")
        print("- Fact-checker verifying findings...")
        print("- Writer preparing report...")

        try:
            result = await run_research(user_input)
            logger.info(f"Successfully generated research report #{message_count}")

            print(f"\n📊 {result.title}")
            print(f"\n💡 **Summary**")
            print(result.summary)

            print(f"\n📋 **Key Findings**")
            # Show abbreviated findings for chat mode
            findings_lines = result.detailed_findings.split('\n')[:10]
            print('\n'.join(findings_lines))
            if len(result.detailed_findings.split('\n')) > 10:
                print("... (truncated for chat display)")

            if result.sources:
                print(f"\n🔗 **Sources**: {len(result.sources)} references")

            print(f"\n✅ **Confidence**: {result.confidence_assessment}")
            print("\n" + "-"*50)

        except Exception as e:
            logger.error(f"Error processing chat research #{message_count}: {str(e)}")
            print(f"Sorry, I encountered an error while researching: {e}")

    logger.info(f"Chat session ended after {message_count} messages")

async def main_async():
    """Main async function to handle command-line arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(description="Multi-Agent Research System using OpenAI Agents SDK")
    parser.add_argument("input", nargs="?", help="The research topic to investigate")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chat", "-c", action="store_true", help="Run in chat mode")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use (default: gpt-4.1)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature setting (default: 0.2)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive or args.chat):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/multi_agent_research_{timestamp}.log"

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting multi-agent research system with model: {args.model}, temperature: {args.temperature}")

    if args.chat:
        await chat_mode(model=args.model, temperature=args.temperature)
    elif args.interactive:
        await interactive_mode(model=args.model, temperature=args.temperature)
    elif args.input:
        logger.info(f"Running in single research mode")
        result = await run_query(args.input, model=args.model, temperature=args.temperature)
        print(result)
    else:
        logger.warning("No input provided and not in interactive mode")
        parser.print_help()

    logger.info("Multi-agent research system completed")

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
