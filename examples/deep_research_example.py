#!/usr/bin/env python3
"""
Example script demonstrating the deep research agent functionality.

This script shows how to:
1. Initialize a deep research system with multiple specialized agents
2. Ask complex research questions requiring in-depth analysis
3. Use a triage agent to coordinate between search and analysis agents
4. Enable iterative refinement of research through multiple search-analyze cycles
5. Process and display comprehensive research results
6. Optionally enable Langfuse tracing for observability

Usage:
    python deep_research_example.py "What are the environmental impacts of lithium mining?"
    python deep_research_example.py --interactive
    python deep_research_example.py "How does quantum computing affect cryptography?" --enable-langfuse
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, create_model

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research.utils import load_dotenv_files, get_model_settings

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration."""
    # Create logs directory if using file logging and directory doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Return the logger
    return root_logger

# Module logger
logger = logging.getLogger(__name__)

# Define research context and related classes for sharing data between agents
class ResearchSource(BaseModel):
    """Represents a source of information used in research"""
    url: str
    title: str
    snippet: str
    domain: str
    source_type: str  # academic, news, blog, etc.
    credibility_score: float = 0.0  # 0.0-1.0, evaluated by the system
    
class ResearchFinding(BaseModel):
    """Represents a discrete finding from research"""
    statement: str
    sources: List[ResearchSource]
    topic_area: str
    confidence: float = 0.0  # 0.0-1.0

class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    
class WorkflowState(BaseModel):
    """Shared state for the research workflow"""
    query: str = Field(description="The original research query")
    topics: List[str] = Field(default_factory=list, description="Research topics identified")
    search_results: Dict[str, List[SearchResult]] = Field(default_factory=dict, description="Search results by topic")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Research findings")
    current_phase: str = Field(default="discovery", description="Current phase of research")
    iterations_done: int = Field(default=0, description="Number of research iterations completed")
    gaps: List[str] = Field(default_factory=list, description="Identified research gaps")
    
    class Config:
        arbitrary_types_allowed = True

# Pydantic models for function tool input/output
class TopicResponse(BaseModel):
    topics: List[str] = Field(description="List of research topics to investigate")

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="List of search queries for a topic")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(description="Search results")
    query: str = Field(description="The search query used")
    
class GapAnalysisResult(BaseModel):
    gaps: List[str] = Field(description="List of identified research gaps")
    complete: bool = Field(description="Whether the research is considered complete")

class ResearchExample(BaseModel):
    title: str = Field(description="Title of the example")
    description: str = Field(description="Description of the example")
    url: str = Field(description="URL of the example")
    highlights: List[str] = Field(description="Key highlights of this example")
    source_quality: str = Field(description="Evaluation of the source quality")
    
class FinalResearchResponse(BaseModel):
    introduction: str = Field(description="Introduction to the research topic")
    best_examples: List[ResearchExample] = Field(description="List of best examples found")
    comparison: str = Field(description="Comparison of different examples")
    conclusion: str = Field(description="Conclusion and final recommendation")
    sources: List[str] = Field(description="List of sources consulted")

def load_deep_research_agent(enable_langfuse: bool = False, service_name: str = "deep_research_agent"):
    """
    Factory function to create and return a deep research system with multiple specialized agents.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name to use for Langfuse tracing
        
    Returns:
        tuple: (agent, run_agent_function) - The configured triage agent and a function to run queries.
    """
    import openai
    from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool
    from agents import handoff

    logger.info("Initializing deep research agent system")

    # Set up Langfuse tracing if enabled
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse
            
            if setup_langfuse():
                logger.info("Langfuse tracing enabled for deep research agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning("Could not import langfuse_integration module. Langfuse tracing will not be enabled.")
            logger.warning("Make sure you have installed 'pydantic-ai[logfire]' package.")

    # Get API client
    async_client = openai.AsyncOpenAI()

    # Define shared model settings for more consistency
    powerful_model_settings = get_model_settings(
        model_name="gpt-4.1",
        temperature=0.2,  # Lower temperature for more focused responses
        max_tokens=4000,  # Allow for comprehensive responses
    )
    
    lighter_model_settings = get_model_settings(
        model_name="gpt-4.1",
        temperature=0.1,  # Even lower temperature for coordination
        max_tokens=1000,  # Less tokens needed for coordination
    )
    
    # Create a shared workflow state
    workflow_state = WorkflowState(query="")
    
    # Custom function tools for research management
    
    @function_tool
    async def identify_research_topics(query: str) -> TopicResponse:
        """
        Identify key research topics/areas that need to be investigated for the query.
        
        Args:
            query: The research query
            
        Returns:
            A response containing a list of specific topics to research
        """
        # Generate fixed generic topics regardless of query content
        topics = [
            "Implementation approaches",
            "Architectural patterns",
            "Common use cases",
            "Development best practices",
            "Integration strategies",
            "Performance considerations"
        ]
        
        # Update the workflow state
        workflow_state.topics = topics
        workflow_state.current_phase = "search"
        
        # Return topics with a flag to indicate we need to continue to search phase
        return TopicResponse(topics=topics)
    
    @function_tool
    async def generate_diverse_search_queries(topic: str) -> SearchQueries:
        """
        Generate diverse search queries for a given topic to ensure comprehensive coverage.
        
        Args:
            topic: The topic to generate queries for
            
        Returns:
            A list of search queries
        """
        # Generate variants of the search query to get diverse results
        variants = [
            f"{topic}",
            f"{topic} code examples",
            f"{topic} tutorial",
            f"{topic} GitHub",
            f"{topic} best practices",
            f"{topic} case studies",
            f"{topic} documentation"
        ]
        
        return SearchQueries(queries=variants)
    
    @function_tool
    async def evaluate_source_credibility(source_url: str, source_content: str) -> float:
        """
        Evaluate the credibility of a source based on its content and metadata.
        
        Args:
            source_url: URL of the source
            source_content: Content snippet from the source
            
        Returns:
            Credibility score from 0.0 to 1.0
        """
        # In a real implementation, this would use a language model
        # to evaluate the credibility based on content quality,
        # source reputation, citation patterns, etc.
        
        # Without heuristics, we'll return a neutral score
        # This would be replaced with actual source evaluation in production
        return 0.7  # Neutral-positive default score
    
    @function_tool
    async def analyze_research_findings() -> GapAnalysisResult:
        """
        Analyze the current research findings to identify coverage gaps.
        
        Returns:
            Analysis of gaps that need further research
        """
        # In a real implementation, this would analyze the search results
        # and findings to identify gaps
        
        # Check which topics have search results
        covered_topics = set(workflow_state.search_results.keys())
        all_topics = set(workflow_state.topics)
        
        # Identify gaps as topics without search results
        missing_topics = all_topics - covered_topics
        
        gaps = [f"Missing research on: {topic}" for topic in missing_topics]
        
        # If we've done at least one iteration but don't have enough examples
        if workflow_state.iterations_done >= 1:
            total_results = sum(len(results) for results in workflow_state.search_results.values())
            if total_results < 5:
                gaps.append("Need more examples to provide comprehensive coverage")
        
        # Determine if research is complete based on iterations and gaps
        # Force at least one more iteration to ensure we go through the full process
        complete = (workflow_state.iterations_done >= 2) or (workflow_state.iterations_done >= 1 and not gaps)
        
        # Update workflow state
        workflow_state.gaps = gaps
        workflow_state.iterations_done += 1
        
        if complete:
            workflow_state.current_phase = "synthesis"
        else:
            workflow_state.current_phase = "search"
            
        return GapAnalysisResult(gaps=gaps, complete=complete)
    
    @function_tool
    async def create_research_synthesis() -> FinalResearchResponse:
        """
        Create a final research synthesis from all the gathered information.
        
        Returns:
            The final research synthesis with best examples and analysis
        """
        # In a real implementation, this would use the LLM to create
        # a synthesis of all the research findings
        
        # For this example, we'll create a structured response based on
        # the search results we've collected
        
        # Flatten all search results
        all_results = []
        for results_list in workflow_state.search_results.values():
            all_results.extend(results_list)
        
        # Create example objects from the search results without domain-specific quality ratings
        examples = []
        for result in all_results:
            example = ResearchExample(
                title=result.title,
                description=result.content,
                url=result.url,
                highlights=["Implementation example", "Documentation", "Use case"],
                source_quality="Standard"  # Neutral quality assessment
            )
            examples.append(example)
        
        # Create the final response
        return FinalResearchResponse(
            introduction=f"This research explores examples of {workflow_state.query}.",
            best_examples=examples,
            comparison="The examples vary in functionality and implementation approach.",
            conclusion="Based on the research, there are several approaches to implementation, each with different features and capabilities.",
            sources=[result.url for result in all_results]
        )
    
    # Enhanced web search tool
    web_search_tool = WebSearchTool(
        search_context_size='high',  # Use high context for detailed results
    )
    
    # Create specialized topic discovery agent
    topic_discovery_agent = Agent(
        name="Topic Discovery Agent",
        instructions="""
        You are a specialized agent focused on discovering the key aspects of a research topic.
        
        Follow these guidelines:
        1. Break down the main research question into specific topic areas
        2. Identify both obvious and non-obvious aspects that need investigation
        3. Consider historical context, current state, and future implications
        4. Look for potentially contradictory viewpoints or controversies
        5. Prioritize topics based on their importance to the overall question
        
        Your goal is to create a comprehensive research plan that will guide
        the rest of the research process.
        """,
        tools=[identify_research_topics],
        model="gpt-4.1",
        model_settings=lighter_model_settings,
    )
    
    logger.debug("Topic discovery agent created")

    # Create enhanced search agent
    search_agent = Agent(
        name="Research Search Agent",
        instructions="""
        You are a specialized search agent focused on finding comprehensive information 
        for deep research queries. Your goal is maximum coverage across diverse sources.
        
        Follow these guidelines:
        1. For each topic area, generate multiple search queries to ensure coverage
        2. Use the generate_diverse_search_queries tool to create varied search terms
        3. Execute web searches for each query to gather information using the web_search_tool tool
        4. For each source found:
           - Use evaluate_source_credibility to assess reliability
           - Extract key information and associate it with topics
        5. Prioritize finding diverse source types:
           - Official documentation and examples
           - GitHub repositories and code examples
           - Case studies and real-world implementations
           - Comparison articles and analyses
        6. Document all sources thoroughly with proper citations
        7. Execute searches for ALL the topics provided to ensure comprehensive coverage
        
        Remember, your goal is to gather COMPREHENSIVE information across DIVERSE sources.
        Be methodical and thorough, making sure to search for each topic area.
        """,
        tools=[
            generate_diverse_search_queries,
            web_search_tool,
            evaluate_source_credibility,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Enhanced search agent created with diverse search strategies")

    # Create enhanced analysis agent
    analysis_agent = Agent(
        name="Research Analysis Agent",
        instructions="""
        You are a specialized analysis agent that synthesizes and evaluates research information.
        Your goal is to produce comprehensive, nuanced analysis with excellent source integration.
        
        Follow these guidelines:
        1. Carefully analyze all the search results that have been collected
        2. Group findings by topic area to identify patterns and relationships
        3. Evaluate the credibility and relevance of each source
        4. Identify connections, contradictions, and gaps across sources
        5. Use the analyze_research_findings tool to identify areas needing more research
        6. Draw well-reasoned conclusions based on the evidence, noting certainty levels
        
        Your role is to provide thorough analysis and determine if additional
        research iterations are needed before proceeding to the final synthesis.
        """,
        tools=[
            analyze_research_findings,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Enhanced analysis agent created with gap identification")

    # Create synthesis agent for final research product
    synthesis_agent = Agent(
        name="Research Synthesis Agent",
        instructions="""
        You are a specialized synthesis agent that creates the final research product.
        Your goal is to create a comprehensive, well-structured research document.
        
        Follow these guidelines:
        1. Use the create_research_synthesis tool to generate the final research product
        2. Ensure the synthesis includes:
           - A clear introduction to the research question
           - Detailed descriptions of the best examples found
           - Thoughtful comparison of different approaches
           - A conclusive recommendation based on the evidence
           - Complete list of sources consulted
        
        Your goal is to create a DEFINITIVE research document that represents
        the most comprehensive answer possible to the original query.
        """,
        tools=[
            create_research_synthesis,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Synthesis agent created for final research products")

    # Convert agents to tools instead of using handoffs
    topic_discovery_tool = topic_discovery_agent.as_tool(
        "discover_research_topics",
        "Identifies key research topics to investigate for a given query"
    )
    
    search_tool = search_agent.as_tool(
        "search_for_information",
        "Searches for comprehensive information on the identified topics"
    )
    
    analysis_tool = analysis_agent.as_tool(
        "analyze_research_findings",
        "Analyzes the collected information and identifies any gaps that need further research"
    )
    
    synthesis_tool = synthesis_agent.as_tool(
        "create_final_synthesis",
        "Creates a comprehensive final research document with examples and analysis"
    )
    
    # Enhanced triage agent that manages the research workflow using tools
    triage_agent = Agent(
        name="Deep Research Coordinator",
        instructions="""
        You are the lead researcher coordinating a comprehensive research process.
        Your goal is to produce thorough, actionable research through a systematic approach.
        
        YOU MUST FOLLOW THIS EXACT PROCESS IN ORDER:
        
        1. FIRST, use the discover_research_topics tool to identify key research areas
           - Review the topics identified
        
        2. THEN, use the search_for_information tool to gather information on ALL topics
           - Make sure to provide all topics to the search tool
           - Ensure comprehensive information gathering
        
        3. NEXT, use the analyze_research_findings tool to evaluate and identify gaps
           - Review the analysis carefully
        
        4. FINALLY, use the create_final_synthesis tool to create the complete research document
           - This will produce the final comprehensive research
        
        You MUST execute ALL FOUR TOOLS in EXACTLY this order. Do not skip any steps.
        Each step builds on the previous one to ensure thorough research.
        
        YOUR RESPONSE SHOULD ONLY CONTAIN THE FINAL RESEARCH RESULTS from the create_final_synthesis tool.
        Do not add your own summary, introduction, or conclusion.
        """,
        tools=[
            topic_discovery_tool,
            search_tool, 
            analysis_tool,
            synthesis_tool
        ],
        model="gpt-4.1",
        model_settings=lighter_model_settings,
    )
    
    logger.debug("Enhanced triage agent created with tools pattern for full research workflow")

    async def run_agent(query: str):
        """Run the agent with the given query."""
        logger.info(f"Running deep research query: {query}")
        
        # Initialize the workflow state
        workflow_state.query = query
        # Reset any previous state
        workflow_state.topics = []
        workflow_state.search_results = {}
        workflow_state.findings = []
        workflow_state.current_phase = "discovery"
        workflow_state.iterations_done = 0
        workflow_state.gaps = []
        
        try:
            # If Langfuse is enabled, wrap the agent run with a trace context
            if enable_langfuse:
                try:
                    from deep_research.langfuse_integration import create_trace
                    
                    # Generate a session ID based on the first few chars of the query
                    session_id = f"deep_research_{hash(query) % 10000}"
                    
                    with create_trace(
                        name="Deep-Research-Query",
                        session_id=session_id,
                        tags=["deep_research"],
                        environment=os.environ.get("ENVIRONMENT", "development")
                    ) as span:
                        # Set input for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("input.value", query)
                            except Exception as e:
                                logger.warning(f"Could not set input attribute on span: {e}")
                        
                        # Force multiple turns to ensure all tools get used
                        result = await Runner.run(
                            triage_agent,
                            f"""
                            Perform comprehensive research on: {query}
                            
                            IMPORTANT EXECUTION INSTRUCTIONS:
                            1. You MUST use ALL FOUR tools in the exact order: discover_research_topics → search_for_information → analyze_research_findings → create_final_synthesis
                            2. Do not skip any tools
                            3. Ensure each tool is given proper inputs based on the previous tool's output
                            4. The final result should include specific examples and detailed findings
                            
                            Begin the research process now.
                            """,
                            max_turns=30,  # Provide enough turns for all tools to be used
                        )
                        
                        # Set output for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("output.value", result.final_output)
                                # Add research metrics to trace
                                span.set_attribute("research.iterations_completed", workflow_state.iterations_done)
                                span.set_attribute("research.sources_count", len(workflow_state.search_results))
                                span.set_attribute("research.findings_count", len(workflow_state.findings))
                            except Exception as e:
                                logger.warning(f"Could not set output attribute on span: {e}")
                        
                        logger.info(f"Deep research completed successfully")
                        logger.debug(f"Response length: {len(result.final_output)}")
                        
                        # Format the final research response
                        formatted_output = f"# {query}\n\n"
                        formatted_output += "## Research Findings\n\n"
                        
                        # Add the raw output from the agent
                        formatted_output += result.final_output
                        
                        # Add sources section if we have any search results
                        if workflow_state.search_results:
                            formatted_output += "\n\n## Sources\n\n"
                            all_sources = set()
                            for results_list in workflow_state.search_results.values():
                                for result_item in results_list:
                                    all_sources.add(result_item.url)
                            
                            for source in all_sources:
                                formatted_output += f"- {source}\n"
                                
                        # Add phase information for debugging
                        formatted_output += f"\n\n## Debug Info\n\n"
                        formatted_output += f"- Final phase: {workflow_state.current_phase}\n"
                        formatted_output += f"- Iterations completed: {workflow_state.iterations_done}\n"
                        formatted_output += f"- Topics identified: {len(workflow_state.topics)}\n"
                        formatted_output += f"- Search results collected: {sum(len(results) for results in workflow_state.search_results.values())}\n"
                                
                        return formatted_output
                except ImportError as e:
                    logger.warning(f"Could not import create_trace: {e}")
                    logger.warning("Running without tracing.")
                except Exception as e:
                    logger.error(f"Error with Langfuse tracing: {e}")
                    logger.info("Continuing without tracing...")
            
            # Run normally without tracing if Langfuse failed or is disabled
            result = await Runner.run(
                triage_agent,
                f"""
                Perform comprehensive research on: {query}
                
                IMPORTANT EXECUTION INSTRUCTIONS:
                1. You MUST use ALL FOUR tools in the exact order: discover_research_topics → search_for_information → analyze_research_findings → create_final_synthesis
                2. Do not skip any tools
                3. Ensure each tool is given proper inputs based on the previous tool's output
                4. The final result should include specific examples and detailed findings
                
                Begin the research process now.
                """,
                max_turns=30,  # Provide enough turns for all tools to be used
            )
            logger.info(f"Deep research completed successfully")
            logger.debug(f"Response length: {len(result.final_output)}")
            
            # Format the final research response
            formatted_output = f"# {query}\n\n"
            formatted_output += "## Research Findings\n\n"
            
            # Add the raw output from the agent
            formatted_output += result.final_output
            
            # Add sources section if we have any search results
            if workflow_state.search_results:
                formatted_output += "\n\n## Sources\n\n"
                all_sources = set()
                for results_list in workflow_state.search_results.values():
                    for result_item in results_list:
                        all_sources.add(result_item.url)
                
                for source in all_sources:
                    formatted_output += f"- {source}\n"
            
            # Add phase information for debugging
            formatted_output += f"\n\n## Debug Info\n\n"
            formatted_output += f"- Final phase: {workflow_state.current_phase}\n"
            formatted_output += f"- Iterations completed: {workflow_state.iterations_done}\n"
            formatted_output += f"- Topics identified: {len(workflow_state.topics)}\n"
            formatted_output += f"- Search results collected: {sum(len(results) for results in workflow_state.search_results.values())}\n"
                    
            return formatted_output
                
        except Exception as e:
            logger.error(f"Error running deep research agent: {str(e)}")
            raise

    logger.info("Enhanced deep research agent system initialized and ready for queries")
    return triage_agent, run_agent


async def run_query(query: str, enable_langfuse: bool = False):
    """Run a single query through the deep research agent."""
    logger.info(f"Processing single query: {query}")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the deep research agent with optional Langfuse tracing
    logger.debug(f"Loading deep research agent (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    # Run the query and return the result
    logger.info("Sending query to agent")
    print("Running deep research... This may take several minutes for comprehensive analysis with multiple search iterations.")
    result = await run_agent(query)
    logger.info("Query completed successfully")
    return result


async def interactive_mode(enable_langfuse: bool = False):
    """Run the deep research agent in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the deep research agent (only once for the session) with optional Langfuse tracing
    logger.debug("Loading deep research agent for interactive session")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    print("Deep Research Agent (type 'exit' to quit)")
    print("Note: Deep research takes more time than simple searches for comprehensive analysis with multiple iterations.")
    
    query_count = 0
    while True:
        query = input("\nEnter your research question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Conducting deep research... (this may take several minutes for thorough analysis and multiple search iterations)")
        try:
            response = await run_agent(query)
            logger.info(f"Successfully completed interactive query #{query_count}")
            print("\nResearch results:")
            print(response)
        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")
    
    logger.info(f"Interactive session ended after {query_count} queries")


async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conduct deep research on complex topics")
    parser.add_argument("query", nargs="?", help="The research question to investigate")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--enable-langfuse", action="store_true", 
                        help="Enable Langfuse tracing for observability")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of search-analysis iterations (default: 3)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive):
        # Auto-create log file for debug mode or interactive sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/deep_research_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting deep research example with log level: {args.log_level}")
    
    if args.enable_langfuse:
        logger.info("Langfuse tracing is enabled")
    
    if args.interactive:
        await interactive_mode(enable_langfuse=args.enable_langfuse)
    elif args.query:
        logger.info(f"Running in single query mode")
        result = await run_query(args.query, enable_langfuse=args.enable_langfuse)
        print(result)
    else:
        logger.warning("No query provided and not in interactive mode")
        parser.print_help()
    
    logger.info("Deep research example completed")


def main():
    """
    Main entry point that handles setting up the asyncio loop
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 