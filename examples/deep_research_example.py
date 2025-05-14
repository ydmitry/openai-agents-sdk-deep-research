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

# Add new models for task-oriented approach
class TaskComponent(BaseModel):
    """Represents a component of the solution to the user's task"""
    component_type: str  # code, instruction, resource, etc.
    content: str
    usage_instructions: str
    relevance_score: float = 0.0  # 0.0-1.0

class TaskSolution(BaseModel):
    """Represents a complete solution to the user's task"""
    task_summary: str = Field(description="Summary of the identified task")
    solution_components: List[TaskComponent] = Field(description="Components of the solution")
    prerequisites: List[str] = Field(description="Prerequisites for implementing the solution")
    implementation_steps: List[str] = Field(description="Step-by-step implementation guide")
    verification_steps: List[str] = Field(description="Steps to verify the solution works")
    fallback_options: List[str] = Field(description="Alternative approaches if main solution fails")
    
class WorkflowState(BaseModel):
    """Shared state for the task solution workflow"""
    query: str = Field(description="The original user query")
    task_type: str = Field(default="", description="The identified task type")
    task_components: List[str] = Field(default_factory=list, description="Components needed for the solution")
    search_results: Dict[str, List[SearchResult]] = Field(default_factory=dict, description="Search results by component")
    solution_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Solution elements")
    current_phase: str = Field(default="analysis", description="Current phase of task solution")
    iterations_done: int = Field(default=0, description="Number of solution iterations completed")
    remaining_needs: List[str] = Field(default_factory=list, description="Identified remaining solution needs")
    
    class Config:
        arbitrary_types_allowed = True

# Pydantic models for function tool input/output
class TaskAnalysisResponse(BaseModel):
    task_type: str = Field(description="The type of task identified")
    task_components: List[str] = Field(description="Required components to complete the task")

class SolutionQueries(BaseModel):
    queries: List[str] = Field(description="List of search queries for solution components")

class ComponentEvaluationResult(BaseModel):
    quality_score: float = Field(description="Quality score from 0.0 to 1.0")
    implementation_difficulty: str = Field(description="Difficulty of implementation (easy, medium, hard)")
    completeness: float = Field(description="Completeness score from 0.0 to 1.0")

class SolutionGapAnalysisResult(BaseModel):
    remaining_needs: List[str] = Field(description="List of identified missing solution components")
    complete: bool = Field(description="Whether the solution is considered complete")

class FinalTaskSolution(BaseModel):
    task_summary: str = Field(description="Summary of the identified task")
    solution_components: List[TaskComponent] = Field(description="Components of the solution")
    implementation_guide: str = Field(description="Comprehensive implementation guide")
    verification_method: str = Field(description="Method to verify the solution works")
    resources_used: List[str] = Field(description="List of resources consulted")

def load_deep_research_agent(enable_langfuse: bool = False, service_name: str = "deep_research_agent"):
    """
    Factory function to create and return a task solution system with multiple specialized agents.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name to use for Langfuse tracing
        
    Returns:
        tuple: (agent, run_agent_function) - The configured triage agent and a function to run queries.
    """
    import openai
    from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool
    from agents import handoff

    logger.info("Initializing task solution agent system")

    # Set up Langfuse tracing if enabled
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse
            
            if setup_langfuse():
                logger.info("Langfuse tracing enabled for task solution agent")
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
    
    # Custom function tools for task solution
    
    @function_tool
    async def identify_user_task(query: str) -> TaskAnalysisResponse:
        """
        Analyze the user's query to identify the specific task they need completed.
        
        Args:
            query: The user's query
            
        Returns:
            A response containing the task type and required components
        """
        # In a real implementation, this would use a language model to analyze the task
        # For this example, we'll return fixed generic task components
        
        task_components = [
            "Implementation code",
            "Configuration setup",
            "Integration steps",
            "Testing procedure",
            "Deployment instructions",
            "Troubleshooting guide"
        ]
        
        # Update the workflow state
        workflow_state.task_components = task_components
        workflow_state.task_type = "implementation"
        workflow_state.current_phase = "research"
        
        return TaskAnalysisResponse(task_type="implementation", task_components=task_components)
    
    @function_tool
    async def generate_solution_queries(component: str) -> SolutionQueries:
        """
        Generate search queries to find solutions for a specific component of the task.
        
        Args:
            component: The solution component to generate queries for
            
        Returns:
            A list of search queries
        """
        # Generate variants of the search query to get diverse solution approaches
        variants = [
            f"{component} {workflow_state.task_type}",
            f"{component} tutorial",
            f"{component} example code",
            f"{component} best practices",
            f"{component} step by step",
            f"{component} modern approach",
            f"{component} quick implementation"
        ]
        
        return SolutionQueries(queries=variants)
    
    @function_tool
    async def evaluate_solution_component(component_content: str, component_type: str) -> ComponentEvaluationResult:
        """
        Evaluate a potential solution component for quality and completeness.
        
        Args:
            component_content: Content of the potential solution component
            component_type: Type of component (code, config, etc.)
            
        Returns:
            Evaluation results including quality and completeness scores
        """
        # In a real implementation, this would use a language model to evaluate
        # the solution component for quality, completeness, and difficulty
        
        # Return placeholder scores
        return ComponentEvaluationResult(
            quality_score=0.85,
            implementation_difficulty="medium",
            completeness=0.9
        )
    
    @function_tool
    async def analyze_solution_completeness() -> SolutionGapAnalysisResult:
        """
        Analyze the current solution to identify any missing components or gaps.
        
        Returns:
            Analysis of remaining needs and completeness status
        """
        # Check which components have search results
        covered_components = set(workflow_state.search_results.keys())
        all_components = set(workflow_state.task_components)
        
        # Identify gaps as components without search results
        missing_components = all_components - covered_components
        
        remaining_needs = [f"Missing solution for: {component}" for component in missing_components]
        
        # If we've done at least one iteration but don't have complete solution components
        if workflow_state.iterations_done >= 1:
            total_results = sum(len(results) for results in workflow_state.search_results.values())
            if total_results < len(workflow_state.task_components):
                remaining_needs.append("Need more comprehensive solution components")
        
        # Determine if solution is complete based on iterations and gaps
        complete = (workflow_state.iterations_done >= 2) or (workflow_state.iterations_done >= 1 and not remaining_needs)
        
        # Update workflow state
        workflow_state.remaining_needs = remaining_needs
        workflow_state.iterations_done += 1
        
        if complete:
            workflow_state.current_phase = "delivery"
        else:
            workflow_state.current_phase = "research"
            
        return SolutionGapAnalysisResult(remaining_needs=remaining_needs, complete=complete)
    
    @function_tool
    async def create_executable_solution() -> FinalTaskSolution:
        """
        Create a complete, executable solution for the user's task from gathered components.
        
        Returns:
            The final task solution with implementation steps and verification methods
        """
        # In a real implementation, this would use the LLM to assemble all solution
        # components into a cohesive, executable solution
        
        # Flatten all search results
        all_results = []
        for results_list in workflow_state.search_results.values():
            all_results.extend(results_list)
        
        # Create a much more focused set of solution components
        # Prioritize code and immediate action steps over educational content
        essential_components = []
        code_components = []
        instruction_components = []
        
        # Process and filter components for a focused solution
        for i, result in enumerate(all_results):
            # Extract the most actionable content from each result
            content = result.content
            
            # Categorize content based on its actionability
            if "```" in content or "import" in content or "class" in content or "function" in content:
                # This looks like code - prioritize it
                component_type = "code"
                # Try to extract just the code if it's embedded in explanatory text
                code_blocks = []
                if "```" in content:
                    # Extract code blocks
                    parts = content.split("```")
                    for j in range(1, len(parts), 2):
                        if j < len(parts):
                            # Extract content between triple backticks, removing language identifier
                            code = parts[j]
                            if code and "\n" in code:
                                # Remove language identifier line if present
                                first_line_end = code.find("\n")
                                if first_line_end > -1:
                                    if not any(c.isalnum() for c in code[:first_line_end].strip()):
                                        code = code[first_line_end+1:]
                                code_blocks.append(code)
                
                if code_blocks:
                    content = "\n\n".join(code_blocks)
                
                # Add to code components
                code_components.append(TaskComponent(
                    component_type=component_type,
                    content=content,
                    usage_instructions="Copy and use this code directly",
                    relevance_score=0.95
                ))
            elif any(keyword in content.lower() for keyword in ["step", "install", "setup", "configure", "run", "execute"]):
                # This looks like actionable instructions
                component_type = "instruction"
                # Try to extract just the steps
                steps = []
                for line in content.split("\n"):
                    if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "•", "*", "-", "Step")):
                        steps.append(line.strip())
                
                if steps:
                    content = "\n".join(steps)
                
                # Add to instruction components
                instruction_components.append(TaskComponent(
                    component_type=component_type,
                    content=content,
                    usage_instructions="Follow these steps exactly",
                    relevance_score=0.9
                ))
        
        # Prioritize most valuable components first
        essential_components.extend(code_components[:3])  # Top 3 code components
        essential_components.extend(instruction_components[:3])  # Top 3 instruction components
        
        # Ensure we have at least one code example if available
        if not any(c.component_type == "code" for c in essential_components) and code_components:
            essential_components.append(code_components[0])
        
        # Create the final solution - much more focused and actionable
        implementation_steps = [
            "1. Copy the provided code into your development environment",
            "2. Install any required dependencies",
            "3. Configure the system as specified",
            "4. Execute the code as instructed",
            "5. Verify the solution works using the verification steps"
        ]
        
        # Create a focused, action-oriented solution
        return FinalTaskSolution(
            task_summary=f"Implementation solution for {workflow_state.query}",
            solution_components=essential_components,
            implementation_guide="# Quick Implementation Guide\n\n" + 
                               "Below is your ready-to-use solution. Just follow these steps:\n\n" +
                               "\n".join(implementation_steps) + 
                               "\n\nThe code provided is complete and should work immediately without additional research.",
            verification_method="Run the code and verify it produces the expected output. If any issues occur, check the troubleshooting section.",
            resources_used=[result.url for result in all_results[:3]]  # Limit to top 3 resources
        )
    
    # Enhanced web search tool
    web_search_tool = WebSearchTool(
        search_context_size='high',  # Use high context for detailed results
    )
    
    # Create specialized task analysis agent
    task_analysis_agent = Agent(
        name="Task Analysis Agent",
        instructions="""
        You are a specialized agent focused on analyzing user queries to identify specific tasks.
        
        Follow these guidelines:
        1. Determine exactly what task the user is trying to accomplish
        2. Break down the task into specific components that must be addressed
        3. Identify any constraints or special requirements
        4. Prioritize components based on their importance to task completion
        5. Look for implied needs that the user may not have explicitly stated
        
        Your goal is to create a precise task analysis that will guide
        the rest of the solution process.
        """,
        tools=[identify_user_task],
        model="gpt-4.1",
        model_settings=lighter_model_settings,
    )
    
    logger.debug("Task analysis agent created")

    # Create solution research agent
    solution_research_agent = Agent(
        name="Solution Research Agent",
        instructions="""
        You are a specialized agent focused on finding practical solutions to complete the user's task.
        Your goal is to find concrete, implementable solution components.
        
        Follow these guidelines:
        1. For each task component, generate focused search queries using generate_solution_queries
        2. Execute web searches for each query to find solution components
        3. For each potential solution component:
           - Use evaluate_solution_component to assess quality and completeness
           - Extract practical, usable code and instructions
        4. Prioritize finding:
           - Working code examples
           - Step-by-step implementation guides
           - Configuration templates
           - Testing procedures
           - Performance optimization techniques
        5. Document all solution components thoroughly
        6. Research ALL the task components provided to ensure a complete solution
        
        Remember, your goal is to find PRACTICAL solution components that can be
        DIRECTLY IMPLEMENTED by the user to complete their task.
        Be methodical and thorough, making sure to search for each required component.
        """,
        tools=[
            generate_solution_queries,
            web_search_tool,
            evaluate_solution_component,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Solution research agent created with focused solution search strategies")

    # Create solution integration agent
    solution_integration_agent = Agent(
        name="Solution Integration Agent",
        instructions="""
        You are a specialized agent that integrates solution components into a cohesive whole.
        Your goal is to ensure all components work together to complete the user's task.
        
        Follow these guidelines:
        1. Carefully analyze all the solution components that have been collected
        2. Identify any inconsistencies or conflicts between components
        3. Ensure all dependencies are identified and addressed
        4. Verify that all task components have corresponding solution elements
        5. Use the analyze_solution_completeness tool to identify any missing components
        6. Create a logical sequence for implementing the complete solution
        
        Your role is to ensure all components fit together into a complete solution
        and determine if additional research is needed before proceeding to delivery.
        """,
        tools=[
            analyze_solution_completeness,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Solution integration agent created with gap identification")

    # Create solution delivery agent
    solution_delivery_agent = Agent(
        name="Solution Delivery Agent",
        instructions="""
        You are a specialized agent that creates the final executable solution for the user.
        Your goal is to deliver a complete, ready-to-use solution that accomplishes the user's task 
        with minimal explanation and maximum actionability.
        
        Follow these guidelines:
        1. Use the create_executable_solution tool to generate the final solution package
        2. Focus EXCLUSIVELY on providing an executable solution:
           - Prioritize working code over explanations
           - Include ONLY necessary configuration details
           - Provide BRIEF implementation steps in logical order
           - Include minimal verification methods to confirm success
           - Add only ESSENTIAL troubleshooting tips
        
        IMPORTANT STYLE GUIDELINES:
        - Be extremely concise and action-oriented
        - Cut all educational/background material
        - Remove ALL theoretical explanations
        - Eliminate ALL content not directly needed to implement the solution
        - Format code and instructions for immediate use
        - Focus on "DO THIS NOW" rather than "here's what you could do"
        
        Your goal is to create a MINIMAL solution that the user can implement IMMEDIATELY
        to accomplish their task with ZERO additional effort or research.
        
        The perfect solution should look like a quick recipe or cheat sheet,
        not an educational document or tutorial.
        """,
        tools=[
            create_executable_solution,
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Solution delivery agent created for final executable solutions")

    # Convert agents to tools instead of using handoffs
    task_analysis_tool = task_analysis_agent.as_tool(
        "analyze_user_task",
        "Analyzes the user's query to identify the specific task and required components"
    )
    
    solution_research_tool = solution_research_agent.as_tool(
        "research_solution_components",
        "Researches practical solutions for each task component"
    )
    
    solution_integration_tool = solution_integration_agent.as_tool(
        "integrate_solution_components",
        "Integrates individual solution components into a cohesive whole and identifies any gaps"
    )
    
    solution_delivery_tool = solution_delivery_agent.as_tool(
        "deliver_executable_solution",
        "Creates a complete, executable solution package for the user's task"
    )
    
    # Enhanced triage agent that manages the task solution workflow using tools
    triage_agent = Agent(
        name="Task Solution Coordinator",
        instructions="""
        You are the lead solution architect coordinating the process of completing the user's task.
        Your goal is to deliver a quick, actionable solution that accomplishes exactly what the user needs.
        
        YOU MUST FOLLOW THIS EXACT PROCESS IN ORDER:
        
        1. FIRST, use the analyze_user_task tool to identify the specific task and required components
           - Understand exactly what the user needs to accomplish
        
        2. THEN, use the research_solution_components tool to find practical solutions for ALL components
           - Focus on finding code and implementation steps, not explanations
        
        3. NEXT, use the integrate_solution_components tool to ensure completeness and coherence
           - Make sure all components work together
        
        4. FINALLY, use the deliver_executable_solution tool to create the ready-to-use solution
           - The solution must be concise, focused and immediately actionable
        
        You MUST execute ALL FOUR TOOLS in EXACTLY this order. Do not skip any steps.
        
        YOUR SOLUTION SHOULD:
        - Contain primarily code and direct implementation steps
        - Avoid unnecessary explanations or background
        - Be formatted for immediate use with minimal reading
        - Include only what's needed to complete the task
        - Be testable and verifiable
        
        YOUR RESPONSE SHOULD ONLY CONTAIN THE FINAL SOLUTION without any wrapper text.
        Do not add your own introduction, explanation, or conclusion.
        """,
        tools=[
            task_analysis_tool,
            solution_research_tool, 
            solution_integration_tool,
            solution_delivery_tool
        ],
        model="gpt-4.1",
        model_settings=lighter_model_settings,
    )
    
    logger.debug("Enhanced triage agent created with tools pattern for full task solution workflow")

    async def run_agent(query: str):
        """Run the agent with the given query."""
        logger.info(f"Running task solution query: {query}")
        
        # Initialize the workflow state
        workflow_state.query = query
        # Reset any previous state
        workflow_state.task_components = []
        workflow_state.search_results = {}
        workflow_state.solution_elements = []
        workflow_state.current_phase = "analysis"
        workflow_state.iterations_done = 0
        workflow_state.remaining_needs = []
        
        try:
            # If Langfuse is enabled, wrap the agent run with a trace context
            if enable_langfuse:
                try:
                    from deep_research.langfuse_integration import create_trace
                    
                    # Generate a session ID based on the first few chars of the query
                    session_id = f"task_solution_{hash(query) % 10000}"
                    
                    with create_trace(
                        name="Task-Solution-Query",
                        session_id=session_id,
                        tags=["task_solution"],
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
                            I need a ready-to-use solution for this task: {query}
                            
                            IMPORTANT:
                            1. Provide a focused, minimal solution I can implement immediately 
                            2. Prioritize code and direct actions over explanations
                            3. Include only what's necessary to complete the task
                            4. Format everything for immediate use
                            
                            Begin the task solution process now.
                            """,
                            max_turns=30,  # Provide enough turns for all tools to be used
                        )
                        
                        # Set output for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("output.value", result.final_output)
                                # Add solution metrics to trace
                                span.set_attribute("solution.iterations_completed", workflow_state.iterations_done)
                                span.set_attribute("solution.components_count", len(workflow_state.search_results))
                            except Exception as e:
                                logger.warning(f"Could not set output attribute on span: {e}")
                        
                        logger.info(f"Task solution completed successfully")
                        logger.debug(f"Response length: {len(result.final_output)}")
                        
                        # Format the final task solution response - much more minimal formatting
                        formatted_output = result.final_output
                        
                        # Add only minimal resources section if we have any search results
                        if workflow_state.search_results:
                            all_sources = set()
                            for results_list in workflow_state.search_results.values():
                                for result_item in results_list:
                                    all_sources.add(result_item.url)
                            
                            if all_sources:
                                formatted_output += "\n\n## Key Resources\n\n"
                                for source in list(all_sources)[:3]:  # Limit to top 3
                                    formatted_output += f"- {source}\n"
                        
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
                I need a ready-to-use solution for this task: {query}
                
                IMPORTANT:
                1. Provide a focused, minimal solution I can implement immediately 
                2. Prioritize code and direct actions over explanations
                3. Include only what's necessary to complete the task
                4. Format everything for immediate use
                
                Begin the task solution process now.
                """,
                max_turns=30,  # Provide enough turns for all tools to be used
            )
            logger.info(f"Task solution completed successfully")
            logger.debug(f"Response length: {len(result.final_output)}")
            
            # Format the final task solution response - much more minimal formatting
            formatted_output = result.final_output
            
            # Add only minimal resources section if we have any search results
            if workflow_state.search_results:
                all_sources = set()
                for results_list in workflow_state.search_results.values():
                    for result_item in results_list:
                        all_sources.add(result_item.url)
                
                if all_sources:
                    formatted_output += "\n\n## Key Resources\n\n"
                    for source in list(all_sources)[:3]:  # Limit to top 3
                        formatted_output += f"- {source}\n"
            
            return formatted_output
                
        except Exception as e:
            logger.error(f"Error running task solution agent: {str(e)}")
            raise

    logger.info("Task solution agent system initialized and ready for queries")
    return triage_agent, run_agent


async def run_query(query: str, enable_langfuse: bool = False):
    """Run a single query through the task solution agent."""
    logger.info(f"Processing single query: {query}")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the task solution agent with optional Langfuse tracing
    logger.debug(f"Loading task solution agent (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    # Run the query and return the result
    logger.info("Sending query to agent")
    print("Creating solution... This may take several minutes for comprehensive task completion.")
    result = await run_agent(query)
    logger.info("Query completed successfully")
    return result


async def interactive_mode(enable_langfuse: bool = False):
    """Run the task solution agent in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the task solution agent (only once for the session) with optional Langfuse tracing
    logger.debug("Loading task solution agent for interactive session")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    print("Task Solution Agent (type 'exit' to quit)")
    print("Tell me what task you need completed, and I'll provide a complete solution.")
    
    query_count = 0
    while True:
        query = input("\nWhat task can I help you complete? ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Creating solution... (this may take several minutes to develop a complete solution)")
        try:
            response = await run_agent(query)
            logger.info(f"Successfully completed interactive query #{query_count}")
            print("\nYour solution:")
            print(response)
        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")
    
    logger.info(f"Interactive session ended after {query_count} queries")


async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Get complete solutions for your tasks")
    parser.add_argument("query", nargs="?", help="The task you need completed")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--enable-langfuse", action="store_true", 
                        help="Enable Langfuse tracing for observability")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of solution refinement iterations (default: 3)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive):
        # Auto-create log file for debug mode or interactive sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/task_solution_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting task solution system with log level: {args.log_level}")
    
    if args.enable_langfuse:
        logger.info("Langfuse tracing is enabled")
    
    if args.interactive:
        await interactive_mode(enable_langfuse=args.enable_langfuse)
    elif args.query:
        logger.info(f"Running in single query mode")
        result = await run_query(args.query, enable_langfuse=args.enable_langfuse)
        print(result)
    else:
        logger.warning("No task provided and not in interactive mode")
        parser.print_help()
    
    logger.info("Task solution system completed")


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