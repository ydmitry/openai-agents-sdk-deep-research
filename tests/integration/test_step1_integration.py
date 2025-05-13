"""
Integration tests for the deep_research.step1 module.

These tests demonstrate how to test the integration with external services
without making real API calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from deep_research.step1 import generate_research_plan, ResearchPlan, SubTask

# -----------------------------
# Tests
# -----------------------------

@patch("agents.Runner.run")
def test_generate_research_plan_integration(mock_runner_run):
    """
    Test the generate_research_plan function with a mocked OpenAI API response.
    
    This test verifies that the function can process a realistic API response
    and convert it into a proper ResearchPlan object.
    """
    # 1. Set up the mock response data
    json_response = json.dumps({
        "objective": "Assess the economic impact of generative AI on the 2025 EU labor market",
        "sub_tasks": [
            {"id": 1, "task": "Identify key sectors in the EU labor market affected by generative AI", "priority": 1},
            {"id": 2, "task": "Gather 2024-2025 employment statistics from Eurostat and OECD", "priority": 2},
            {"id": 3, "task": "Research current generative AI adoption rates across EU industries", "priority": 3},
            {"id": 4, "task": "Analyze productivity gains from generative AI in comparable markets", "priority": 4},
            {"id": 5, "task": "Identify potential job displacement scenarios and timelines", "priority": 5},
            {"id": 6, "task": "Evaluate reskilling initiatives and their effectiveness in the EU", "priority": 6},
            {"id": 7, "task": "Compile expert opinions on generative AI's impact on EU employment", "priority": 7},
            {"id": 8, "task": "Examine EU policy response to AI-driven labor market changes", "priority": 8}
        ]
    })
    
    # Create a mock result object with final_output directly
    mock_result = MagicMock()
    mock_result.final_output = json_response
    
    # Configure the mock
    mock_runner_run.return_value = mock_result
    
    # 2. Call the function
    objective = "Assess the economic impact of generative AI on the 2025 EU labor market"
    plan = generate_research_plan(objective)
    
    # 3. Verify the result
    assert isinstance(plan, ResearchPlan)
    assert plan.objective == objective
    assert len(plan.sub_tasks) == 8
    
    # Verify first and last subtasks to ensure proper conversion
    assert isinstance(plan.sub_tasks[0], SubTask)
    assert plan.sub_tasks[0].id == 1
    assert plan.sub_tasks[0].priority == 1
    assert "key sectors" in plan.sub_tasks[0].task
    
    assert isinstance(plan.sub_tasks[-1], SubTask)
    assert plan.sub_tasks[-1].id == 8
    assert plan.sub_tasks[-1].priority == 8
    assert "policy response" in plan.sub_tasks[-1].task
    
    # 4. Verify the generated JSON is valid and contains all tasks
    json_output = plan.to_json()
    parsed = json.loads(json_output)
    assert len(parsed["sub_tasks"]) == 8
    assert parsed["objective"] == objective 