"""
Tests for the deep_research.step1 module.
"""
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from deep_research.step1 import (
    SubTask,
    ResearchPlan,
    generate_research_plan,
    _async_generate_plan,
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_subtasks():
    """Return a list of sample SubTask objects."""
    return [
        SubTask(id=1, task="Research task 1", priority=1),
        SubTask(id=2, task="Research task 2", priority=2),
        SubTask(id=3, task="Research task 3", priority=3),
    ]

@pytest.fixture
def sample_plan(sample_subtasks):
    """Return a sample ResearchPlan object."""
    return ResearchPlan(
        objective="Test research objective",
        sub_tasks=sample_subtasks,
    )

@pytest.fixture
def sample_json_response():
    """Return a sample JSON response from the OpenAI API."""
    return json.dumps({
        "objective": "Test research objective",
        "sub_tasks": [
            {"id": 1, "task": "Research task 1", "priority": 1},
            {"id": 2, "task": "Research task 2", "priority": 2},
            {"id": 3, "task": "Research task 3", "priority": 3},
        ]
    })

# -----------------------------
# Tests for data classes
# -----------------------------

def test_subtask_init():
    """Test that a SubTask can be initialized with the correct properties."""
    subtask = SubTask(id=1, task="Test task", priority=2)
    
    assert subtask.id == 1
    assert subtask.task == "Test task"
    assert subtask.priority == 2

def test_research_plan_init(sample_subtasks):
    """Test that a ResearchPlan can be initialized with the correct properties."""
    plan = ResearchPlan(objective="Test objective", sub_tasks=sample_subtasks)
    
    assert plan.objective == "Test objective"
    assert len(plan.sub_tasks) == 3
    assert isinstance(plan.sub_tasks[0], SubTask)
    assert plan.sub_tasks[0].id == 1
    assert plan.sub_tasks[0].task == "Research task 1"
    assert plan.sub_tasks[0].priority == 1

def test_research_plan_to_json(sample_plan):
    """Test that a ResearchPlan can be converted to JSON."""
    json_str = sample_plan.to_json()
    parsed = json.loads(json_str)
    
    assert parsed["objective"] == "Test research objective"
    assert len(parsed["sub_tasks"]) == 3
    assert parsed["sub_tasks"][0]["id"] == 1
    assert parsed["sub_tasks"][0]["task"] == "Research task 1"
    assert parsed["sub_tasks"][0]["priority"] == 1

# -----------------------------
# Tests for API functions
# -----------------------------

@patch("deep_research.step1._async_generate_plan")
def test_generate_research_plan(mock_async_generate_plan, sample_plan):
    """Test that generate_research_plan calls _async_generate_plan correctly."""
    # Set up the mock
    mock_async_generate_plan.return_value = sample_plan
    
    # Call the function
    result = generate_research_plan("Test objective", temperature=0.7)
    
    # Verify the mock was called with the right arguments
    mock_async_generate_plan.assert_called_once_with("Test objective", temperature=0.7)
    
    # Verify the result
    assert result is sample_plan
    assert result.objective == "Test research objective"
    assert len(result.sub_tasks) == 3

@patch("deep_research.step1.Runner.run")
@pytest.mark.asyncio
async def test_async_generate_plan(mock_runner_run, sample_json_response):
    """Test that _async_generate_plan processes the response correctly."""
    # Set up the mock result with final_output directly
    mock_result = MagicMock()
    
    # Set up the final_output attribute directly
    mock_result.final_output = sample_json_response
    
    # Set up the Runner.run mock to return our mock_result
    mock_runner_run.return_value = mock_result
    
    # Call the function
    result = await _async_generate_plan("Test objective", temperature=0.7)
    
    # Verify the runner was called
    mock_runner_run.assert_called_once()
    
    # Verify result structure
    assert result.objective == "Test research objective"
    assert len(result.sub_tasks) == 3
    assert isinstance(result.sub_tasks[0], SubTask)
    assert result.sub_tasks[0].id == 1
    assert result.sub_tasks[0].task == "Research task 1"
    assert result.sub_tasks[0].priority == 1
