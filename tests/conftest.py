"""
Common fixtures and setup for all tests.
"""
import json
import os
import pytest
from unittest.mock import MagicMock, AsyncMock

from deep_research.step1 import SubTask, ResearchPlan

# -----------------------------
# Environment setup
# -----------------------------

@pytest.fixture(autouse=True)
def setup_test_env():
    """
    Set up the test environment automatically for all tests.
    
    This fixture ensures that the OPENAI_API_KEY is set to a dummy value
    for testing purposes, preventing accidental API calls.
    """
    # Save the existing API key if present
    original_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Set a dummy API key for testing
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-api-key"
    
    yield  # This is where the test runs
    
    # Restore the original API key if it existed
    if original_api_key is not None:
        os.environ["OPENAI_API_KEY"] = original_api_key
    else:
        del os.environ["OPENAI_API_KEY"]

# -----------------------------
# Common test data
# -----------------------------

@pytest.fixture
def mock_api_response():
    """
    Create a mock API response object.
    
    Returns a MagicMock with a final_output attribute containing a valid JSON response.
    """
    mock_response = MagicMock()
    mock_response.final_output = json.dumps({
        "objective": "Test research objective",
        "sub_tasks": [
            {"id": 1, "task": "Research task 1", "priority": 1},
            {"id": 2, "task": "Research task 2", "priority": 2},
            {"id": 3, "task": "Research task 3", "priority": 3},
        ]
    })
    return mock_response

@pytest.fixture
def mock_async_runner():
    """Create a mock for the Runner.run method that returns an AsyncMock."""
    return AsyncMock()

# -----------------------------
# Helper functions
# -----------------------------

@pytest.fixture
def create_subtask():
    """
    Factory fixture to create SubTask objects with custom parameters.
    
    Example:
        def test_something(create_subtask):
            task1 = create_subtask(id=1, task="Custom task", priority=5)
    """
    def _create_subtask(id=1, task="Test task", priority=1):
        return SubTask(id=id, task=task, priority=priority)
    
    return _create_subtask

@pytest.fixture
def create_research_plan():
    """
    Factory fixture to create ResearchPlan objects with custom parameters.
    
    Example:
        def test_something(create_research_plan, create_subtask):
            tasks = [create_subtask(id=1), create_subtask(id=2)]
            plan = create_research_plan(objective="Custom objective", sub_tasks=tasks)
    """
    def _create_research_plan(objective="Test objective", sub_tasks=None):
        if sub_tasks is None:
            sub_tasks = [
                SubTask(id=1, task="Default task 1", priority=1),
                SubTask(id=2, task="Default task 2", priority=2),
            ]
        return ResearchPlan(objective=objective, sub_tasks=sub_tasks)
    
    return _create_research_plan 