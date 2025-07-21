import os
from unittest.mock import patch

import pytest
from langchain_community.utilities import GoogleSerperAPIWrapper

from src.tools.serper_search import SerperSearchTool


@pytest.fixture(scope="module", autouse=True)
def mock_serper_api_key():
    """
    Mocks the SERPER_API_KEY environment variable to ensure tests run without
    needing a real key and are isolated from actual API calls during initialization.
    """
    with patch.dict(os.environ, {"SERPER_API_KEY": "mock_test_key"}):
        yield

@pytest.fixture
def mock_serper_wrapper_run():
    """
    Mocks the .run() method of GoogleSerperAPIWrapper to control its return value.
    """
    with patch.object(GoogleSerperAPIWrapper, 'run') as mock_run:
        yield mock_run


def test_serper_search_tool_initialization_success():
    """
    Verifies that SerperSearchTool initializes correctly when API key is present.
    """
    ss_tool = SerperSearchTool()
    assert ss_tool is not None
    assert isinstance(ss_tool.tool, GoogleSerperAPIWrapper)

def test_serper_search_tool_initialization_failure():
    """
    Tests that SerperSearchTool raises a ValueError if SERPER_API_KEY is missing.
    """
    # Temporarily remove SERPER_API_KEY from environment for this specific test
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SERPER_API_KEY environment variable not set"):
            SerperSearchTool()

def test_run_search_success(mock_serper_wrapper_run):
    """
    Tests that run_search method correctly calls the underlying Serper API and returns results.
    """
    # Configure the mock to return a predefined search result string
    mock_serper_wrapper_run.return_value = "Mock search results for 'test query'."

    ss_tool = SerperSearchTool()
    query = "test query"
    results = ss_tool.run_search(query)

    # Assert that the underlying Serper API's run method was called with the correct query
    mock_serper_wrapper_run.assert_called_once_with(query)
    # Assert that the returned results match the mock's return value
    assert results == "Mock search results for 'test query'."

def test_run_search_retry_on_failure(mock_serper_wrapper_run):
    """
    Tests that run_search retries on transient failures.
    """
    # Configure the mock to raise an exception twice, then succeed on the third attempt
    mock_serper_wrapper_run.side_effect = [
        Exception("Transient network error 1"),
        Exception("Transient network error 2"),
        "Successful search results after retries."
    ]

    ss_tool = SerperSearchTool()
    query = "query for retry"
    results = ss_tool.run_search(query)

    # Assert that the underlying Serper API's run method was called 3 times (initial + 2 retries)
    assert mock_serper_wrapper_run.call_count == 3
    # Assert that the final result is the successful one
    assert results == "Successful search results after retries."

def test_run_search_all_retries_fail(mock_serper_wrapper_run):
    """
    Tests that run_search raises an exception if all retry attempts fail.
    """
    # Configure the mock to always raise an exception
    mock_serper_wrapper_run.side_effect = Exception("Persistent API error")

    ss_tool = SerperSearchTool()
    query = "query that always fails"

    # Expect an exception to be raised after all retries are exhausted
    with pytest.raises(Exception, match="Persistent API error"):
        ss_tool.run_search(query)

    # Assert that the underlying Serper API's run method was called 3 times (initial + 2 retries)
    assert mock_serper_wrapper_run.call_count == 3
