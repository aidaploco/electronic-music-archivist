import os
from unittest.mock import MagicMock, patch

import pytest

from src.agent.research_agent import ResearchAgent
from src.models.house_dj import HouseDJ


@pytest.fixture(scope="module")
def research_agent():
    """
    Fixture to initialize a ResearchAgent for testing.
    Mocks environment variables and external dependencies to ensure tests are isolated.
    """
    # Mock SERPER_API_KEY to prevent actual API calls during initialization
    with patch.dict(os.environ, {"SERPER_API_KEY": "mock_serper_api_key"}):
        # Patch the OllamaLLM and AgentExecutor classes that ResearchAgent instantiates in its __init__
        # This ensures that all objects are mocked from the moment ResearchAgent is created
        with patch('src.agent.research_agent.ChatOllama'), \
            patch('src.agent.research_agent.SerperSearchTool'), \
            patch('src.agent.research_agent.AgentExecutor'):
                agent = ResearchAgent(llm_model_name="mistral", llm_temperature=0.0)
                yield agent


def test_research_agent_initialization(research_agent):
    """
    Verifies that the ResearchAgent initializes correctly.
    """
    assert research_agent is not None
    assert hasattr(research_agent, 'llm')
    assert hasattr(research_agent, 'search_tool')
    assert hasattr(research_agent, 'agent_executor')
    assert hasattr(research_agent, 'parser')
    # Assert that the attributes are indeed mocks
    assert isinstance(research_agent.llm, MagicMock)
    assert isinstance(research_agent.search_tool_wrapper, MagicMock)
    assert isinstance(research_agent.agent_executor, MagicMock)

def test_run_research_successful_house_dj_output(research_agent):
    """
    Tests that run_research successfully returns a HouseDJ object for a valid query.
    Mocks the LLM's response to simulate a perfect structured output.
    """
    mock_house_dj_json_output = """
    {
      "name": "Frankie Knuckles",
      "aliases": ["The Godfather of House"],
      "birth_date": "1955-01-18",
      "birth_place": "The Bronx, New York, USA",
      "active_years": "1970s-2014",
      "notable_tracks": ["Your Love", "The Whistle Song"],
      "associated_labels": ["Trax Records"],
      "influences": ["Disco", "Soul"],
      "known_for": "Pioneering House Music",
      "biography_summary": "Frankie Knuckles was a legendary DJ.",
      "genres": ["House"],
      "website": "http://example.com/frankie",
      "social_media": {"twitter": "frankie_k"},
      "awards": ["Grammy"],
      "collaborations": ["Jamie Principle"],
      "legacy": "Changed music forever."
    }
    """

    with patch.object(research_agent.agent_executor, "invoke") as mock_invoke:
        mock_invoke.return_value = {"output": mock_house_dj_json_output}

        query = "Provide detailed information about the House DJ Frankie Knuckles."
        result = research_agent.run_research(query)

        assert isinstance(result, HouseDJ)
        assert result.name == "Frankie Knuckles"
        assert "The Godfather of House" in result.aliases
        assert result.birth_date == "1955-01-18"
        assert str(result.website) == "http://example.com/frankie"

def test_run_research_invalid_json_output(research_agent):
    """
    Tests that run_research raises an error when the LLM returns invalid JSON.
    """
    with patch.object(research_agent.agent_executor, "invoke") as mock_invoke:
        mock_invoke.return_value = {"output": "This is not valid JSON."}

        query = "Tell me about a non-existent DJ."

        with pytest.raises(Exception) as e: # Catching general Exception as LangChain might wrap it
            research_agent.run_research(query)

        assert "Invalid json output:" in str(e.value) or "JSONDecodeError" in str(e.value)

def test_run_research_non_conforming_json_output(research_agent):
    """
    Tests that run_research raises a ValidationError when the LLM returns
    valid JSON that does not conform to the HouseDJ schema (e.g., missing required fields, extra fields).
    """
    # Mock the agent_executor.invoke method to return valid JSON but missing 'name'
    mock_non_conforming_json = """
    {
      "aliases": ["Test Alias"],
      "birth_date": "2000-01-01"
    }
    """
    with patch.object(research_agent.agent_executor, "invoke") as mock_invoke:
        mock_invoke.return_value = {"output": mock_non_conforming_json}

        query = "Tell me about a DJ with incomplete info."

        with pytest.raises(Exception) as e: # Catching general Exception as LangChain might wrap it
            research_agent.run_research(query)

        assert "Failed to parse HouseDJ from completion" in str(e.value) or "field required" in str(e.value)

def test_run_research_search_tool_failure(research_agent):
    """
    Tests that run_research handles failures from the search tool gracefully.
    """
    with patch.object(research_agent.search_tool_wrapper, "run_search", side_effect=Exception("Mock search error")):
        # Mock the LLM's invoke method to ensure it doesn't try to generate output if tool fails early
        with patch.object(research_agent.agent_executor, "invoke",
                          side_effect=Exception("Agent execution failed due to tool error")):
            query = "Search for something that will fail."
            with pytest.raises(Exception) as e:
                research_agent.run_research(query)

            assert "Agent execution failed due to tool error" in str(e.value)
