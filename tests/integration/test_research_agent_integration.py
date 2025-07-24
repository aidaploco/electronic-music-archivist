import os

import pytest
from dotenv import load_dotenv

from src.agent.research_agent import ResearchAgent
from src.models.house_dj import HouseDJ

load_dotenv()
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_research_agent():
    """
    Fixture to initialize a ResearchAgent for integration testing.
    It requires a running Ollama server and a valid SERPER_API_KEY.
    """
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key or serper_api_key == "your_serper_api_key_here":
        pytest.skip("SERPER_API_KEY not set or is default. Skipping integration tests.")

    agent = ResearchAgent(llm_model_name="mistral", llm_temperature=0.0)
    yield agent


def test_integration_structured_output(live_research_agent):
    """
    Verifies that the ResearchAgent can fetch and parse structured information using real LLM and search.
    """
    query = "Provide detailed information about the House DJ Frankie Knuckles."

    try:
        result = live_research_agent.run_research(query)
        assert isinstance(result, HouseDJ)
        assert result.name == "Frankie Knuckles"
        assert result.birth_date is not None
        assert result.biography_summary is not None
        assert len(result.notable_tracks) > 0 if result.notable_tracks else False

    except Exception as e:
        pytest.fail(f"Integration test failed for Frankie Knuckles: {e}")

def test_integration_genre_natural_language_output(live_research_agent):
    """
    Verifies that the ResearchAgent correctly handles queries that do not conform
    to the HouseDJ schema by raising an error.
    """
    query = "What are the key characteristics of Chicago House music?"

    with pytest.raises(Exception) as e:
        live_research_agent.run_research(query)

    assert "AgentExecutor output did not contain a valid JSON block." in str(e.value)
