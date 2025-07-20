import logging
import os
from typing import cast

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper as SerperAPIWrapper
from tenacity import after_log, before_log, retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)
load_dotenv()


class SerperSearchTool:
    """
    A wrapper for the Serper.dev Google Search API.
    Provides a search tool for the agent to query the web.
    Incorporates retry logic using tenacity for robustness.
    """
    def __init__(self):
        """
        Initializes the SerperSearchTool.
        Raises ValueError if SERPER_API_KEY is not found in environment variables.
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            logger.error("SERPER_API_KEY environment variable not set.")
            raise ValueError(
                "SERPER_API_KEY environment variable not set. "
                "Ensure API Key from serper.dev is added to your .env file."
            )

        logger.info("Initializing SerperSearchTool with provided API key.")
        self.tool = SerperAPIWrapper(serper_api_key=api_key)
        logger.info("SerperSearchTool initialized successfully.")

    def get_tool(self) -> SerperAPIWrapper:
        """
        Returns the initialized LangChain SerperAPIWrapper.
        """
        # Use cast to explicitly tell mypy the type of self.tool
        return cast(SerperAPIWrapper, self.tool)

    @retry(
        stop=stop_after_attempt(3), # Try up to 3 times
        wait=wait_random_exponential(min=1, max=5), # Wait exponentially between 1 and 5 seconds
        before=before_log(logger, logging.INFO), # Log before each retry attempt
        after=after_log(logger, logging.WARNING), # Log after each retry attempt (especially on failure)
        reraise=True # Re-raise the exception if all attempts fail
    )
    def run_search(self, query: str) -> str:
        """
        Executes a search query using the Serper Search tool and returns the results.
        """
        logger.info(f"Attempting search for query: '{query}'")
        try:
            # Get the formatted summary of the top results.
            results = self.tool.run(query)
            logger.info(f"Search for '{query}' completed successfully.")
            # Use cast to explicitly tell mypy that results is a string
            return cast(str, results)
        except Exception as e:
            logger.error(f"Error running Serper Search for query '{query}': {e}. Retrying...", exc_info=True)
            raise # Re-raise the exception to trigger tenacity's retry mechanism


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting SerperSearchTool test with retry logic...")
    try:
        search_tool_instance = SerperSearchTool()
        serper_tool = search_tool_instance.get_tool()
        logger.info("SerperSearch tool instance obtained.")

        # Example query for our "Electronic Music Archivist" project
        test_query = "influential house DJs of the 90s"
        logger.info(f"Searching for: '{test_query}'")
        search_results = search_tool_instance.run_search(test_query)
        logger.info("Search Results for first query:")
        print(search_results)

        test_query_2 = "evolution of Chicago house music"
        logger.info(f"\nSearching for: '{test_query_2}'")
        search_results_2 = search_tool_instance.run_search(test_query_2)
        logger.info("Search Results for second query:")
        print(search_results_2)

    except ValueError as ve:
        logger.critical(f"Configuration Error during test: {ve}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during test: {e}")
        print(f"Test failed due to: {e}")
