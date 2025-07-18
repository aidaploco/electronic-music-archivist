import logging
from typing import cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

from src.utils.serper_search import SerperSearchTool


class ResearchAgent:
    """
    The core autonomous research agent for "The Electronic Music Archivist" project.
    It orchestrates web search, information synthesis, and structured output generation.
    """
    def __init__(self, llm_model_name: str = "mistral"):
        """
        Initializes the ResearchAgent with an LLM and a search tool.

        Args:
            llm_model_name (str): The name of the Ollama model to use (e.g., "mistral", "llama3").
        """
        logger.info("Initializing ResearchAgent...")
        try:
            self.llm = OllamaLLM(model=llm_model_name)
            logger.info(f"LLM '{llm_model_name}' successfully loaded for agent.")

            self.search_tool_wrapper = SerperSearchTool()
            # LangChain agents often expect tools to be wrapped as langchain_core.tools.Tool
            self.search_tool = Tool(
                name="Serper Search",
                func=self.search_tool_wrapper.run_search,
                description="A search engine. Useful for when you need to answer questions about current events \
                            or retrieve information from the web. Input should be a concise search query."
            )
            logger.info("Serper Search Tool successfully loaded for agent.")

            self.tools = [self.search_tool]
            logger.info("Agent initialized with LLM and Search Tool.")

        except Exception as e:
            logger.critical(f"Failed to initialize ResearchAgent: {e}")
            raise

    def _get_base_chain(self):
        """
        Defines a basic LangChain runnable chain for demonstration.
        This will be expanded to include agentic capabilities.
        """
        # A simple prompt for testing the LLM directly through the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant specializing in electronic music. \
                        Answer the user's question concisely."),
            ("user", "{question}")
        ])

        # A simple chain that just passes the question to the LLM
        # This will be replaced by a more complex agent executor later
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def run_research(self, query: str) -> str:
        """
        Executes a research query using the agent.
        This is a placeholder and will be expanded to use tools and structured output.

        Args:
            query (str): The research query.

        Returns:
            str: The research result (currently a direct LLM response).
        """
        logger.info(f"Agent received research query: '{query}'")

        # For now, we'll just use the base chain to get a direct LLM response.
        # In future steps, this will involve using the search tool and processing results.
        response = self._get_base_chain().invoke({"question": query})
        logger.info("Agent completed initial research phase.")

        # Explicitly cast the response to str to satisfy mypy
        return cast(str, response)


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting ResearchAgent test...")
    try:
        # Ensure Ollama server is running and 'mistral' model is pulled
        # Ensure SERPER_API_KEY is set in .env

        agent = ResearchAgent(llm_model_name="mistral")
        test_query_llm = "Who is the 'Godfather of House Music'?"
        print(f"\n--- Running direct LLM query through agent: '{test_query_llm}' ---")
        llm_response = agent.run_research(test_query_llm)
        print("LLM Response:")
        print(llm_response)

        # In a later step, we will make the agent *use* the search tool.
        # For now, the run_research method only uses the LLM directly.
        # We will integrate tool usage and structured output in the next iterations.

    except Exception as e:
        logger.critical(f"An error occurred during ResearchAgent test: {e}")

