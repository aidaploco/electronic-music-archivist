import logging
from typing import cast

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from src.utils.serper_search import SerperSearchTool


class ResearchAgent:
    """
    The core autonomous research agent for "The Electronic Music Archivist" project.
    It orchestrates web search, information synthesis, and structured output generation.
    """
    def __init__(self, llm_model_name: str = "mistral", llm_temperature: float = 0.0):
        """
        Initializes the ResearchAgent with an LLM and a search tool.

        Args:
            llm_model_name (str): The name of the Ollama model to use (e.g., "mistral", "llama3").
            llm_temperature (float): The temperature to use for the LLM (higher values = more creative).
        """
        logger.info("Initializing ResearchAgent...")
        try:
            self.llm = ChatOllama(model=llm_model_name, temperature=llm_temperature)
            logger.info(f"LLM '{llm_model_name}' (temperature={llm_temperature}) successfully loaded for agent.")

            self.search_tool = Tool(
                name="Serper Search",
                func=SerperSearchTool().run_search,
                description="A search engine. Useful for when you need to answer questions about current events \
                            or retrieve information from the web. Input should be a concise search query."
            )
            logger.info("Serper Search Tool successfully loaded for agent.")

            self.tools = [self.search_tool]

            self.agent_prompt = ChatPromptTemplate([
                ("system", """You are an expert electronic music archivist. Your goal is to research and provide
                            comprehensive, accurate information about House DJs and electronic music history.
                            Always cite your sources when you use information obtained via search.
                            If you can't find relevant information, state that clearly."""),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            self.agent = create_tool_calling_agent(self.llm, self.tools, self.agent_prompt)

            # The runtime that drives the agent, executing its decisions.
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
            logger.info("Agent Executor configured for autonomous research.")

        except Exception as e:
            logger.critical(f"Failed to initialize ResearchAgent: {e}")
            raise

    def run_research(self, query: str) -> str:
        """
        Executes a research query using the autonomous agent.
        The agent will decide whether to use the search tool based on the query.

        Args:
            query (str): The research query.

        Returns:
            str: The research result synthesized by the agent.
        """
        logger.info(f"Agent received research query: '{query}'")

        try:
            # Chat_history is an empty list for now
            result = self.agent_executor.invoke({"input": query, "chat_history": []})
            response = result.get("output", "No output found from agent.")

            logger.info("Agent completed research phase.")
            # Use cast to explicitly tell mypy that response is a string
            return cast(str, response)

        except Exception as e:
            logger.error(f"Error during agent research for query '{query}': {e}")
            return f"Error during agent research: {e}"


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting ResearchAgent test with autonomous capabilities...")
    try:
        # Ensure Ollama server is running and 'mistral' model is pulled
        # Ensure SERPER_API_KEY is set in .env

        # Test with Mistral model (default temperature 0.0)
        agent_default_temp = ResearchAgent(llm_model_name="mistral")
        test_query_default = "Who is the Godfather of House Music?"
        print(f"\n--- Running agent research query (default temp): '{test_query_default}' ---")
        agent_response_default = agent_default_temp.run_research(test_query_default)
        print("\nAgent's Final Response (default temp):")
        print(agent_response_default)

        # Test with a higher temperature for potentially more creative responses
        agent_higher_temp = ResearchAgent(llm_model_name="mistral", llm_temperature=0.7)
        test_query_creative = "Who is the Godfather of House Music?"
        print(f"\n--- Running agent research query (higher temp): '{test_query_creative}' ---")
        agent_response_creative = agent_higher_temp.run_research(test_query_creative)
        print("\nAgent's Final Response (higher temp):")
        print(agent_response_creative)

    except Exception as e:
        logger.critical(f"An error occurred during ResearchAgent test: {e}")

