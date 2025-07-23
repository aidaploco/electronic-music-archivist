import json
import logging
import re

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.models.house_dj import HouseDJ
from src.tools.serper_search import SerperSearchTool

logger = logging.getLogger(__name__)


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

            self.search_tool_wrapper = SerperSearchTool()
            self.search_tool = Tool(
                name="Serper Search",
                func=self.search_tool_wrapper.run_search,
                description="A search engine. Useful for when you need to answer questions about current events \
                            or retrieve information from the web. Input should be a concise search query."
            )
            logger.info("Serper Search Tool successfully loaded for agent.")

            self.tools = [self.search_tool]

            self.parser = PydanticOutputParser(pydantic_object=HouseDJ)
            # Get the format instructions and escape curly braces to prevent misinterpretation
            format_instructions = self.parser.get_format_instructions()
            escaped_format_instructions = format_instructions.replace('{', '{{').replace('}', '}}')

            self.agent_prompt = ChatPromptTemplate([
                ("system", """You are an expert electronic music archivist. Your goal is to research and provide
                            comprehensive, accurate information about House DJs and electronic music history.
                            Always cite your sources when you use information obtained via search.
                            If you can't find relevant information, state that clearly.
                            Your final answer MUST be a JSON object conforming to the following schema:\n"""
                            f"{escaped_format_instructions}"),
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

    def run_research(self, query: str) -> HouseDJ:
        """
        Executes a research query using the autonomous agent.
        The agent will decide whether to use the search tool based on the query.

        Args:
            query (str): The research query.

        Returns:
            HouseDJ: The research result as a structured HouseDJ object.
        """
        logger.info(f"Agent received research query: '{query}'")

        try:
            result = self.agent_executor.invoke({"input": query, "chat_history": []})
            raw_output_string = result.get("output", "")

            # Extract JSON block using regex starting with ``` and ending with ```
            json_match = re.search(r"```\s*(.*?)\s*```", raw_output_string, re.DOTALL)

            if json_match:
                json_string = json_match.group(1)
                try:
                    # Attempt to parse the extracted JSON string
                    parsed_output: HouseDJ = self.parser.parse(json_string)
                    logger.info("Agent completed research phase and parsed output to HouseDJ object.")
                    return parsed_output
                except ValidationError as ve:
                    logger.error(f"Pydantic validation failed for extracted JSON. Error: {ve}")
                    raise ValueError(f"Extracted JSON did not conform to HouseDJ schema: {ve}") from ve
                except json.JSONDecodeError as jde:
                    logger.error(f"JSON decoding failed for extracted string. Error: {jde}")
                    raise ValueError(f"Extracted string was not valid JSON: {jde}") from jde
            else:
                error_msg = f"AgentExecutor output did not contain a valid JSON block. Raw output: {raw_output_string}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e: # Catch any exception that propagates
            logger.critical(f"Error during agent research for query '{query}': {e}")
            raise # Re-raise the exception to indicate failure


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting ResearchAgent test with autonomous capabilities...")
    try:
        # Ensure Ollama server is running and 'mistral' model is pulled
        # Ensure SERPER_API_KEY is set in .env

        # Test with Mistral model (default temperature 0.0)
        agent = ResearchAgent(llm_model_name="mistral")

        test_query_dj = "Provide detailed information about the House DJ Frankie Knuckles."
        print(f"\n--- Running agent research query for Frankie Knuckles: '{test_query_dj}' ---")
        dj_info_result = agent.run_research(test_query_dj)

        print("\nAgent's Final Response (Structured HouseDJ Object):")
        print(dj_info_result.model_dump_json(indent=2))
        print("\n--- Testing specific attribute access ---")
        print(f"DJ Name: {dj_info_result.name}")
        if dj_info_result.notable_tracks:
            print(f"Notable Tracks: {', '.join(dj_info_result.notable_tracks[:2])}...")
        if dj_info_result.biography_summary:
            print(f"Biography Summary (first 100 chars): {dj_info_result.biography_summary[:100]}...")

    except Exception as e:
        logger.critical(f"An error occurred during ResearchAgent test: {e}")

