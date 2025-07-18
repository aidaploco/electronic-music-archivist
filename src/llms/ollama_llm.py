import logging

from langchain_ollama import OllamaLLM


if __name__ == "__main__":
    # Configure basic logging for direct execution of this script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # This block is for testing the LangChain OllamaLLM directly.
    logger.info("Starting Ollama LLM test (direct LangChain OllamaLLM)...")
    try:
        # Ensure the Ollama server is running in a separate terminal via 'ollama serve'

        # Ensure the 'mistral' model is pulled via 'ollama pull mistral'
        logger.info("\n--- Testing langchain_ollama.OllamaLLM with 'mistral' ---")
        mistral_llm = OllamaLLM(model="mistral")
        prompt_test = "Who were the pioneers of house music?"
        logger.info(f"Sending prompt to mistral OllamaLLM: '{prompt_test}'")
        response_test = mistral_llm.invoke(prompt_test)
        logger.info("Mistral response:")
        print(response_test)

        # Ensure the 'llama3' model is pulled via 'ollama pull llama3'
        logger.info("\n--- Testing OllamaLLM with 'llama3' ---")
        llama3_llm = OllamaLLM(model="llama3")
        prompt_llama3 = "Who were the pioneers of house music?"
        logger.info(f"Sending prompt to llama3 OllamaLLM: '{prompt_llama3}'")
        response_llama3 = llama3_llm.invoke(prompt_llama3)
        logger.info("LLama3 response:")
        print(response_llama3)

    except Exception as e:
        logger.critical(f"An error occurred during LLM test: {e}")
