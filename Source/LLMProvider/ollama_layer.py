from langchain_ollama import ChatOllama
import configparser
from LLMProvider.base_llm_provider import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        model = config.get('OLLAMA', 'model')
        base_url = config.get('OLLAMA', 'base_url')
        chat_llm = ChatOllama(model=model, base_url=base_url, temperature=0.7, num_ctx=3000)
        super().__init__(chat_llm)

Provider = OllamaProvider
