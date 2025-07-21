from langchain_ollama import ChatOllama
from Database.db_config_loader import load_config_from_db
from LLMProvider.base_llm_provider import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        config = load_config_from_db()
        model = config['OLLAMA']['model']
        base_url = config['OLLAMA']['base_url']
        chat_llm = ChatOllama(model=model, base_url=base_url, temperature=0.7, num_ctx=3000)
        super().__init__(chat_llm)

Provider = OllamaProvider
