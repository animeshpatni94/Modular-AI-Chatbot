from langchain_google_genai import ChatGoogleGenerativeAI
from Database.db_config_loader import load_config_from_db
from LLMProvider.base_llm_provider import BaseLLMProvider


class GoogleAIProvider(BaseLLMProvider):
    def __init__(self):
        config = load_config_from_db()
        chat_llm = ChatGoogleGenerativeAI(
            model=config['GOOGLEAI']['model_name'],
            google_api_key=config['GOOGLEAI']['google_api_key'],
            temperature=0.5,
            max_tokens=4096
        )
        
        super().__init__(chat_llm)


Provider = GoogleAIProvider
