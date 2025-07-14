from langchain_google_genai import ChatGoogleGenerativeAI
import configparser
from base_llm_provider import BaseLLMProvider


class GoogleAIProvider(BaseLLMProvider):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        chat_llm = ChatGoogleGenerativeAI(
            model=config.get('GOOGLEAI', 'model_name'),
            google_api_key=config.get('GOOGLEAI', 'google_api_key'),
            temperature=0.5,
            max_tokens=4096
        )
        
        super().__init__(chat_llm)


Provider = GoogleAIProvider
