from langchain_openai import AzureChatOpenAI
import configparser
from base_llm_provider import BaseLLMProvider

class AzureProvider(BaseLLMProvider):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        azure_deployment = config.get('AZUREAI', 'azure_deployment')
        api_version = config.get('AZUREAI', 'api_version')
        azure_endpoint = config.get('AZUREAI', 'azure_endpoint')
        openai_api_key = config.get('AZUREAI', 'openai_api_key')
        chat_llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_key=openai_api_key,
            max_tokens=4096,
            temperature=0.5 
        ) 
        super().__init__(chat_llm)

Provider = AzureProvider
