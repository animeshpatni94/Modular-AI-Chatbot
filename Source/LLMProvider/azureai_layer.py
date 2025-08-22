from langchain_openai import AzureChatOpenAI
from Database.db_config_loader import load_config_from_db
from LLMProvider.base_llm_provider import BaseLLMProvider

class AzureProvider(BaseLLMProvider):
    def __init__(self):
        config = load_config_from_db()
        azure_deployment = config['AZUREAI']['azure_deployment']
        api_version = config['AZUREAI']['api_version']
        azure_endpoint = config['AZUREAI']['azure_endpoint']
        openai_api_key = config['AZUREAI']['openai_api_key']
        chat_llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_key=openai_api_key,
            max_tokens=4096,        # Increased for detailed legal analysis
            temperature=0.0,        # Set to 0 for maximum determinism
            top_p=1.0,             # Standard setting for consistency
            frequency_penalty=0.0,  # No penalty to maintain exact legal language
            presence_penalty=0.0,   # No penalty to preserve precise terminology
            stop=None              # No stop sequences for complete responses
        ) 
        super().__init__(chat_llm)

Provider = AzureProvider
