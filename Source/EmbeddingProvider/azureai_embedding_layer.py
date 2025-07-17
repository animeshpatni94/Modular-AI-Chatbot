import configparser
from langchain_openai import AzureOpenAIEmbeddings
from EmbeddingProvider.base_embedding_provider import BaseEmbeddingProvider

class AzureAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        azure_deployment = config.get('AZUREAI_EMBED', 'azure_deployment')
        api_version = config.get('AZUREAI_EMBED', 'api_version')
        azure_endpoint = config.get('AZUREAI_EMBED', 'azure_endpoint')
        api_key = config.get('AZUREAI_EMBED', 'api_key')
        
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            max_retries=3
        )

Provider = AzureAIEmbeddingProvider
