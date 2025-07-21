from Database.db_config_loader import load_config_from_db
from langchain_openai import AzureOpenAIEmbeddings
from EmbeddingProvider.base_embedding_provider import BaseEmbeddingProvider

class AzureAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        config = load_config_from_db()
        azure_deployment = config['AZUREAI_EMBED']['azure_deployment']
        api_version = config['AZUREAI_EMBED']['api_version']
        azure_endpoint = config['AZUREAI_EMBED']['azure_endpoint']
        api_key = config['AZUREAI_EMBED']['api_key']
        
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            max_retries=3
        )

Provider = AzureAIEmbeddingProvider
