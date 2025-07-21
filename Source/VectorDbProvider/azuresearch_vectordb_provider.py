from langchain_community.vectorstores import AzureSearch
from VectorDbProvider.base_vectordb_provider import BaseVectorDBProvider
from Database.db_config_loader import load_config_from_db
import logging

class AzureSearchVectorDBProvider(BaseVectorDBProvider):
    def __init__(self, embedding_model):
        config = load_config_from_db()
        endpoint = config['AZURESEARCH']['endpoint']
        api_key = config['AZURESEARCH']['api_key']
        index_name = config['AZURESEARCH']['index_name']
        self.vector_store = AzureSearch(
            azure_search_endpoint=endpoint,
            azure_search_key=api_key,
            index_name=index_name,
            embedding_function=embedding_model.embed_query
        )

Provider = AzureSearchVectorDBProvider
