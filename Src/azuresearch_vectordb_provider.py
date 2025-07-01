from langchain_community.vectorstores import AzureSearch
from base_vectordb_provider import BaseVectorDBProvider
import configparser
import logging

class AzureSearchVectorDBProvider(BaseVectorDBProvider):
    def __init__(self, embedding_model):
        config = configparser.ConfigParser()
        config.read('config.ini')
        endpoint = config.get('AZURESEARCH', 'endpoint')
        api_key = config.get('AZURESEARCH', 'api_key')
        index_name = config.get('AZURESEARCH', 'index_name')
        self.vector_store = AzureSearch(
            azure_search_endpoint=endpoint,
            azure_search_key=api_key,
            index_name=index_name,
            embedding_function=embedding_model.embed_query
        )

Provider = AzureSearchVectorDBProvider
