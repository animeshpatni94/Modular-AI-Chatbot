import configparser
from langchain_ollama.embeddings import OllamaEmbeddings
from base_embedding_provider import BaseEmbeddingProvider

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        model = config.get('OLLAMA_EMBED', 'model')
        base_url = config.get('OLLAMA_EMBED', 'base_url')
        self.embedding_model = OllamaEmbeddings(model=model, base_url=base_url)

Provider = OllamaEmbeddingProvider
