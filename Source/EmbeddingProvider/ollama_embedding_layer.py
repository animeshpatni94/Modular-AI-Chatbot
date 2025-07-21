from Database.db_config_loader import load_config_from_db
from langchain_ollama.embeddings import OllamaEmbeddings
from EmbeddingProvider.base_embedding_provider import BaseEmbeddingProvider

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        config = load_config_from_db()
        model = config['OLLAMA_EMBED']['model']
        base_url = config['OLLAMA_EMBED']['base_url']
        self.embedding_model = OllamaEmbeddings(model=model, base_url=base_url)

Provider = OllamaEmbeddingProvider
