from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from base_vectordb_provider import BaseVectorDBProvider
import configparser
import logging

class QdrantVectorDBProvider(BaseVectorDBProvider):
    def __init__(self, embedding_model):
        config = configparser.ConfigParser()
        config.read('config.ini')
        url = config.get('QDRANT', 'url')
        collection_name = config.get('QDRANT', 'collection_name')
        self.client = QdrantClient(url=url)
        
        try:
            embedding_dim = len(embedding_model.embed_query("test"))
        except Exception as e:
            logging.error(f"Embedding dimension check failed: {str(e)}")
            raise ValueError("Invalid embedding model") from e
        
        try:
            if self.client.collection_exists(collection_name):
                collection_info = self.client.get_collection(collection_name)
                if collection_info.config.params.vectors.size != embedding_dim:
                    self.client.recreate_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logging.error(f"Collection setup failed: {str(e)}")
            raise
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_model
        )

Provider = QdrantVectorDBProvider
