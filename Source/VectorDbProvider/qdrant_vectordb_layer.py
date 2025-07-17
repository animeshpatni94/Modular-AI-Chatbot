from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance 
from VectorDbProvider.base_vectordb_provider import BaseVectorDBProvider
import configparser
import logging
import uuid
from qdrant_client.http import models

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


    def upsert_vectors(self, collection_name, vectors):
        try:
            if not self.client.collection_exists(collection_name):
                self._create_collection(collection_name, len(vectors[0]["values"]))
        
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, vector["id"])),
                    vector=vector["values"],
                    payload={
                        "page_content": vector["payload"]["page_content"],  # or "content"
                        "metadata": {k: v for k, v in vector["payload"]["metadata"].items()}
                    }
                )
            for vector in vectors
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            return True
        except Exception as e:
            print(f"Qdrant upsert failed: {str(e)}")
            return False

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    
    def FormatFilter(self, keyword_strings):
        filters = models.Filter(must=[models.FieldCondition(key="keywords",match=models.MatchAny(any=keyword_strings))])
        return filters


Provider = QdrantVectorDBProvider
