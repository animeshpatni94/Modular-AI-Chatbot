from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List


class BaseVectorDBProvider:
    def _create_langchain_embeddings(self):
        return self.embedding_provider

    def store_embeddings(self, texts, metadatas):
        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        return self.vector_store.add_documents(documents=documents)

    def search(self, query, k, filters=None):
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filters)

    def delete(self, ids: List[str]) -> None:
        self.vector_store.delete(ids)

    def upsert_vectors(self, collection_name, vectors):
        raise NotImplementedError("upsert_vectors must be implemented in subclass.")

    def FormatFilter(self, keyword_strings):
        raise NotImplementedError("FormatFilter must be implemented in subclass.")
