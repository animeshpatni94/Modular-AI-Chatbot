class BaseEmbeddingProvider:
    def generate_embedding(self, text):
        return self.embedding_model.embed_query(text)
    
    def generate_embeddings(self, texts):
        return self.embedding_model.embed_documents(texts)
