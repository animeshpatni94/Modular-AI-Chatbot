from langchain_community.vectorstores import AzureSearch
from VectorDbProvider.base_vectordb_provider import BaseVectorDBProvider
from Database.db_config_loader import load_config_from_db
from langchain_core.documents import Document
import nltk

class AzureSearchVectorDBProvider(BaseVectorDBProvider):
    def __init__(self, embedding_model, kw_model):
        config = load_config_from_db()
        endpoint = config['AZURESEARCH']['endpoint']
        api_key = config['AZURESEARCH']['api_key']
        index_name = config['AZURESEARCH']['index_name']
        self.kw_model = kw_model
        self.embedding_provider = embedding_model
        self.vector_store = AzureSearch(
            azure_search_endpoint=endpoint,
            azure_search_key=api_key,
            index_name=index_name,
            embedding_function=self._create_langchain_embeddings().embed_query
        )

    def upsert_vectors(self, collection_name, vectors):
        try:
            docs = []
            for vec in vectors:
                doc = {
                    "id": vec["id"],
                    "contentVector": vec["values"],
                    "page_content": vec["payload"]["page_content"],
                    "keywords": vec["payload"].get("keywords", []),
                    "metadata": vec["payload"]["metadata"]
                }
                docs.append(doc)
            self.vector_store.add_documents(docs)
            return True
        except Exception as e:
            return False
    
    def search(self, query, search=10, top_r=10, filters=None):
        try:
            stop_words = nltk.corpus.stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
            stop_words = nltk.corpus.stopwords.words("english")

        keywords = self.kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 3), stop_words=stop_words, top_n=10)
        keyword_text = " ".join([kw[0] for kw in keywords]) if keywords else query

        embedding = self.embedding_provider.embed_query(query)
        raw_results = self.vector_store.client.search(
            index_name=self.vector_store.index_name,
            search=keyword_text,
            vector_queries=[{
                "kind": "vector",
                "vector": embedding,
                "k": search,
                "fields": "contentVector"
            }],
            top=search
        )
        return [
            Document(
                page_content=doc["@search.document"]["page_content"],
                metadata=doc["@search.document"].get("metadata", {})
            )
            for doc in raw_results.get("value", [])
        ]

Provider = AzureSearchVectorDBProvider
