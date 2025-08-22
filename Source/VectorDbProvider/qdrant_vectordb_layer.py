from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import (PointStruct, VectorParams, Distance, SparseVector, NamedVector, NamedSparseVector)
from qdrant_client.http.models import (Filter, FieldCondition, MatchAny, SparseVectorParams)
from VectorDbProvider.base_vectordb_provider import BaseVectorDBProvider
from Database.db_config_loader import load_config_from_db
from Database.db_doc_title import DocTitleDBManager
import logging
import uuid
from langchain_core.documents import Document
from Database.db_vocab import VocabDBManager
from Helper.keyword_vocab import KeywordVocabulary
import nltk
from collections import defaultdict

class QdrantVectorDBProvider(BaseVectorDBProvider):
    def __init__(self, embedding_model, kw_model):
        # Load Qdrant config from DB
        config = load_config_from_db()
        self.url = config['QDRANT']['url']
        self.collection_name = config['QDRANT']['collection_name']
        self.client = QdrantClient(url=self.url)
        self.kw_model = kw_model

        # Check and store the embedding dimension
        try:
            self.embedding_dim = len(embedding_model.embed_query("test"))
        except Exception as e:
            logging.error(f"Embedding dimension check failed: {str(e)}")
            raise ValueError("Invalid embedding model") from e

        self.embedding_provider = embedding_model
        self._ensure_collection_exists()

        # Create LangChain-compatible vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self._create_langchain_embeddings(),
            vector_name="text-dense",
            sparse_vector_name="text-sparse"
        )

    def _ensure_collection_exists(self):
        if self.client.collection_exists(self.collection_name):
            collection = self.client.get_collection(self.collection_name)
            vectors = collection.config.params.vectors

            # Check embedding dimensions match
            current_dim = vectors["text-dense"].size if isinstance(vectors, dict) else vectors.size

            if current_dim != self.embedding_dim:
                logging.warning(f"Embedding dim mismatch: expected={self.embedding_dim}, found={current_dim}")
                self._recreate_hybrid_collection()
            else:
                logging.info(f"✅ Using existing Qdrant collection '{self.collection_name}' with dimension {current_dim}")
        else:
            self._create_collection()

    def _create_collection(self, collection_name=None):
        collection_name = collection_name or self.collection_name
        logging.info(f"📁 Creating Qdrant collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams()
            }
        )

    def _recreate_hybrid_collection(self):
        logging.info(f"🔁 Recreating collection: {self.collection_name}")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams()
            }
        )

    def upsert_vectors(self, collection_name, vectors):
        try:
            if not self.client.collection_exists(collection_name):
                self._create_collection(collection_name)

            points = []
            for vector in vectors:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vector["id"]))
                dense_vector = vector["values"]
                vector_dict = {
                    "text-dense": dense_vector
                }
                sparse_indices = vector.get("sparse_indices", [])
                sparse_values = vector.get("sparse_values", [])

                # ✅ Add sparse vector only if both indices and values are present
                if sparse_indices and sparse_values:
                    vector_dict["text-sparse"] = SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                point_args = {
                    "id": point_id,
                    "vector": vector_dict,
                    "payload": {
                        "page_content": vector["payload"]["page_content"],
                        "sparse_text": vector["payload"].get("sparse_text", ""),
                        "keywords": vector["payload"].get("keywords", []),
                        "metadata": vector["payload"]["metadata"]
                    }
                }

                points.append(PointStruct(**point_args))

            # Perform upsert
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )

            return True

        except Exception as e:
            logging.error(f"Qdrant upsert failed: {str(e)}")
            return False
        
    def search(self, query: str, search=10, top_r=10, filters=None):
        # try:
        #     stop_words = nltk.corpus.stopwords.words("english")
        # except LookupError:
        #     nltk.download("stopwords")
        #     stop_words = nltk.corpus.stopwords.words("english")

        # Convert UI filters to Qdrant filter format
        qdrant_filter = None
        if filters:
            conditions = []
            for filter_value in filters:
                conditions.append(
                    FieldCondition(
                        key="metadata.document_title",
                        match=models.MatchValue(value=filter_value)
                )
            )
            
            qdrant_filter = models.Filter(should=conditions)

        # # Extract keyword candidates
        # keywords = self.kw_model.extract_keywords(
        #     query,
        #     keyphrase_ngram_range=(1, 3),
        #     stop_words=stop_words,
        #     top_n=10
        # )
        # keyword_terms = [kw[0] for kw in keywords]
        # keyword_scores = [kw[1] for kw in keywords]

        # # Load vocabulary for sparse vector generation
        # try:
        #     vocab_dict = VocabDBManager.load_vocab_from_db(self.collection_name)
        #     vocab = KeywordVocabulary()
        #     vocab.load_from_dict(vocab_dict)
        # except Exception as e:
        #     logging.warning(f"No vocab found for collection '{self.collection_name}': {e}")
        #     vocab = None


        sparse_indices = []
        sparse_values = []
        # if vocab:
        #     for kw, score in zip(keyword_terms, keyword_scores):
        #         if kw in vocab.vocab:
        #             sparse_indices.append(vocab.vocab[kw])
        #             sparse_values.append(score)

        # Dense vector query
        dense_vector = self.embedding_provider.embed_query(query)  # List[float]
        results_by_id = defaultdict(lambda: {"score": 0, "payload": None})

        # 1️⃣ Dense vector search
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedVector(name="text-dense", vector=dense_vector),
            query_filter=qdrant_filter,
            limit=search,
            with_payload=True,
        )
        for res in dense_results:
            results_by_id[res.id]["score"] += res.score
            results_by_id[res.id]["payload"] = res.payload

        # 2️⃣ Sparse vector search (if we have valid data)
        if sparse_indices and sparse_values:
            sparse = NamedSparseVector(
                name="text-sparse",
                vector=SparseVector(indices=sparse_indices, values=sparse_values)
            )
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=sparse,
                query_filter=qdrant_filter,
                limit=search,
                with_payload=True
            )
            for res in sparse_results:
                results_by_id[res.id]["score"] += res.score
                results_by_id[res.id]["payload"] = res.payload

        # 3️⃣ Sort by combined score
        ranked_docs = sorted(results_by_id.values(), key=lambda r: r["score"], reverse=True)

        # 4️⃣ Convert to LangChain Documents
        return [
            Document(
            page_content=doc["payload"].get("page_content", ""),
            metadata=doc["payload"].get("metadata", {}) or {}
        )
        for doc in ranked_docs[:top_r]
    ]

# Module-level provider reference
Provider = QdrantVectorDBProvider
