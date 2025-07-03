from sentence_transformers import CrossEncoder

class BgeReranker:
    def __init__(self, model_path):
        self.model = CrossEncoder(model_path)

    def rerank(self, query, documents, top_n):
        if not documents:
            return []
        doc_texts = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, str):
                doc_texts.append(doc)
            else:
                doc_texts.append(str(doc))
        pairs = [[query, doc_text] for doc_text in doc_texts]
        scores = self.model.predict(pairs)
        sorted_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return sorted_docs[:top_n]
