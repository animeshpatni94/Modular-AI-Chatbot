import os
import re
from typing import List, Dict
from dataclasses import dataclass
import pdfplumber
from Helper.provider_manager import ProviderManager
#from keybert import KeyBERT
import configparser

@dataclass
class TextChunk:
    id: str
    page_content: str
    page_number: int
    metadata: Dict[str, str]

class PDFTextIngestor:
    def __init__(self, embedding_provider, vectordb_provider):
        self.embedding_provider = embedding_provider
        self.vectordb_provider = vectordb_provider

    def ingest_pdf(self, pdf_path, collection_name):
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            doc_title = self._extract_document_title(pdf_path)
            chunks = self._extract_text_chunks(pdf_path, doc_title)
            embeddings = self._generate_embeddings([chunk.page_content for chunk in chunks])

            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": chunk.id,
                    "values": embedding,
                    "payload": {
                        "page_content": chunk.page_content,
                        "metadata": {
                            "page": chunk.page_number,
                            **chunk.metadata  # includes document_title, source, chunk_id, keywords
                        }
                    }
                })

            success = self.vectordb_provider.upsert_vectors(collection_name, vectors)
            if success:
                print(f"Ingested {len(chunks)} text chunks from {pdf_path}")
            return success

        except Exception as e:
            print(f"PDF ingestion failed: {str(e)}")
            return False

    def _extract_document_title(self, pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata
                if metadata.get('Title'):
                    return metadata['Title'].strip()
        except:
            pass
        filename = os.path.basename(pdf_path)
        return re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)

    def _extract_text_chunks(self, pdf_path, doc_title):
        chunks = []
        chunk_counter = 0
        config = configparser.ConfigParser()
        config.read('config.ini') 
        chunk_size = int(config['TEXT_SPLITTING']['chunk_size']) 
        overlap = int(config['TEXT_SPLITTING']['overlap']) 
        #kw_model = KeyBERT()
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                page_chunks = self._split_text(text, chunk_size, overlap)
                for chunk_text in page_chunks:
                    chunk_id = f"{doc_title}_pg{page_num}_ch{chunk_counter}"
                    chunks.append(TextChunk(
                        id=chunk_id,
                        page_content=chunk_text.strip(),
                        page_number=page_num,
                        metadata={
                            "document_title": doc_title,
                            "source": chunk_id,
                            "chunk_id": chunk_id
                            #"keywords": kw_model.extract_keywords(chunk_text.strip(), keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
                        }
                    ))
                    chunk_counter += 1
        return chunks

    def _split_text(self, text, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        prev_start = -1  
    
        while start < len(words):
            if start == prev_start:
                break
            prev_start = start  
        
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        
            if start < 0:
                start = 0
        return chunks


    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.embedding_provider.embed_documents(texts)
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            raise

#Run Data Ingestion pipeline
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    folder_path = config['DEFAULT']['folder_path']
    collection_name = config['DEFAULT']['ingestion_collection_name']
    manager = ProviderManager()
    embedding_provider = manager.embedding_provider.embedding_model
    vectordb_provider = manager.vectordb_provider
    pdf_files = [
    os.path.join(folder_path, fname)
    for fname in os.listdir(folder_path)
    if fname.lower().endswith('.pdf')
    ]

    for pdf_path in pdf_files:
        ingestor = PDFTextIngestor(embedding_provider, vectordb_provider)
        print(f"Ingesting: {pdf_path}")
        success = ingestor.ingest_pdf(pdf_path, collection_name)
        if success:
            print(f"Successfully ingested {pdf_path} into collection {collection_name}")
        else:
            print(f"Failed to ingest {pdf_path}")