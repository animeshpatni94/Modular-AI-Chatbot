import os
import re
from typing import List, Dict
from dataclasses import dataclass
from Database.db_config_loader import load_config_from_db
from Helper.provider_manager import ProviderManager


@dataclass
class TextChunk:
    id: str
    page_content: str
    page_number: int
    metadata: Dict[str, str]


class TextFileIngestor:
    def __init__(self, embedding_provider, vectordb_provider):
        self.embedding_provider = embedding_provider
        self.vectordb_provider = vectordb_provider
        self.supported_extensions = {'.cs', '.ns', '.xml', '.sql'}

    def ingest_text_file(self, file_path, collection_name):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if file extension is supported
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_ext}")

            doc_title = self._extract_document_title(file_path)
            chunks = self._extract_text_chunks(file_path, doc_title)
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
                            **chunk.metadata  # includes document_title, source, chunk_id, file_type
                        }
                    }
                })

            success = self.vectordb_provider.upsert_vectors(collection_name, vectors)
            if success:
                print(f"Ingested {len(chunks)} text chunks from {file_path}")
            return success

        except Exception as e:
            print(f"Text file ingestion failed: {str(e)}")
            return False

    def _extract_document_title(self, file_path):
        filename = os.path.basename(file_path)
        # Remove file extension
        return os.path.splitext(filename)[0]

    def _extract_text_chunks(self, file_path, doc_title):
        chunks = []
        chunk_counter = 0
        config = load_config_from_db()
        chunk_size = int(config['TEXT_SPLITTING']['chunk_size'])
        overlap = int(config['TEXT_SPLITTING']['overlap'])
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if not content or not content.strip():
                return chunks
            text_chunks = self._split_text_intelligently(content, chunk_size, overlap, file_ext)
            
            for chunk_text in text_chunks:
                chunk_id = f"{doc_title}_ch{chunk_counter}"
                chunks.append(TextChunk(
                    id=chunk_id,
                    page_content=chunk_text.strip(),
                    page_number=1,  # Text files don't have pages, so we use 1
                    metadata={
                        "document_title": doc_title,
                        "source": chunk_id,
                        "chunk_id": chunk_id,
                        "file_type": file_ext,
                        "file_path": file_path
                    }
                ))
                chunk_counter += 1

        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                text_chunks = self._split_text_intelligently(content, chunk_size, overlap, file_ext)
                
                for chunk_text in text_chunks:
                    chunk_id = f"{doc_title}_ch{chunk_counter}"
                    chunks.append(TextChunk(
                        id=chunk_id,
                        page_content=chunk_text.strip(),
                        page_number=1,
                        metadata={
                            "document_title": doc_title,
                            "source": chunk_id,
                            "chunk_id": chunk_id,
                            "file_type": file_ext,
                            "file_path": file_path
                        }
                    ))
                    chunk_counter += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                
        return chunks

    def _split_text_intelligently(self, text, chunk_size: int, overlap: int, file_ext: str) -> List[str]:
        
        if file_ext == '.cs':
            return self._split_csharp_code(text, chunk_size, overlap)
        elif file_ext == '.sql':
            return self._split_sql_code(text, chunk_size, overlap)
        elif file_ext == '.xml':
            return self._split_xml_content(text, chunk_size, overlap)
        elif file_ext == '.ns':
            return self._split_text(text, chunk_size, overlap)  # Generic splitting for .ns files
        else:
            return self._split_text(text, chunk_size, overlap)

    def _split_csharp_code(self, text, chunk_size: int, overlap: int) -> List[str]:
        class_pattern = r'((?:public|private|protected|internal)?\s*(?:static)?\s*(?:partial)?\s*class\s+\w+)'
        method_pattern = r'((?:public|private|protected|internal)?\s*(?:static)?\s*(?:virtual|override)?\s*\w+\s+\w+\s*\([^)]*\))'
        
        # Split by classes first, then by methods if chunks are still too large
        parts = re.split(class_pattern, text, flags=re.MULTILINE)
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk.split()) + len(part.split()) <= chunk_size:
                    current_chunk += part
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks if chunks else self._split_text(text, chunk_size, overlap)
        
        return self._split_text(text, chunk_size, overlap)

    def _split_sql_code(self, text, chunk_size: int, overlap: int) -> List[str]:
        sql_keywords = r'(CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE|WITH)\s+'
        parts = re.split(sql_keywords, text, flags=re.IGNORECASE | re.MULTILINE)
        
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for i, part in enumerate(parts):
                if len(current_chunk.split()) + len(part.split()) <= chunk_size:
                    current_chunk += part
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks if chunks else self._split_text(text, chunk_size, overlap)
        
        return self._split_text(text, chunk_size, overlap)

    def _split_xml_content(self, text, chunk_size: int, overlap: int) -> List[str]:
        # Try to split at major XML elements
        element_pattern = r'(<[^/][^>]*>)'
        parts = re.split(element_pattern, text)
        
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk.split()) + len(part.split()) <= chunk_size:
                    current_chunk += part
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks if chunks else self._split_text(text, chunk_size, overlap)
        
        return self._split_text(text, chunk_size, overlap)

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


# Run Data Ingestion pipeline for text files
if __name__ == "__main__":
    config = load_config_from_db()
    folder_path = config['DEFAULT']['folder_path']
    collection_name = config['DEFAULT']['ingestion_collection_name']
    manager = ProviderManager()
    embedding_provider = manager.embedding_provider.embedding_model
    vectordb_provider = manager.vectordb_provider
    supported_extensions = {'.cs', '.ns', '.xml', '.sql'}
    text_files = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if os.path.splitext(fname)[1].lower() in supported_extensions
    ]

    for file_path in text_files:
        ingestor = TextFileIngestor(embedding_provider, vectordb_provider)
        print(f"Ingesting: {file_path}")
        success = ingestor.ingest_text_file(file_path, collection_name)
        if success:
            print(f"Successfully ingested {file_path} into collection {collection_name}")
        else:
            print(f"Failed to ingest {file_path}")
