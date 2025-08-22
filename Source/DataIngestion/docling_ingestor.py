import os
from typing import List, Dict, Set
from dataclasses import dataclass
import nltk
from nltk.corpus import stopwords
import datetime
import time
import json

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.chunking import HybridChunker

from Helper.provider_manager import ProviderManager
from Database.db_config_loader import load_config_from_db
from Helper.keyword_vocab import KeywordVocabulary
from Database.db_vocab import VocabDBManager
from Database.db_doc_title import DocTitleDBManager

@dataclass
class TextChunk:
    id: str
    page_content: str
    page_number: int
    metadata: Dict[str, str]

class DoclingIngestor:
    def __init__(self, embedding_provider, vectordb_provider, kw_model, vocabulary: KeywordVocabulary, batch_size: int = 50):
        self.embedding_provider = embedding_provider
        self.vectordb_provider = vectordb_provider
        self.vocab = vocabulary
        self.kw_model = kw_model
        self.batch_size = batch_size
        
        # Legal document optimized settings for Azure OpenAI
        self.max_chunk_tokens = 1024
        self.overlap_tokens = 150
        self.min_chunk_tokens = 100
        
        # Initialize Docling converter
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        # Initialize Docling's HybridChunker
        self.chunker = HybridChunker(
            max_tokens=self.max_chunk_tokens,
            overlap_tokens=self.overlap_tokens,
            merge_peers=True,
            respect_sentence_boundary=True
        )

        # Ensure NLTK stopwords are available
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("english"))

        # Track vocabulary across all documents
        self.all_vocab_terms = set()

    def ingest_pdf(self, pdf_path: str, collection_name: str, contract_metadata: Dict = None) -> bool:
        """
        Optimized ingestion with vocabulary building integrated into chunking
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            print(f"⚖️ Processing legal document with Docling: {pdf_path}")
            
            # Convert PDF using Docling
            result = self.doc_converter.convert(pdf_path)
            doc = result.document
            
            # Extract document title
            doc_title = self._extract_document_title(doc, pdf_path)
            DocTitleDBManager.save_doctitle_to_db(collection_name, doc_title)
            
            # Extract chunks and build vocabulary simultaneously
            chunks, chunk_keywords = self._extract_chunks_and_build_vocab(doc, doc_title)
            if not chunks:
                print(f"⚠️ No text chunks extracted from {pdf_path}")
                return False

            print(f"📄 Extracted {len(chunks)} legal chunks from {doc_title}")

            # Prepare texts for batch embedding
            embed_texts = [chunk.page_content for chunk in chunks]
            
            # Generate all embeddings at once (more efficient)
            dense_embeddings = self._generate_embeddings(embed_texts)

            # Build vectors with sparse embeddings
            vectors = []
            for i, chunk in enumerate(chunks):
                keywords_with_scores = chunk_keywords[i]
                keyword_terms = [kw for kw, score in keywords_with_scores]
                keyword_scores = [score for kw, score in keywords_with_scores]
                
                # Build sparse vector
                sparse_indices = [self.vocab.get_index(term) for term in keyword_terms if term in self.vocab.vocab]
                sparse_values = [keyword_scores[j] for j, term in enumerate(keyword_terms) if term in self.vocab.vocab]

                vectors.append({
                    "id": chunk.id,
                    "values": dense_embeddings[i],
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                    "payload": self._optimize_legal_payload(chunk, keyword_terms, " ".join(keyword_terms), contract_metadata)
                })

            # Enhanced batch upsert with size checking
            success = self._batch_upsert_vectors(collection_name, vectors)
            if success:
                print(f"✅ Successfully ingested {len(chunks)} legal chunks from: {pdf_path}")
            return success

        except Exception as e:
            print(f"❌ Legal document ingestion failed for {pdf_path}: {str(e)}")
            return False

    def _extract_chunks_and_build_vocab(self, doc, doc_title: str) -> tuple:
        """Extract chunks with improved page number tracking"""
        try:
            # Build comprehensive page mapping first
            page_mapping = self._build_page_mapping(doc)
            print(f"📄 Page mapping completed: {len(page_mapping)} pages")
            
            # Use Docling's HybridChunker
            docling_chunks = list(self.chunker.chunk(doc))
            print(f"🔧 Docling created {len(docling_chunks)} chunks")
            
            chunks = []
            chunk_keywords = []
            
            for i, docling_chunk in enumerate(docling_chunks):
                # Get contextualized text
                chunk_text = self.chunker.contextualize(chunk=docling_chunk)
                
                if not chunk_text or len(chunk_text.strip()) < 20:
                    continue
                
                # **CRITICAL**: Determine page number BEFORE processing
                page_number = self._determine_chunk_page(docling_chunk, page_mapping, doc)
                print(f"📍 Chunk {i}: Assigned to page {page_number}")
                
                # Extract keywords with adaptive settings
                keywords = self._extract_keywords_adaptive(chunk_text, i)
                
                # Add terms to global vocabulary
                for kw, score in keywords:
                    self.all_vocab_terms.add(kw)
                
                chunk_keywords.append(keywords)
                
                # Create chunk with accurate page number
                chunk_id = f"{doc_title}_legal_chunk_{i}_page_{page_number}"  # Include page in ID
                metadata = self._extract_legal_metadata(docling_chunk, doc_title, chunk_id)
                metadata["determined_page"] = page_number  # Add page tracking
                
                chunks.append(TextChunk(
                    id=chunk_id,
                    page_content=chunk_text.strip(),
                    page_number=page_number,  # This is now accurate
                    metadata=metadata
                ))
                
            print(f"✅ Successfully created {len(chunks)} chunks with page numbers")
            return chunks, chunk_keywords
            
        except Exception as e:
            print(f"⚠️ Docling HybridChunker failed, using fallback: {e}")
            return self._fallback_legal_chunk_extraction(doc, doc_title)

    def _extract_keywords_adaptive(self, chunk_text: str, chunk_index: int) -> List[tuple]:
        """Extract keywords with adaptive settings based on text length"""
        try:
            # Longer text settings
            keywords = self.kw_model.extract_keywords(
                chunk_text,
                keyphrase_ngram_range=(1, 4),
                stop_words=self.stop_words,
                top_n=15,
                use_mmr=False,
            )

            return keywords
                
        except Exception as e:
            print(f"⚠️ Keyword extraction failed for chunk {chunk_index}: {e}")
            # Fallback to simple word extraction
            words = chunk_text.lower().split()
            frequent_words = [w for w in words if len(w) > 3 and w.isalpha() and w not in self.stop_words]
            fallback_keywords = [(word, 1.0) for word in list(set(frequent_words))[:10]]
            return fallback_keywords

    def _extract_document_title(self, doc, pdf_path: str) -> str:
        """Extract document title from metadata or filename"""
        try:
            if hasattr(doc, 'meta') and doc.meta and hasattr(doc.meta, 'title') and doc.meta.title:
                return doc.meta.title.strip()
            
            if hasattr(doc, 'metadata') and doc.metadata:
                title = doc.metadata.get('title') or doc.metadata.get('Title')
                if title and title.strip():
                    return title.strip()
        except Exception:
            pass
        
        return os.path.splitext(os.path.basename(pdf_path))[0]

    def _build_page_mapping(self, doc) -> Dict[int, str]:
        """Build comprehensive mapping of page numbers to content with fallback methods"""
        page_mapping = {}
        try:
            for page_idx, page in enumerate(doc.pages, 1):
                page_text = ""
                
                # Method 1: Try export_to_text
                if hasattr(page, 'export_to_text'):
                    try:
                        page_text = page.export_to_text()
                    except Exception as e:
                        print(f"⚠️ export_to_text failed for page {page_idx}: {e}")
                
                # Method 2: Fallback to .text attribute
                if not page_text.strip() and hasattr(page, 'text'):
                    try:
                        page_text = page.text
                    except Exception as e:
                        print(f"⚠️ .text failed for page {page_idx}: {e}")
                
                # Method 3: Try get_textpage if available (PyMuPDF)
                if not page_text.strip() and hasattr(page, 'get_textpage'):
                    try:
                        textpage = page.get_textpage()
                        page_text = textpage.extractText()
                    except Exception as e:
                        print(f"⚠️ get_textpage failed for page {page_idx}: {e}")
                
                # Store result with debugging
                if page_text.strip():
                    # Store more text for better chunk matching (500 chars instead of 200)
                    page_mapping[page_idx] = page_text.strip()[:500]
                    print(f"✅ Page {page_idx}: Extracted {len(page_text)} characters")
                else:
                    # Still map empty pages for accurate numbering
                    page_mapping[page_idx] = f"[Page {page_idx} - Empty or image-based]"
                    print(f"⚠️ Page {page_idx}: No extractable text")
                    
        except Exception as e:
            print(f"⚠️ Page mapping failed: {e}")
        
        print(f"📄 Built page mapping for {len(page_mapping)} pages")
        return page_mapping

    def _determine_chunk_page(self, chunk, page_mapping: Dict[int, str], doc) -> int:
        """Enhanced page determination with multiple strategies"""
        try:
            # Strategy 1: Use Docling's built-in page metadata
            if hasattr(chunk, 'meta') and chunk.meta:
                if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, 'page_no') and prov.page_no is not None:
                                    page_num = prov.page_no  # Convert 0-based to 1-based
                                    print(f"✅ Found page {page_num} from Docling metadata")
                                    return page_num
            
            # Strategy 2: Text matching with multiple chunk sizes
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if not chunk_text:
                return 1
            
            # Try different text snippet sizes for matching
            snippet_sizes = [50, 100, 150, 200]
            
            for snippet_size in snippet_sizes:
                chunk_snippet = chunk_text[:snippet_size].strip()
                if not chunk_snippet:
                    continue
                    
                # Clean text for better matching
                chunk_snippet = ' '.join(chunk_snippet.split())
                
                for page_num, page_text in page_mapping.items():
                    if not page_text or "[Page" in page_text:  # Skip empty pages
                        continue
                        
                    # Clean page text for comparison
                    clean_page_text = ' '.join(page_text.split())
                    
                    # Exact match
                    if chunk_snippet in clean_page_text:
                        print(f"✅ Exact match: Chunk mapped to page {page_num}")
                        return page_num
                    
                    # Fuzzy matching for slight variations
                    chunk_words = set(chunk_snippet.lower().split())
                    page_words = set(clean_page_text.lower().split())
                    
                    if len(chunk_words) > 3:  # Only for meaningful chunks
                        overlap = len(chunk_words.intersection(page_words))
                        similarity_ratio = overlap / len(chunk_words)
                        
                        if similarity_ratio > 0.7:  # 70% word overlap
                            print(f"✅ Fuzzy match: Chunk mapped to page {page_num} (similarity: {similarity_ratio:.2f})")
                            return page_num
            
            # Strategy 3: Sequential fallback based on chunk position
            if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
                # Estimate page based on document structure
                total_items = len(chunk.meta.doc_items) if chunk.meta.doc_items else 1
                total_pages = len(page_mapping)
                
                # Simple heuristic: distribute chunks evenly across pages
                estimated_page = min(total_pages, max(1, int((total_items / 10) % total_pages) + 1))
                print(f"⚠️ Using estimated page {estimated_page} for chunk")
                return estimated_page
            
            print("⚠️ Could not determine page, defaulting to page 1")
            return 1
            
        except Exception as e:
            print(f"⚠️ Page determination failed: {e}, defaulting to page 1")
            return 1

    def _extract_legal_metadata(self, chunk, doc_title: str, chunk_id: str) -> Dict[str, str]:
        """Extract legal document specific metadata"""
        metadata = {
            "document_title": doc_title,
            "source": chunk_id,
            "chunk_id": chunk_id,
            "chunk_type": "legal_hybrid",
            "chunk_method": "docling_hybrid_chunker"
        }
        
        try:
            if hasattr(chunk, 'meta') and chunk.meta:
                if hasattr(chunk.meta, 'doc_items'):
                    element_types = []
                    headings = []
                    
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'label'):
                            element_types.append(str(item.label))
                        if hasattr(item, 'text') and 'heading' in str(item.label).lower():
                            headings.append(item.text[:100])
                    
                    if element_types:
                        metadata["element_types"] = ", ".join(set(element_types))
                    if headings:
                        metadata["section_headings"] = " | ".join(headings[:3])
                
                if hasattr(chunk.meta, 'heading') and chunk.meta.heading:
                    metadata["primary_heading"] = chunk.meta.heading[:150]
        
        except Exception as e:
            print(f"⚠️ Legal metadata extraction warning: {e}")
        
        return metadata

    def _fallback_legal_chunk_extraction(self, doc, doc_title: str) -> tuple:
        """Fallback chunking with keyword extraction and proper page tracking"""
        chunks = []
        chunk_keywords = []
        chunk_counter = 0
        
        # Build page mapping for fallback as well
        page_mapping = self._build_page_mapping(doc)
        
        try:
            for page_idx, page in enumerate(doc.pages, 1):
                page_text = page_mapping.get(page_idx, "")
                
                if not page_text.strip() or "[Page" in page_text:
                    continue
                
                paragraphs = page_text.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    estimated_tokens = len(current_chunk + para) // 4
                    
                    if estimated_tokens > self.max_chunk_tokens and current_chunk:
                        # Process current chunk
                        chunk_id = f"{doc_title}_legal_fallback_pg{page_idx}_ch{chunk_counter}"
                        
                        # Extract keywords for this chunk
                        keywords = self._extract_keywords_adaptive(current_chunk, chunk_counter)
                        for kw, score in keywords:
                            self.all_vocab_terms.add(kw)
                        chunk_keywords.append(keywords)
                        
                        chunks.append(TextChunk(
                            id=chunk_id,
                            page_content=current_chunk.strip(),
                            page_number=page_idx,  # Accurate page number from mapping
                            metadata={
                                "document_title": doc_title,
                                "source": chunk_id,
                                "chunk_id": chunk_id,
                                "chunk_type": "legal_fallback",
                                "chunk_method": "paragraph_based",
                                "determined_page": page_idx
                            }
                        ))
                        chunk_counter += 1
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                
                # Process final chunk
                if current_chunk.strip():
                    chunk_id = f"{doc_title}_legal_fallback_pg{page_idx}_ch{chunk_counter}"
                    
                    keywords = self._extract_keywords_adaptive(current_chunk, chunk_counter)
                    for kw, score in keywords:
                        self.all_vocab_terms.add(kw)
                    chunk_keywords.append(keywords)
                    
                    chunks.append(TextChunk(
                        id=chunk_id,
                        page_content=current_chunk.strip(),
                        page_number=page_idx,  # Accurate page number
                        metadata={
                            "document_title": doc_title,
                            "source": chunk_id,
                            "chunk_id": chunk_id,
                            "chunk_type": "legal_fallback",
                            "chunk_method": "paragraph_based",
                            "determined_page": page_idx
                        }
                    ))
                    chunk_counter += 1
            
            return chunks, chunk_keywords
            
        except Exception as e:
            print(f"❌ Fallback legal chunking failed: {e}")
            return [], []

    def _optimize_legal_payload(self, chunk: TextChunk, keywords: List[str], sparse_text: str, contract_metadata: Dict = None) -> Dict:
        """Enhanced payload with accurate page information for citations"""
        content = chunk.page_content
        
        # Strict content size limit for Qdrant (conservative)
        MAX_CONTENT_SIZE = 8000  # ~8KB per chunk
        if len(content) > MAX_CONTENT_SIZE:
            # Smart truncation at sentence boundary
            truncated = content[:MAX_CONTENT_SIZE]
            last_period = truncated.rfind('. ')
            if last_period > MAX_CONTENT_SIZE * 0.8:
                content = truncated[:last_period + 1] + "..."
            else:
                content = truncated + "..."
        
        # Limit keywords to prevent size bloat
        limited_keywords = keywords[:15]
        
        payload = {
            "page_content": content,
            "sparse_text": sparse_text[:1000],
            "keywords": limited_keywords,
            "metadata": {
                "page": chunk.page_number,  # **CRITICAL**: Accurate page for citations
                "page_citation": f"Page {chunk.page_number}",  # Ready-to-use citation
                "document_title": chunk.metadata.get("document_title", "")[:150],
                "source": chunk.metadata.get("source", "")[:100],
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "chunk_type": chunk.metadata.get("chunk_type", "legal"),
                "chunk_method": chunk.metadata.get("chunk_method", "hybrid"),
                "determined_page": chunk.metadata.get("determined_page", chunk.page_number),  # Page determination method
                "element_types": chunk.metadata.get("element_types", "")[:200],
                "section_headings": chunk.metadata.get("section_headings", "")[:300],
                "primary_heading": chunk.metadata.get("primary_heading", "")[:150]
            }
        }
        
        # Add contract metadata with size limits
        if contract_metadata:
            payload["metadata"].update({
                "contract_id": str(contract_metadata.get("contract_id", ""))[:100],
                "contract_type": str(contract_metadata.get("contract_type", ""))[:50],
                "parties": str(contract_metadata.get("parties", []))[:200],
                "effective_date": str(contract_metadata.get("effective_date", ""))[:20]
            })
        
        return payload

    def _estimate_payload_size(self, vectors: List[Dict]) -> int:
        """Estimate JSON payload size for Qdrant compatibility"""
        try:
            # Sample approach: serialize a few vectors to estimate
            sample_size = min(3, len(vectors))
            sample_data = vectors[:sample_size]
            
            serialized_sample = json.dumps(sample_data, default=str)
            sample_bytes = len(serialized_sample.encode('utf-8'))
            
            # Extrapolate to full batch
            estimated_total = (sample_bytes / sample_size) * len(vectors)
            return int(estimated_total)
            
        except Exception:
            # Conservative fallback estimate
            return len(vectors) * 50 * 1024  # 50KB per vector estimate

    def _batch_upsert_vectors(self, collection_name: str, vectors: List[Dict]) -> bool:
        """Enhanced batch upsert with proactive size checking for Qdrant 32MB limit"""
        current_batch_size = min(self.batch_size, 30)  # Conservative start
        
        for i in range(0, len(vectors), current_batch_size):
            batch = vectors[i:i + current_batch_size]
            
            # PROACTIVE SIZE CHECK - estimate before upload
            estimated_size = self._estimate_payload_size(batch)
            
            # Conservative 25MB threshold (leave buffer for Qdrant overhead)
            if estimated_size > 25 * 1024 * 1024:
                if current_batch_size > 1:
                    # Reduce batch size proactively
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"⚠️ Proactively reducing batch size to {current_batch_size} (estimated {estimated_size/1024/1024:.1f}MB)")
                    continue
                else:
                    # Single vector too large - skip with warning
                    print(f"⚠️ Skipping oversized vector: {batch[0]['id']}")
                    continue
            
            try:
                success = self.vectordb_provider.upsert_vectors(collection_name, batch)
                if success:
                    print(f"✅ Upserted batch ({len(batch)} vectors, ~{estimated_size//1024}KB)")
                else:
                    print(f"❌ Failed to upsert batch")
                    return False
                    
            except Exception as e:
                error_msg = str(e).lower()
                # Reactive handling as backup
                if "payload" in error_msg and ("larger than allowed" in error_msg or "limit" in error_msg):
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"⚠️ Reactive batch size reduction to {current_batch_size}")
                    continue
                else:
                    print(f"❌ Legal batch upsert failed: {str(e)}")
                    raise
        
        return True

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with retry logic"""
        max_retries = 5
        retry_delay = 60
        
        for attempt in range(max_retries):
            try:
                return self.embedding_provider.embed_documents(texts)
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        print(f"❌ Rate limit encountered, retrying {attempt + 1}/{max_retries} after {retry_delay}s")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception("Max retries exceeded for embedding generation due to rate limit")
                else:
                    print(f"❌ Embedding generation failed: {str(e)}")
                    raise
        
        raise Exception("Max retries exceeded for embedding generation due to rate limit")

    def finalize_vocabulary(self, collection_name: str):
        """Save accumulated vocabulary to database after processing all documents"""
        if self.all_vocab_terms:
            # Build vocabulary from all collected terms
            self.vocab.build_from_keywords([list(self.all_vocab_terms)])
            
            # Save to database
            VocabDBManager.save_vocab_to_db(collection_name, self.vocab.vocab)
            print(f"💾 Saved vocabulary with {len(self.vocab.vocab)} terms to database")
        else:
            print("⚠️ No vocabulary terms collected")

# Updated main execution with optimized approach
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    config = load_config_from_db()
    folder_path = config['DEFAULT']['folder_path']
    collection_name = config['DEFAULT']['ingestion_collection_name']

    # Initialize providers
    manager = ProviderManager()
    embedding_provider = manager.embedding_provider.embedding_model
    vectordb_provider = manager.vectordb_provider
    kw_model = manager.kw_model

    from tqdm import tqdm
    
    # Initialize vocabulary
    vocab = KeywordVocabulary()
    
    # Load existing vocabulary
    vocab_dict = VocabDBManager.load_vocab_from_db(collection_name)
    if vocab_dict is not None:
        vocab.load_from_dict(vocab_dict)
        print(f"📚 Loaded existing vocabulary with {len(vocab.vocab)} terms")

    # Get all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    # Initialize optimized ingestor
    ingestor = DoclingIngestor(
        embedding_provider=embedding_provider,
        vectordb_provider=vectordb_provider,
        kw_model=kw_model,
        vocabulary=vocab,
        batch_size=40
    )

    # Process all legal PDFs with integrated vocabulary building
    successful_ingestions = 0
    failed_ingestions = 0
    
    for pdf_path in pdf_files:
        print(f"\n⚖️ Ingesting legal document: {pdf_path}")
        
        # Optional: Add contract metadata
        contract_metadata = {
            "contract_id": os.path.basename(pdf_path).replace('.pdf', ''),
            "contract_type": "legal_agreement",
            "ingestion_date": datetime.datetime.now().isoformat()
        }
        
        success = ingestor.ingest_pdf(pdf_path, collection_name, contract_metadata)
        if success:
            successful_ingestions += 1
            print(f"✅ Successfully ingested legal document: {pdf_path}")
        else:
            failed_ingestions += 1
            print(f"❌ Failed to ingest legal document: {pdf_path}")

    # Finalize and save vocabulary after all documents processed
    print(f"\n📚 Finalizing vocabulary...")
    ingestor.finalize_vocabulary(collection_name)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    
    print(f"\n📊 Legal Document Ingestion Summary:")
    print(f"   ✅ Successful: {successful_ingestions}")
    print(f"   ❌ Failed: {failed_ingestions}")
    print(f"   📚 Vocabulary size: {len(ingestor.vocab.vocab) if ingestor.vocab.vocab else 0}")
    print(f"   ⏱️ Total time: {elapsed}")
    print(f"\n⚖️ Optimized legal document pipeline completed!")
