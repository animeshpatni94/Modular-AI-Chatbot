import os
from typing import List, Dict
from dataclasses import dataclass
import pdfplumber
import nltk
from nltk.corpus import stopwords
from Helper.provider_manager import ProviderManager
from Database.db_config_loader import load_config_from_db
from Helper.keyword_vocab import KeywordVocabulary
from Database.db_vocab import VocabDBManager
import datetime
import fitz
from PIL import Image
import pytesseract
import time
from typing import List

@dataclass
class TextChunk:
    id: str
    page_content: str
    page_number: int
    metadata: Dict[str, str]

class PDFTextIngestor:
    def __init__(self, embedding_provider, vectordb_provider, kw_model, vocabulary: KeywordVocabulary):
        self.embedding_provider = embedding_provider
        self.vectordb_provider = vectordb_provider
        self.vocab = vocabulary
        self.kw_model = kw_model

        # 🌐 Ensure NLTK stopwords are available
        try:
            self.stop_words = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = stopwords.words("english")

    @staticmethod
    def ocr_pdf_page(pdf_path: str, page_number: int) -> str:
        """
        OCR a specific page of a PDF using PyMuPDF for rendering, returns extracted text.
        page_number is 1-based.
        """
        try:
            pdf = fitz.open(pdf_path)
            page = pdf.load_page(page_number - 1)  # fitz: 0-based
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang='eng')
            return text
        except Exception as e:
            print(f"❌ OCR failed for {pdf_path}, page {page_number}: {e}")
            return ""

    def ingest_pdf(self, pdf_path: str, collection_name: str) -> bool:
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            doc_title = self._extract_document_title(pdf_path)
            chunks = self._extract_text_chunks(pdf_path, doc_title)
            if not chunks:
                print(f"⚠️ No text chunks extracted from {pdf_path}")
                return False

            dense_embeddings = self._generate_embeddings([chunk.page_content for chunk in chunks])

            vectors = []
            for i, chunk in enumerate(chunks):
                keywords = self.kw_model.extract_keywords(
                    chunk.page_content,
                    keyphrase_ngram_range=(1, 3),
                    stop_words=self.stop_words,
                    top_n=10
                )
                keyword_terms = [kw[0] for kw in keywords]
                sparse_text = " ".join(keyword_terms)

                sparse_indices = [self.vocab.get_index(term) for term in keyword_terms]
                sparse_values = [1.0] * len(sparse_indices)

                vectors.append({
                    "id": chunk.id,
                    "values": dense_embeddings[i],
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                    "payload": {
                        "page_content": chunk.page_content,
                        "sparse_text": sparse_text,
                        "keywords": keyword_terms,
                        "metadata": {
                            "page": chunk.page_number,
                            **chunk.metadata
                        }
                    }
                })

            success = self.vectordb_provider.upsert_vectors(collection_name, vectors)
            if success:
                print(f"✅ Ingested {len(chunks)} chunks from: {pdf_path}")
            return success

        except Exception as e:
            print(f"❌ PDF ingestion failed for {pdf_path}: {str(e)}")
            return False

    def _extract_document_title(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata
                if metadata and metadata.get('Title'):
                    return metadata['Title'].strip()
        except Exception:
            pass
        return os.path.splitext(os.path.basename(pdf_path))[0]

    def _extract_text_chunks(self, pdf_path: str, doc_title: str) -> List[TextChunk]:
        chunks = []
        chunk_counter = 0
        config = load_config_from_db()
        chunk_size = int(config['TEXT_SPLITTING']['chunk_size'])
        overlap = int(config['TEXT_SPLITTING']['overlap'])

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text_blocks = []
                native_text = page.extract_text() or ""
                if native_text.strip():
                    text_blocks.append(native_text)

                # 2. OCR all images ALWAYS—important for figures/stamps even if native text exists
                for img in (page.images or []):
                    try:
                        x0 = max(0, img["x0"])
                        y0 = max(0, img["top"])
                        x1 = min(page.width, img["x1"])
                        y1 = min(page.height, img["bottom"])
                        if x1 > x0 and y1 > y0:
                            cropped = page.crop((x0, y0, x1, y1))
                            page_img = cropped.to_image(resolution=300)
                            pil_img = page_img.original
                            ocr_img_text = pytesseract.image_to_string(pil_img, lang='eng')
                            if ocr_img_text.strip():
                                text_blocks.append(ocr_img_text)
                    except Exception as err:
                        print(f"⚠️ Failed to OCR image on page {page_num}: {err}")

                # 3. If NO native text, fallback to OCR the whole page image (scanned PDF)
                if not native_text.strip():
                    try:
                        ocr_text = PDFTextIngestor.ocr_pdf_page(pdf_path, page_num)
                        if ocr_text.strip():
                            text_blocks.append(ocr_text)
                    except Exception as err:
                        print(f"⚠️ OCR error on page {page_num}: {err}")


                full_text = "\n".join([t for t in text_blocks if t.strip()])
                if not full_text.strip():
                    continue  # skip empty pages

                page_chunks = self._split_text(full_text, chunk_size, overlap)
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
                        }
                    ))
                    chunk_counter += 1
        return chunks

    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        prev_start = -1
        while start < len(words):
            if start == prev_start:
                break
            prev_start = start
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        max_retries = 5
        retry_delay = 60  # Start with 60 second delay
        for attempt in range(max_retries):
            try:
                return self.embedding_provider.embed_documents(texts)
            except Exception as e:
                # Check for rate limit errors using message or exception type
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"❌ Rate limit encountered, retrying {attempt + 1}/{max_retries} after {retry_delay}s")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Embedding generation failed: {str(e)}")
                    raise
        raise Exception("Max retries exceeded for embedding generation due to rate limit")

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    config = load_config_from_db()
    folder_path = config['DEFAULT']['folder_path']
    collection_name = config['DEFAULT']['ingestion_collection_name']

    manager = ProviderManager()
    embedding_provider = manager.embedding_provider.embedding_model
    vectordb_provider = manager.vectordb_provider
    kw_model = manager.kw_model

    from tqdm import tqdm
    vocab = KeywordVocabulary()
    print("🔍 Building global keyword vocabulary...")

    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    vocab_dict = VocabDBManager.load_vocab_from_db(collection_name)
    if vocab_dict is not None:
        vocab.load_from_dict(vocab_dict)

    # --- VOCAB BUILDER: Extract native text, fallback OCR, and OCR on all images --- #
    if vocab.vocab is None or len(vocab.vocab) == 0:
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text_blocks = []
                    # Extract native text, fallback to OCR if none
                    native_text = page.extract_text() or ""
                    if native_text.strip():
                        text_blocks.append(native_text)

                    # 2. OCR all images ALWAYS—important for figures/stamps even if native text exists
                    for img in (page.images or []):
                        try:
                            x0 = max(0, img["x0"])
                            y0 = max(0, img["top"])
                            x1 = min(page.width, img["x1"])
                            y1 = min(page.height, img["bottom"])
                            if x1 > x0 and y1 > y0:
                                cropped = page.crop((x0, y0, x1, y1))
                                page_img = cropped.to_image(resolution=300)
                                pil_img = page_img.original
                                ocr_img_text = pytesseract.image_to_string(pil_img, lang='eng')
                                if ocr_img_text.strip():
                                    text_blocks.append(ocr_img_text)
                        except Exception as err:
                            print(f"⚠️ Failed to OCR image on page {page_num}: {err}")

                        # 3. If NO native text, fallback to OCR the whole page image (scanned PDF)
                    if not native_text.strip():
                        try:
                            ocr_text = PDFTextIngestor.ocr_pdf_page(pdf_path, page_num)
                            if ocr_text.strip():
                                text_blocks.append(ocr_text)
                        except Exception as err:
                            print(f"⚠️ OCR error on page {page_num}: {err}")

                    full_text = "\n".join([t for t in text_blocks if t.strip()])
                    if not full_text.strip():
                        continue
                    try:
                        keywords = kw_model.extract_keywords(
                            full_text,
                            keyphrase_ngram_range=(1, 3),
                            stop_words=stopwords.words("english"),
                            top_n=10
                        )
                        terms = [kw[0] for kw in keywords]
                        vocab.build_from_keywords([terms])
                    except Exception as err:
                        print(f"⚠️ Skipping keywords for page {page_num}: {err}")

        VocabDBManager.save_vocab_to_db(collection_name, vocab.vocab)

    print(f"📚 Vocabulary size: {len(vocab.vocab)} terms")

    ingestor = PDFTextIngestor(embedding_provider, vectordb_provider, kw_model, vocabulary=vocab)

    for pdf_path in pdf_files:
        print(f"\n📄 Ingesting: {pdf_path}")
        success = ingestor.ingest_pdf(pdf_path, collection_name)
        if success:
            print(f"✅ Successfully ingested: {pdf_path}")
        else:
            print(f"❌ Failed to ingest: {pdf_path}")

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print(f"\n⏱️ Pipeline completed in {elapsed} (hh:mm:ss.microseconds)")
