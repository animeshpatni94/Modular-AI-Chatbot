import json
from datetime import datetime
from .db_connection import get_db_connection
from functools import lru_cache 

class VocabDBManager:
    @staticmethod
    def save_vocab_to_db(collection_name: str, vocab_dict: dict):
        conn = get_db_connection()
        cursor = conn.cursor()

        vocab_json = json.dumps(vocab_dict, ensure_ascii=False)
        now = datetime.now()

        # Check if collection already exists
        cursor.execute("""
            SELECT COUNT(*) FROM VectorVocabularies
            WHERE collection_name = ?
        """, (collection_name,))

        exists = cursor.fetchone()[0] > 0

        if exists:
            # UPDATE path (safe for large NVARCHARs)
            cursor.execute("""
                UPDATE VectorVocabularies
                SET vocab_json = ?, updated_at = ?
                WHERE collection_name = ?
                """, (vocab_json, now, collection_name))
        else:
            # INSERT path (safe for large NVARCHARs)
            cursor.execute("""
                INSERT INTO VectorVocabularies (collection_name, vocab_json, updated_at)
                VALUES (?, ?, ?)
                """, (collection_name, vocab_json, now))

        conn.commit()
        conn.close()

        print(f"✅ Vocabulary saved to DB for collection '{collection_name}' (Length: {len(vocab_json)} characters)")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_vocab_from_db(collection_name: str) -> dict:
        """
        Load the vocabulary dict stored for the given collection_name.
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT vocab_json FROM VectorVocabularies
            WHERE collection_name = ?
        """, (collection_name,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        vocab_dict = json.loads(row[0])
        print(f"📦 Loaded vocabulary for '{collection_name}' with {len(vocab_dict)} entries")
        return vocab_dict
