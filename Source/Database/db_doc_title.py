from .db_connection import get_db_connection
from functools import lru_cache 
import json

class DocTitleDBManager:
    @staticmethod
    def save_doctitle_to_db(collection_name: str, doc_title: dict):
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
                INSERT INTO DocTitleCollection (CollectionName, DocumentTitle)
                VALUES (?, ?)
                """, (collection_name, doc_title))

        conn.commit()
        conn.close()

        print(f"✅ Document Title '{doc_title}' saved to DB for collection '{collection_name}'")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_doctitle_from_db(collection_name: str) -> list:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DocumentTitle FROM DocTitleCollection
            WHERE CollectionName = ?
        """, (collection_name,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []  # Return empty list if no data found
    
        # Extract the DocumentTitle string from each tuple
        return [row[0] for row in rows]
