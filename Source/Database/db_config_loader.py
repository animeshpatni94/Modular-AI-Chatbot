from collections import defaultdict
from .db_connection import get_db_connection
from functools import lru_cache

@lru_cache(maxsize=1)
def load_config_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    config_data = defaultdict(dict)
    cursor.execute("""
        SELECT s.name, e.config_key, e.config_value
        FROM ConfigEntries e
        JOIN ConfigSections s ON e.section_id = s.id
    """)
    for section, key, value in cursor.fetchall():
        config_data[section][key] = value
    conn.close()
    return config_data
