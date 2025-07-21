import json
from .db_connection import get_db_connection

def log_llm_chat(
    session_id,
    user_id,
    user_request,
    llm_response,
    retrieval_docs,
    model_name=None,
    provider=None,
    prompt_template=None
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO LLMChatLogs (
            session_id, user_id, request, response, model_name, provider, prompt_template, retrieval_docs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        user_id,
        user_request,
        llm_response,
        model_name,
        provider,
        prompt_template,
        json.dumps(retrieval_docs) if retrieval_docs is not None else None
    ))
    conn.commit()
    conn.close()
