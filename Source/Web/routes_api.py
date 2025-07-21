from flask import Blueprint, jsonify, request, Response
import uuid
from collections import defaultdict
import json
from .apikey_manager import ApiKeyManager
from Helper.provider_manager import ProviderManager
from Database.db_config_loader import load_config_from_db

def format_citations(docs):
    return [
        {
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("document_title")
        }
        for doc in docs
    ]

def create_api_routes():
    api = Blueprint('api', __name__)
    manager = ProviderManager()
    session_histories = defaultdict(list)
    config = load_config_from_db()
    api_key_manager = ApiKeyManager()

    @api.route('/chat', methods=['POST'])
    @api_key_manager.require_api_key
    def chat_endpoint():
        try:
            data = request.get_json()
            messages = data.get('messages', [])
            stream = data.get('stream', True)
            session_id = data.get('session_id') or str(uuid.uuid4())
            if not messages:
                return jsonify({'error': 'No messages provided'}), 400
            history_length = int(config['SESSION']['history_length'])
            history = session_histories[session_id][history_length:]
            new_user_message = messages[-1]
            messages_for_llm = history + [new_user_message]

            if stream:
                def generate():
                    full_response = ""
                    for token in manager.llm_provider.stream_chat_response(messages_for_llm):
                        full_response += token
                        yield token
                    session_histories[session_id].extend([
                        new_user_message,
                        {"role": "assistant", "content": full_response}
                    ])
                    history_length = int(config['SESSION']['history_length'])
                    session_histories[session_id] = session_histories[session_id][history_length:]
                return Response(
                    generate(),
                    mimetype='text/plain',
                    headers={'X-Session-Id': session_id}
                )
            else:
                response = manager.llm_provider.get_complete_response(messages_for_llm)
                session_histories[session_id].extend([
                    new_user_message,
                    {"role": "assistant", "content": response}
                ])
                history_length = int(config['SESSION']['history_length'])
                session_histories[session_id] = session_histories[session_id][history_length:]
                return jsonify({
                    'response': response,
                    'session_id': session_id
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/embed', methods=['POST'])
    @api_key_manager.require_api_key
    def embed_endpoint():
        try:
            data = request.get_json()
            text = data.get('text')
            texts = data.get('texts')
            if not text and not texts:
                return jsonify({'error': 'No text provided'}), 400
            if text:
                embedding = manager.embedding_provider.generate_embedding(text)
                return jsonify({'embedding': embedding})
            elif texts:
                embeddings = manager.embedding_provider.generate_embeddings(texts)
                return jsonify({'embeddings': embeddings})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/switch', methods=['POST'])
    @api_key_manager.require_api_key
    def switch_provider():
        try:
            data = request.get_json()
            new_provider = data.get('provider', '').lower()
            provider_type = data.get('provider_type', 'llm')
            if not new_provider:
                return jsonify({'error': 'No provider specified'}), 400
            manager.switch_provider(new_provider, provider_type)
            return jsonify({
                'status': 'success',
                'message': f'Switched to {new_provider} ({provider_type})'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/vectordb/store', methods=['POST'])
    @api_key_manager.require_api_key
    def vectordb_store():
        try:
            data = request.get_json()
            texts = data.get('texts')
            metadatas = data.get('metadatas', None)
            if not texts:
                return jsonify({'error': 'No texts provided'}), 400
            result = manager.vectordb_provider.store_embeddings(texts, metadatas)
            return jsonify({'status': 'success', 'result': str(result)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/vectordb/search', methods=['POST'])
    @api_key_manager.require_api_key
    def vectordb_search():
        try:
            data = request.get_json()
            query = data.get('query')
            k = data.get('k', 5)
            if not query:
                return jsonify({'error': 'No query provided'}), 400
            results = manager.vectordb_provider.search(query, k)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
            return jsonify({'results': formatted_results})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/rerank', methods=['POST'])
    @api_key_manager.require_api_key
    def rerank():
        try:
            data = request.get_json()
            query = data.get('query')
            documents = data.get('documents')
            top_n = data.get('top_n', 3)
            if not query or not documents:
                return jsonify({'error': 'Query and documents are required'}), 400
            manager.reranker.top_n = top_n
            reranked = manager.reranker.rerank(query, documents)
            return jsonify({'reranked_documents': reranked})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/rag_chat', methods=['POST'])
    @api_key_manager.require_auth_or_api_key
    def rag_chat_endpoint():
        try:
            data = request.get_json()
            messages = data.get('messages', [])
            stream = data.get('stream', True)
            session_id = data.get('session_id') or str(uuid.uuid4())
            if session_id not in session_histories:
                session_histories[session_id] = []
            if not messages:
                return jsonify({'error': 'No messages provided'}), 400
            history_length = int(config['SESSION']['history_length'])
            history = session_histories[session_id][history_length:]
            new_user_message = messages[-1]
            query = new_user_message.get("content", "")
            vectordb = manager.vectordb_provider
            reranker = manager.reranker
            llm = manager.llm_provider
            retrieval_k = 50

            results = vectordb.search(query, k=retrieval_k)
            docs = [doc for doc, score in results] if results and isinstance(results[0], tuple) else results

            if not docs:
                supporting_docs = []
                context = "No relevant documents found."
            else:
                top_docs = reranker.rerank(query, docs, 20)
                supporting_docs = top_docs
                context = "\n\n".join([doc.page_content for doc in supporting_docs])

            default_prompt = config['DEFAULT']['default_prompt']
            prompt_json = config['SYSTEM_PROMPTS'][default_prompt]
            system_message = json.loads(prompt_json)
            system_message['content'] = system_message['content'].replace('{context}', context)
            chat_messages = [system_message] + history + [new_user_message]

            if stream:
                def generate():
                    full_response = ""
                    for token in llm.stream_chat_response(chat_messages):
                        full_response += token
                        yield token
                    session_histories[session_id].extend([
                        new_user_message,
                        {"role": "assistant", "content": full_response}
                    ])
                    history_length = int(config['SESSION']['history_length'])
                    session_histories[session_id] = session_histories[session_id][history_length:]
                    yield "\n---CITATIONS---\n"
                    yield json.dumps({
                        "citations": format_citations(supporting_docs)
                    })
                return Response(
                    generate(),
                    mimetype='text/plain',
                    headers={'X-Session-Id': session_id}
                )
            else:
                answer = "".join(llm.get_complete_response(chat_messages))
                session_histories[session_id].extend([
                    new_user_message,
                    {"role": "assistant", "content": answer}
                ])
                history_length = int(config['SESSION']['history_length'])
                session_histories[session_id] = session_histories[session_id][history_length:]
                return jsonify({
                    'response': answer,
                    'citations': format_citations(supporting_docs[:5]),
                    'session_id': session_id
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'llm_provider': manager.llm_provider_name,
            'embedding_provider': manager.embedding_provider_name,
            'vectordb_provider': manager.vectordb_provider_name
        })

    return api
