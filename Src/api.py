from flask import Flask, request, Response, jsonify, send_from_directory
import uuid
from collections import defaultdict
from provider_manager import ProviderManager

app = Flask(__name__)
session_histories = defaultdict(list)
manager = ProviderManager()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        stream = data.get('stream', True)
        session_id = data.get('session_id') or str(uuid.uuid4())
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        history = session_histories[session_id][-6:]
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
                session_histories[session_id] = session_histories[session_id][-6:]
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
            session_histories[session_id] = session_histories[session_id][-6:]
            return jsonify({
                'response': response,
                'session_id': session_id
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch', methods=['POST'])
def switch_provider():
    try:
        data = request.get_json()
        new_provider = data.get('provider', '').lower()
        provider_type = data.get('provider_type', 'llm')  # 'llm', 'embedding', or 'vectordb'
        if not new_provider:
            return jsonify({'error': 'No provider specified'}), 400
        manager.switch_provider(new_provider, provider_type)
        return jsonify({
            'status': 'success',
            'message': f'Switched to {new_provider} provider ({provider_type})'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/embed', methods=['POST'])
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

@app.route('/vectordb/store', methods=['POST'])
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

@app.route('/vectordb/search', methods=['POST'])
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'llm_provider': manager.llm_provider_name,
        'embedding_provider': manager.embedding_provider_name,
        'vectordb_provider': manager.vectordb_provider_name 
    })

@app.route('/chat-widget')
def chat_widget():
    return send_from_directory('static', 'chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
