from flask import Flask, request, Response, jsonify, send_from_directory
import configparser
import importlib
import uuid
from collections import defaultdict
from llm_factory import LLMFactory

app = Flask(__name__)

session_histories = defaultdict(list)
factory = LLMFactory()
factory.register_provider('ollama', 'ollama_layer')
factory.register_provider('azureai', 'azureai_layer')
config = configparser.ConfigParser()
config.read('config.ini')
provider_name = config.get('DEFAULT', 'provider', fallback='ollama').lower()
provider_module_name = factory.get_provider(provider_name)
provider_module = importlib.import_module(provider_module_name)
print(f"Using provider: {provider_name} ({provider_module_name})")

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
                for token in provider_module.stream_chat_response(messages_for_llm):
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
            response = provider_module.get_complete_response(messages_for_llm)
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
    global provider_name, provider_module, provider_module_name
    try:
        data = request.get_json()
        new_provider = data.get('provider', '').lower()
        if not new_provider:
            return jsonify({'error': 'No provider specified'}), 400
        new_module_name = factory.get_provider(new_provider)
        new_module = importlib.import_module(new_module_name)
        config.set('DEFAULT', 'provider', new_provider)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        provider_name = new_provider
        provider_module_name = new_module_name
        provider_module = new_module

        return jsonify({
            'status': 'success',
            'message': f'Switched to {provider_name} provider'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'provider': provider_name,
        'module': provider_module_name
    })

@app.route('/chat-widget')
def chat_widget():
    return send_from_directory('static', 'chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
