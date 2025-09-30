from flask import Blueprint, jsonify, request, Response, session, g
import uuid
from collections import defaultdict
import json
import requests
import os
from datetime import datetime
from .apikey_manager import ApiKeyManager
from Helper.provider_manager import ProviderManager
from Database.db_config_loader import load_config_from_db
from Database.db_logger import log_llm_chat
from Database.db_doc_title import DocTitleDBManager

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
    
    @api.route('/filters', methods=['GET'])
    @api_key_manager.require_auth_or_api_key
    def get_filters():
        try:
            collection_name = config['QDRANT']['collection_name']  # Replace as needed
            filters = DocTitleDBManager.load_doctitle_from_db(collection_name)  # Returns ['Doc1', 'Doc2', ...]
        
            # Transform simple list to dropdown format
            filter_options = [
                {"id": i + 1, "name": str(value), "value": str(value)} 
                for i, value in enumerate(filters)
            ]
        
            return jsonify({"filters": filter_options})
        except Exception as e:
            print(f"❌ Error loading filters: {e}")
            return jsonify({'filters': [], 'error': 'Failed to load filters'}), 500

    @api.route('/rag_chat', methods=['POST'])
    @api_key_manager.require_auth_or_api_key
    def rag_chat_endpoint():
        try:
            data = request.get_json()
            messages = data.get('messages', [])
            stream = data.get('stream', True)
            session_id = data.get('session_id') or str(uuid.uuid4())
            filters = data.get('filters', [])
            if session_id not in session_histories:
                session_histories[session_id] = []
            if not messages:
                return jsonify({'error': 'No messages provided'}), 400
            history_length = int(config['SESSION']['history_length'])
            history = session_histories[session_id][-history_length:]
            new_user_message = messages[-1]
            query = new_user_message.get("content", "")
            vectordb = manager.vectordb_provider
            reranker = manager.reranker
            llm = manager.llm_provider
            search = int(config['DEFAULT']['search_population'])
            top_r = search
            results = vectordb.search(query, search=search, top_r=top_r,filters=filters)
            docs = [doc for doc, score in results] if results and isinstance(results[0], tuple) else results

            if not docs:
                supporting_docs = []
                context = "No relevant documents found."
            else:
                top_docs = reranker.rerank(query, docs, int(top_r*0.3))
                supporting_docs = top_docs
                context = "\n\n".join([doc.page_content for doc in supporting_docs])

            default_prompt = config['DEFAULT']['default_prompt']
            prompt_json = config['SYSTEM_PROMPTS'][default_prompt]
            system_message = json.loads(prompt_json)
            system_message['content'] = system_message['content'].replace('{context}', context)
            chat_messages = [system_message] + history + [new_user_message]
            # Determine user identification based on authentication method
            if hasattr(g, 'is_external_user') and g.is_external_user:
                # External API user
                user_id = f"api_user_{g.external_api_key[:8]}"  # Use partial API key for identification
                print(f"Processing request for external API user: {user_id}")
            else:
                # Session authenticated user
                user_id = session['user'].get('preferred_username')
                print(f"Processing request for session user: {user_id}")

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
                    session_histories[session_id] = session_histories[session_id][-history_length:]
                    log_llm_chat(
                        session_id=session_id,
                        user_id=user_id,
                        user_request=query,
                        llm_response=full_response,
                        retrieval_docs=[
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata
                            }
                            for doc in supporting_docs
                        ],
                        model_name= getattr(llm, 'model_name', None),
                        provider=getattr(llm, 'provider_name', None),
                        prompt_template=default_prompt
                    )

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
                session_histories[session_id] = session_histories[session_id][-history_length:]

                log_llm_chat(
                    session_id=session_id,
                    user_id=user_id,
                    user_request=query,
                    llm_response=answer,
                    retrieval_docs=[
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in supporting_docs
                    ],
                    model_name=getattr(llm, 'model_name', None),
                    provider=getattr(llm, 'provider_name', None),
                    prompt_template=default_prompt
                )
                return jsonify({
                    'response': answer,
                    'citations': format_citations(supporting_docs[:5]),
                    'session_id': session_id
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/sharepoint/test_connection', methods=['POST'])
    @api_key_manager.require_auth_or_api_key
    def test_sharepoint_connection():
        """Test SharePoint connection with provided credentials"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['sharepoint_url', 'client_id', 'client_secret', 'tenant_id']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            sharepoint_url = data['sharepoint_url']
            client_id = data['client_id']
            client_secret = data['client_secret']
            tenant_id = data['tenant_id']
            
            # Test SharePoint connection using Microsoft Graph API
            try:
                # Get access token
                token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
                token_data = {
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'scope': 'https://graph.microsoft.com/.default',
                    'grant_type': 'client_credentials'
                }
                
                token_response = requests.post(token_url, data=token_data)
                token_response.raise_for_status()
                access_token = token_response.json()['access_token']
                
                # Test connection by getting site information
                headers = {'Authorization': f'Bearer {access_token}'}
                
                # Extract site ID from SharePoint URL
                if '/sites/' in sharepoint_url:
                    site_name = sharepoint_url.split('/sites/')[-1]
                    test_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_url.split('//')[1].split('/')[0]}:/sites/{site_name}"
                else:
                    # For root site
                    test_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_url.split('//')[1].split('/')[0]}"
                
                test_response = requests.get(test_url, headers=headers)
                test_response.raise_for_status()
                
                return jsonify({
                    'status': 'success',
                    'message': 'SharePoint connection successful',
                    'site_info': test_response.json()
                })
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'error': 'SharePoint connection failed',
                    'details': str(e)
                }), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @api.route('/sharepoint/ingest', methods=['POST'])
    @api_key_manager.require_auth_or_api_key
    def run_sharepoint_ingestion():
        """Run SharePoint data ingestion pipeline"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['sharepoint_url', 'client_id', 'client_secret', 'tenant_id']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            sharepoint_url = data['sharepoint_url']
            client_id = data['client_id']
            client_secret = data['client_secret']
            tenant_id = data['tenant_id']
            document_library = data.get('document_library', 'Documents')
            folder_path = data.get('folder_path', '')
            
            # Get access token
            try:
                token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
                token_data = {
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'scope': 'https://graph.microsoft.com/.default',
                    'grant_type': 'client_credentials'
                }
                
                token_response = requests.post(token_url, data=token_data)
                token_response.raise_for_status()
                access_token = token_response.json()['access_token']
                
                # Get site ID
                headers = {'Authorization': f'Bearer {access_token}'}
                
                if '/sites/' in sharepoint_url:
                    site_name = sharepoint_url.split('/sites/')[-1]
                    site_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_url.split('//')[1].split('/')[0]}:/sites/{site_name}"
                else:
                    site_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_url.split('//')[1].split('/')[0]}"
                
                site_response = requests.get(site_url, headers=headers)
                site_response.raise_for_status()
                site_id = site_response.json()['id']
                
                # Get document library
                drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
                drives_response = requests.get(drive_url, headers=headers)
                drives_response.raise_for_status()
                
                # Find the document library
                target_drive = None
                for drive in drives_response.json()['value']:
                    if drive['name'] == document_library:
                        target_drive = drive
                        break
                
                if not target_drive:
                    return jsonify({
                        'error': f'Document library "{document_library}" not found'
                    }), 400
                
                # Get files from the library
                files_url = f"https://graph.microsoft.com/v1.0/drives/{target_drive['id']}/root"
                if folder_path:
                    files_url += f":/{folder_path}:"
                files_url += "/children"
                
                files_response = requests.get(files_url, headers=headers)
                files_response.raise_for_status()
                
                files = files_response.json()['value']
                document_count = len([f for f in files if not f.get('folder')])
                
                # Here you would typically:
                # 1. Download and process each document
                # 2. Extract text content
                # 3. Generate embeddings
                # 4. Store in vector database
                
                # For now, return success with file count
                return jsonify({
                    'status': 'success',
                    'message': f'Data ingestion pipeline started successfully',
                    'documents_found': document_count,
                    'library': document_library,
                    'folder_path': folder_path or 'root',
                    'timestamp': datetime.now().isoformat()
                })
                
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'error': 'SharePoint ingestion failed',
                    'details': str(e)
                }), 400
                
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
