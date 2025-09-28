from flask import request, jsonify, session
from functools import wraps
from Database.db_config_loader import load_config_from_db

class ApiKeyManager:
    def __init__(self):
        config = load_config_from_db()
        keys = config['OIDC'].get('valid_api_keys', '')
        self.valid_keys = [k.strip() for k in keys.split(',') if k.strip()]

    def require_api_key(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_key = request.headers.get('x-api-key')
            if not api_key or api_key not in self.valid_keys:
                return jsonify({'error': 'Unauthorized'}), 401
            return func(*args, **kwargs)
        return wrapper

    def require_auth_or_api_key(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check session authentication first
            user_authenticated = 'user' in session
        
            if user_authenticated:
                # User is authenticated via session - regular user
                return func(*args, **kwargs)
        
            # Check for API key authentication
            api_key = request.headers.get('x-api-key')
        
            if api_key and api_key in self.valid_keys:
                # External user authenticated via API key
                # Set a flag to indicate this is an external API user
                from flask import g
                g.is_external_user = True
                g.external_api_key = api_key
                print("External user logged in via API key")  # Optional logging
                return func(*args, **kwargs)
        
            # Neither session nor valid API key found
            return jsonify({'error': 'Unauthorized: Valid session or API key required'}), 401
    
        return wrapper

