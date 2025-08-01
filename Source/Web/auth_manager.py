
import secrets
import string
from authlib.integrations.flask_client import OAuth
from flask import session, redirect, request, url_for, abort
from functools import wraps
import requests
from Database.db_config_loader import load_config_from_db
import time

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour

class AuthManager:
    def __init__(self, app):
        config = load_config_from_db()
        oidc = config['OIDC']
        self.client_id = oidc.get('client_id')
        self.client_secret = oidc.get('client_secret')
        self.discovery_url = oidc.get('discovery_url')
        self.scope = oidc.get('scope', 'openid email profile')
        app.secret_key = oidc.get('valid_api_keys')
        self.oauth = OAuth(app)
        self.oauth.register(
            name='oidc',
            client_id=self.client_id,
            client_secret=self.client_secret,
            server_metadata_url=self.discovery_url,
            client_kwargs={
                'scope': self.scope,
                'code_challenge_method': 'S256'
            }
        )
        self.clear_session = oidc.get('clear_session', 'false').strip().lower() in ['true']
        client = self.oauth.create_client('oidc')
        metadata = client.load_server_metadata()
        self.introspection_endpoint = metadata.get('introspection_endpoint')

    def _generate_nonce(self, length=32):
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

    def login(self):
        next_url = request.args.get('next') or request.referrer or url_for('ui.chat_widget')
        session['next'] = next_url
        nonce = self._generate_nonce()
        session['oidc_nonce'] = nonce
        redirect_uri = url_for('auth_callback', _external=True)
        return self.oauth.oidc.authorize_redirect(
            redirect_uri=redirect_uri,
            nonce=nonce 
        )

    def auth_callback(self):
        token = self.oauth.oidc.authorize_access_token()
        nonce = session.pop('oidc_nonce', None)
        if not nonce:
            abort(400, 'Missing nonce in session.')
        id_token = self.oauth.oidc.parse_id_token(token, nonce=nonce)
        session['user'] = id_token
        session['access_token'] = token.get('access_token')
        session['login_time'] = time.time()
        next_url = session.pop('next', url_for('ui.chat_widget'))
        return redirect(next_url)

    def logout(self):
        session.pop('user', None)
        session.pop('access_token', None)
        session.pop('oidc_nonce', None)
        return redirect('/')

    def oidc_login_required(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for session timeout
            login_time = session.get('login_time')
            if not login_time or (time.time() - login_time) > SESSION_TIMEOUT_SECONDS:
                # Clear session and force re-login
                session.clear()
                return redirect(url_for('login', next=request.url))
            
            # Check if user session exists
            if 'user' not in session:
                return redirect(url_for('login', next=request.url))

            # (Optional) Token introspection to ensure access token validity
            if self.introspection_endpoint and 'access_token' in session:
                access_token = session.get('access_token')
                if not self._introspect_token(access_token):
                    session.clear()
                    return abort(401, 'Access token is invalid or expired.')

            return func(*args, **kwargs)
        return wrapper

    def _introspect_token(self, access_token):
        try:
            response = requests.post(
                self.introspection_endpoint,
                auth=(self.client_id, self.client_secret),
                data={'token': access_token}
            )
            data = response.json()
            return data.get('active', False)
        except Exception as e:
            print(f"Token introspection failed: {e}")
            return False
