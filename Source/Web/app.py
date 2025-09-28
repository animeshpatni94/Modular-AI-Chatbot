from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from .auth_manager import AuthManager
from .routes_ui import create_ui_routes
from .routes_api import create_api_routes
import os

# Initialize Flask with correct static configuration for IIS virtual directory
app_path = os.environ.get('ASPNETCORE_APPL_PATH', '/').lower()
app = Flask(__name__, 
           static_folder='Static', 
           static_url_path=f'{app_path}/Static' if app_path != '/' else '/Static')

# Configure Flask for HTTPS reverse proxy (IIS)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure Flask for IIS virtual directory and HTTPS
app.config.update(
    APPLICATION_ROOT=app_path,
    PREFERRED_URL_SCHEME='https'
)

print(f"Application root set to: {app_path}", flush=True)
print(f"Static URL path set to: {app.static_url_path}", flush=True)
print(f"Preferred URL scheme: {app.config['PREFERRED_URL_SCHEME']}", flush=True)

auth = AuthManager(app)

# Register authentication routes WITH url_prefix for IIS virtual directory
app.add_url_rule(f'{app_path}/login', view_func=auth.login)
app.add_url_rule(f'{app_path}/auth/callback', view_func=auth.auth_callback)
app.add_url_rule(f'{app_path}/logout', view_func=auth.logout)

# Register blueprints WITH url_prefix for IIS virtual directory
app.register_blueprint(create_ui_routes(auth), url_prefix=app_path)
app.register_blueprint(create_api_routes(), url_prefix=app_path)

if __name__ == '__main__':
    import sys
    
    # Get port from AspNetCoreModuleV2
    port = int(os.environ.get('ASPNETCORE_PORT', 5000))
    print(f"SUCCESS: Port from ASPNETCORE_PORT: {port}", flush=True)
    print(f"Final port decision: {port}", flush=True)
    print(f"Application will be accessible at: https://yourserver{app_path}/", flush=True)
    
    # Start Flask
    try:
        print(f"Attempting to bind to 127.0.0.1:{port}", flush=True)
        app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"FATAL ERROR starting Flask: {e}", flush=True)
        raise
