from flask import Flask
from .auth_manager import AuthManager
from .routes_ui import create_ui_routes
from .routes_api import create_api_routes

app = Flask(__name__)
auth = AuthManager(app)
app.add_url_rule('/login', view_func=auth.login)
app.add_url_rule('/auth/callback', view_func=auth.auth_callback)
app.add_url_rule('/logout', view_func=auth.logout)
app.register_blueprint(create_ui_routes(auth))
app.register_blueprint(create_api_routes())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
