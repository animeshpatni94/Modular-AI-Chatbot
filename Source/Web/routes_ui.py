from flask import Blueprint, send_from_directory, jsonify, session

def create_ui_routes(auth):
    ui = Blueprint('ui', __name__)

    @ui.route('/chat-widget')
    @auth.oidc_login_required
    def chat_widget():
        return send_from_directory('Static', 'chat.html')

    @ui.route('/full-ui-screen')
    @auth.oidc_login_required
    def full_ui_screen():
        return send_from_directory('Static', 'full_ui_screen.html')

    @ui.route('/default_message')
    @auth.oidc_login_required
    def default_message():
        full_name = session['user'].get('name', 'User')
        first_name = full_name.split()[0] if full_name else 'User'
        content = f"Hello {first_name}, How can I help?"
        return jsonify({'role': 'ai', 'content': content})

    return ui
