from flask import Blueprint, send_from_directory

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

    return ui
