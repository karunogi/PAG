import os
import uuid
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from .routes.pages import bp as pages_bp
from .routes.pipeline import bp as pipeline_bp
from .routes.data import bp as data_bp
from .routes.training import bp as training_bp


def create_app():
    app = Flask(__name__, template_folder="templates")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_" + uuid.uuid4().hex)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # Blueprints
    app.register_blueprint(pages_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(training_bp)

    # Misc
    @app.get("/healthz")
    def healthz():
        return "ok"

    return app
