from pag_demo import create_app

if __name__ == "__main__":
    # FLASK_RUN_PORT=5000 python run_app.py
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
