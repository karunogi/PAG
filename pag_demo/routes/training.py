import json
import uuid
from flask import Blueprint, request, jsonify, session, Response
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from ..state import STATE
from ..services.utils import build_auto_onehot_converter, build_schema_from_json
from ..services.models import TrainedPredictor

bp = Blueprint("training", __name__)


@bp.post("/train_start")
def train_start():
    data = request.get_json(force=True)
    model_type = data.get("model_type","logreg")
    schema_text = data.get("schema_json","").strip()
    if not schema_text:
        return jsonify({"ok": False, "error":"schema_json is required."}), 400

    token = uuid.uuid4().hex
    session["train_token"] = token
    session["train_model_type"] = model_type
    session["train_schema_text"] = schema_text
    return jsonify({"ok": True, "token": token})


@bp.get("/train_stream")
def train_stream():
    token = request.args.get("token","")
    if not token or token != session.get("train_token"):
        return "invalid token", 400

    dataset = STATE.get("dataset")
    if dataset is None:
        return "no dataset", 400

    model_type = session.get("train_model_type","logreg")
    schema_text = session.get("train_schema_text","")
    try:
        schema_json = json.loads(schema_text)
    except Exception as e:
        return f"invalid schema_json: {e}", 400

    def event(data): return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        yield event({"type":"log","msg":"splitting dataset..."})
        X = dataset["X"]; y = dataset["y"]
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
        )

        yield event({"type":"log","msg":"building one-hot converter from schema..."})
        converter = build_auto_onehot_converter(schema_json)

        yield event({"type":"log","msg":"vectorizing train samples..."})
        tr_vecs = np.vstack([converter(row.to_dict()) for _,row in Xtr.iterrows()])
        yield event({"type":"log","msg":"vectorizing test samples..."})
        te_vecs = np.vstack([converter(row.to_dict()) for _,row in Xte.iterrows()])

        yield event({"type":"log","msg":f"initializing model: {model_type}"})
        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        else:
            params = LogisticRegression(max_iter=1000).get_params()
            model = LogisticRegression(max_iter=1000, n_jobs=-1) if "n_jobs" in params else LogisticRegression(max_iter=1000)

        yield event({"type":"log","msg":"training..."})
        model.fit(tr_vecs, ytr)

        acc = float(model.score(te_vecs, yte))
        yield event({"type":"log","msg":f"validation accuracy = {acc:.4f}"})

        STATE["trained"] = {"model_name": f"{model_type}_auto", "model_obj": model, "converter": converter}
        yield event({"type":"final","ok": True, "acc": acc, "model_name": STATE['trained']['model_name']})

    return Response(gen(), mimetype="text/event-stream")


@bp.post("/register_trained_model")
def register_trained_model():
    """
    Register the trained model into PAG.
    - If STATE["schema"] is missing, build it from train_schema_text.
    - Align predicted probabilities to the order of schema.labels.
    - After this, ensure_pag_ready() will prefer the trained model.
    """
    trained = STATE.get("trained")
    schema_obj = STATE.get("schema")

    if trained is None:
        return jsonify({"ok": False, "error": "No training result found. Please train first."}), 400

    if schema_obj is None:
        schema_text = session.get("train_schema_text", "")
        if not schema_text:
            return jsonify({"ok": False, "error": "Schema is missing. Save settings or provide the schema_json used for training."}), 400
        try:
            schema_json = json.loads(schema_text)
            schema_obj = build_schema_from_json(schema_json)

            diseases = schema_json.get("labels", []) or []
            symptoms = []
            for f in schema_json.get("features", []):
                if f.get("name") == "symptoms":
                    av = f.get("allowed_values")
                    if isinstance(av, list):
                        symptoms = [str(x) for x in av]
                    break
            STATE["schema"] = schema_obj
            STATE["diseases"] = diseases
            STATE["symptoms"] = symptoms
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to parse schema: {e}"}), 400

    labels = STATE["schema"].labels or []

    def schema_converter(structured_input: dict):
        return trained["converter"](structured_input)

    STATE["trained"]["predictor_cls"] = TrainedPredictor
    STATE["trained"]["schema_converter"] = schema_converter
    STATE["use_trained"] = True

    return jsonify({"ok": True, "model_name": trained["model_name"]})
