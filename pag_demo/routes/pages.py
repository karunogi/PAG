# pag_demo/routes/pages.py
from flask import Blueprint, render_template, request, redirect, url_for
import json

from ..services.utils import default_demo_values, build_schema_from_json
from ..state import STATE

bp = Blueprint("pages", __name__)

def _to_bool(v):
    return str(v).lower() in ("1", "true", "on", "yes", "y")

@bp.route("/", methods=["GET"])
def index():
    # Prefill schema box
    diseases, symptoms, schema_json, _ = default_demo_values()
    if STATE["schema"] is None:
        schema_text = json.dumps(schema_json, ensure_ascii=False, indent=2)
    else:
        feats = []
        for f in STATE["schema"].fields.values():
            feats.append({
                "name": f.name,
                "dtype": f.dtype.__name__ if hasattr(f.dtype, "__name__") else str(f.dtype),
                "allowed_values": f.allowed_values,
                "needs_alignment": getattr(f, "needs_alignment", True),
                "description": f.description
            })
        schema_text = json.dumps({"features": feats, "labels": STATE["schema"].labels}, ensure_ascii=False, indent=2)

    return render_template(
        "index.html",
        api_key=STATE["api_key"],
        llm_model=STATE["llm_model"],
        emb_model=STATE["emb_model"],
        schema_json=schema_text,
        pag_cfg=STATE["pag_cfg"],
        align_cfg=STATE["align_cfg"],
    )

@bp.route("/setup", methods=["POST"])
def setup():
    api_key  = request.form.get("api_key", "").strip()
    llm_model = request.form.get("llm_model", "gpt-4o-mini").strip()
    emb_model = request.form.get("emb_model", "text-embedding-3-small").strip()
    schema_text = request.form.get("schema_json","").strip()
    if not schema_text:
        return "schema_json is required.", 400

    # Parse schema JSON
    try:
        schema_json = json.loads(schema_text)
        schema = build_schema_from_json(schema_json)
        diseases = schema_json.get("labels", []) or []
        symptoms = []
        for f in schema_json.get("features", []):
            if f.get("name") == "symptoms":
                av = f.get("allowed_values")
                if isinstance(av, list):
                    symptoms = [str(x) for x in av]
                break
    except Exception as e:
        return f"Failed to parse schema: {e}", 400

    # Persist basics
    STATE["api_key"] = api_key
    STATE["llm_model"] = llm_model
    STATE["emb_model"] = emb_model
    STATE["schema"] = schema
    STATE["diseases"] = diseases
    STATE["symptoms"] = symptoms
    STATE["pag"] = None  # rebuild later

    # Read PAGConfig
    try:
        STATE["pag_cfg"] = {
            "top_k_predictions": int(request.form.get("pag_top_k_predictions", 5)),
            "num_schema_categorical": int(request.form.get("pag_num_schema_categorical", 10)),
            "short_knowledge": _to_bool(request.form.get("pag_short_knowledge")),
            "knowledge_per_label": _to_bool(request.form.get("pag_knowledge_per_label")),
            "aggregate_with_pm": _to_bool(request.form.get("pag_aggregate_with_pm")),
            "user_input_with_kg": _to_bool(request.form.get("pag_user_input_with_kg")),
            "embedding_cache_path": request.form.get("pag_embedding_cache_path", "./embedding_cache.pickle").strip() or "./embedding_cache.pickle",
            "match_cache_path": request.form.get("pag_match_cache_path", "./match_cache.pickle").strip() or "./match_cache.pickle",
        }
    except Exception as e:
        return f"Invalid PAG config: {e}", 400

    # Read AlignmentConfig
    try:
        STATE["align_cfg"] = {
            "top_k_candidates": int(request.form.get("align_top_k_candidates", 30)),
            "use_semantic": _to_bool(request.form.get("align_use_semantic")),
        }
    except Exception as e:
        return f"Invalid Alignment config: {e}", 400

    return redirect(url_for("pages.run_page"))

@bp.route("/run", methods=["GET"])
def run_page():
    _, _, _, demo_user_input = default_demo_values()
    return render_template("run.html",
                           llm_model=STATE["llm_model"],
                           emb_model=STATE["emb_model"],
                           diseases_count=len(STATE["diseases"]),
                           demo_user_input=demo_user_input)
