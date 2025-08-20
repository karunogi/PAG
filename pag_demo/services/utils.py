import io
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.PAG import PAG, FeatureSchema  # your existing package
from .models import DummyPredictor, FeatEncoder
from ..state import STATE

_DTYPE_MAP = {"int": int, "float": float, "bool": bool, "list": list, "str": str}


def default_demo_values() -> Tuple[List[str], List[str], Dict[str, Any], str]:
    diseases = [
        "Common cold", "Pneumonia", "Diabetes mellitus", "Hypertension", "Asthma"
    ]
    symptoms = [
        "Runny nose","Sneezing","Sore throat","Cough","Mild headache","Fatigue",
        "Cough with phlegm","Fever and chills","Shortness of breath",
        "Chest pain when breathing or coughing","Nausea or vomiting",
        "Frequent urination","Excessive thirst","Increased hunger","Blurred vision",
        "Slow-healing wounds","Numbness or tingling in hands/feet","Headache","Dizziness",
        "Nosebleeds","Wheezing","Chest tightness"
    ]
    schema_json = {
        "features": [
            {"name": "age", "dtype": "int", "needs_alignment": False, "description": "Age in years, int type"},
        {"name": "sex", "dtype": "str", "allowed_values": ["male","female"], "needs_alignment": True},
            {"name": "symptoms", "dtype": "list", "allowed_values": symptoms, "needs_alignment": True,
             "description": "symptoms of the patient, valid json list[str]"}
        ],
        "labels": diseases
    }
    demo_user_input = (
        "Doctor, I’ve been feeling really unwell for the past week. I have a persistent cough "
        "that brings up thick phlegm, and I often get chills with a high fever. Breathing has "
        "become harder, especially when I try to walk even a short distance, and I feel sharp "
        "pain in my chest whenever I take a deep breath or cough. I feel extremely tired all the "
        "time, and yesterday I even felt nauseous and vomited. I’m really worried because it seems "
        "to be getting worse."
    )
    return diseases, symptoms, schema_json, demo_user_input


def _smart_split_text_list(text: str) -> List[str]:
    """
    Split text like:
      - "a\nb\nc"
      - "a, b, c"
      - "a;b|c"
      - "a、b，c"
    into ["a","b","c"], trimming empties.
    """
    parts = re.split(r"[\r\n,;|、，]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_allowed_values(av) -> List[str] | None:
    """
    Normalize allowed_values when user pasted a single multiline string.
    Accepts list[str] or str; returns list[str] or None.
    """
    if av is None:
        return None
    if isinstance(av, str):
        return _smart_split_text_list(av) or [av]
    if isinstance(av, list):
        # If it's a single big string with \n, split it.
        if len(av) == 1 and isinstance(av[0], str):
            split = _smart_split_text_list(av[0])
            return split or av
        # Already a list: ensure all stringified & trimmed
        return [str(x).strip() for x in av if str(x).strip()]
    # Fallback
    return [str(av).strip()]


def build_schema_from_json(schema_json: Dict[str, Any]) -> FeatureSchema:
    fs = FeatureSchema()
    for f in schema_json.get("features", []):
        dtype_str = f.get("dtype", "str")
        dtype = _DTYPE_MAP.get(dtype_str, str)

        allowed_raw = f.get("allowed_values")
        allowed_values = _normalize_allowed_values(allowed_raw)

        fs.register_feature(
            name=f["name"],
            dtype=dtype,
            allowed_values=allowed_values,
            needs_alignment=f.get("needs_alignment", True),
            description=f.get("description","")
        )
    fs.register_labels(schema_json.get("labels", []))
    return fs


def build_auto_onehot_converter(schema_json: Dict[str, Any]):
    feats = schema_json.get("features", [])
    layout = []
    total_dim = 0
    for f in feats:
        name = f["name"]
        dtype = _DTYPE_MAP.get(f.get("dtype","str"), str)
        allowed = _normalize_allowed_values(f.get("allowed_values"))
        if dtype in (int, float, bool):
            size = 1
            layout.append({"name":name, "dtype":dtype, "start":total_dim, "size":size})
            total_dim += size
        elif dtype is str:
            size = 1
            indexer = {val:i for i,val in enumerate(allowed or [])}
            layout.append({"name":name, "dtype":dtype, "indexer":indexer, "start":total_dim, "size":size})
            total_dim += size
        elif dtype is list:
            allowed = allowed or []
            size = len(allowed)
            indexer = {val:i for i,val in enumerate(allowed)}
            layout.append({"name":name, "dtype":dtype, "indexer":indexer, "start":total_dim, "size":size})
            total_dim += size
        else:
            size = 1
            layout.append({"name":name, "dtype":str, "start":total_dim, "size":size})
            total_dim += size

    def _coerce_list(v):
        # Already list
        if isinstance(v, list):
            return v
        # JSON-like list/tuple in string
        if isinstance(v, str):
            s = v.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    parsed = json.loads(s.replace("(", "[").replace(")", "]"))
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            # Fallback: smart split on newline/commas/semicolons/pipe etc.
            return _smart_split_text_list(s)
        # None/NaN → empty
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        # Anything else → single-item list
        return [v]

    def converter(feature_dict: dict):
        vec = np.zeros(total_dim, dtype=float)
        for item in layout:
            name = item["name"]; start=item["start"]; dtype=item["dtype"]
            val = feature_dict.get(name, None)

            if dtype in (int, float, bool):
                try:
                    x = 0 if val is None else dtype(val)
                except Exception:
                    x = 0
                vec[start] = float(x)

            elif dtype is str:
                idx = -1
                if val is not None and item.get("indexer"):
                    s = str(val)
                    idx = item["indexer"].get(s, -1)
                vec[start] = float(idx)

            elif dtype is list:
                arr = _coerce_list(val)
                indexer = item.get("indexer") or {}
                for s in arr:
                    j = indexer.get(str(s))
                    if j is not None:
                        vec[start + j] = 1.0

        return vec

    return converter


def ensure_pag_ready():
    """
    Build a fresh PAG with STATE settings using the package's own clients.
    Prefer the trained model if available and flagged. Also apply UI-configurable
    PAGConfig and AlignmentConfig values onto the new instance.
    """
    api_key = STATE["api_key"]
    llm_model = STATE["llm_model"]
    emb_model = STATE["emb_model"]
    schema: FeatureSchema = STATE["schema"]
    diseases = STATE["diseases"]
    symptoms = STATE["symptoms"]

    if not api_key:
        raise RuntimeError("OpenAI API Key is required.")
    if not schema or not diseases:
        raise RuntimeError("Schema and labels are required.")

    pag = PAG()
    pag.register_feature_schema(schema)

    # Predictive model (prefer trained)
    if STATE.get("use_trained") and STATE.get("trained") and STATE["trained"].get("predictor_cls"):
        predictor_cls = STATE["trained"]["predictor_cls"]
        model_obj = STATE["trained"]["model_obj"]
        predictor = predictor_cls(model_obj, labels_order=schema.labels)
        pag.register_predictive_model("trained", predictor)
        pag.register_schema_converter("trained", STATE["trained"]["schema_converter"])
    else:
        pag.register_predictive_model("default", DummyPredictor(labels=diseases))
        pag.register_schema_converter("default", FeatEncoder(symptoms))

    # LLM / Embedding
    pag.register_llm("openai", llm_model, api_key=api_key)
    pag.register_emb("openai", emb_model, api_key=api_key)

    # === Apply UI configs ===
    ui_pag = STATE.get("pag_cfg", {})
    ui_align = STATE.get("align_cfg", {})

    # PAGConfig (embed_fn is intentionally not exposed in UI)
    if hasattr(pag, "cfg"):
        pag.cfg.top_k_predictions = int(ui_pag.get("top_k_predictions", 5))
        pag.cfg.num_schema_categorical = int(ui_pag.get("num_schema_categorical", 10))
        pag.cfg.short_knowledge = bool(ui_pag.get("short_knowledge", True))
        pag.cfg.knowledge_per_label = bool(ui_pag.get("knowledge_per_label", True))
        pag.cfg.aggregate_with_pm = bool(ui_pag.get("aggregate_with_pm", False))
        pag.cfg.user_input_with_kg = bool(ui_pag.get("user_input_with_kg", False))
        pag.cfg.embedding_cache_path = str(ui_pag.get("embedding_cache_path", "./embedding_cache.pickle"))
        pag.cfg.match_cache_path = str(ui_pag.get("match_cache_path", "./match_cache.pickle"))

    # AlignmentConfig
    if hasattr(pag, "align_cfg"):
        pag.align_cfg.top_k_candidates = int(ui_align.get("top_k_candidates", 30))
        pag.align_cfg.use_semantic = bool(ui_align.get("use_semantic", True))

    STATE["pag"] = pag
    return pag


def sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
