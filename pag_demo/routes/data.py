import io
import json
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np

from ..state import STATE

bp = Blueprint("data", __name__)


@bp.post("/upload_data")
def upload_data():
    """
    Upload CSV / Excel / JSON → store as DataFrame + return preview.
    For JSON:
      - Top-level list[dict], or
      - Provide json_key pointing to list[dict] inside top-level dict.
    """
    f = request.files.get("file")
    json_key = request.form.get("json_key", "").strip()
    if not f:
        return jsonify({"ok": False, "error":"No file found."}), 400

    filename = (f.filename or "").lower()
    try:
        if filename.endswith(".csv"):
            buf = io.BytesIO(f.read())
            df = pd.read_csv(buf)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            buf = io.BytesIO(f.read())
            df = pd.read_excel(buf)

        elif filename.endswith(".json"):
            raw = f.read()
            obj = json.loads(raw)

            records = None
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict):
                if not json_key:
                    return jsonify({"ok": False, "error": "For JSON, top-level must be a list or you must provide json_key."}), 400
                if json_key not in obj:
                    return jsonify({"ok": False, "error": f"json_key '{json_key}' not found in JSON."}), 400
                records = obj[json_key]
            else:
                return jsonify({"ok": False, "error": "Unsupported JSON structure. Must be list or dict."}), 400

            if not isinstance(records, list) or not records or not isinstance(records[0], dict):
                return jsonify({"ok": False, "error": "records must be list[dict]."}), 400

            df = pd.DataFrame(records)

        else:
            return jsonify({"ok": False, "error":"Unsupported extension (.csv/.xlsx/.xls/.json only)."}), 400

    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to parse file: {e}"}), 400

    if df.shape[1] < 2:
        return jsonify({"ok": False, "error":"At least 2 columns are required (last column = label)."}), 400

    STATE["uploaded_df"] = df

    preview = df.head(10).to_dict(orient="records")
    cols = list(df.columns)

    return jsonify({
        "ok": True,
        "columns": cols,
        "preview": preview,
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
    })


@bp.post("/register_data")
def register_data():
    """
    Uploaded DF → build (X,y) + auto-generate schema → return(JSON)
      - Last column = label
      - dtype inferred as int|float|str|list
      - allowed_values:
          * str: unique values (<=200)
          * list: union of all unique items (<=200)
    """
    df = STATE.get("uploaded_df")
    if df is None:
        return jsonify({"ok": False, "error": "No uploaded data found."}), 400

    cols = list(df.columns)
    if len(cols) < 2:
        return jsonify({"ok": False, "error":"At least 2 columns are required (last column = label)."}), 400

    label_col = cols[-1]
    feat_cols = cols[:-1]

    def coerce_list_cell(v):
        """
        Normalize list-like cell values.
        Accept:
          - real list
          - JSON-like "[a, b]" or "(a, b)" strings
          - newline/comma/semicolon/pipe separated strings
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON-like list/tuple first
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    parsed = json.loads(s.replace("(", "[").replace(")", "]"))
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            # Fallback: smart split on separators (newline, comma, semicolon, pipe, Japanese comma)
            parts = re.split(r"[\r\n,;|、，]+", s)
            return [p.strip() for p in parts if p and p.strip()]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        return [v]

    features = []
    for c in feat_cols:
        s = df[c]

        # detect list-like
        is_list_col = False
        for sv in s.head(20).tolist():
            if isinstance(sv, list):
                is_list_col = True; break
            if isinstance(sv, str):
                ss = sv.strip()
                if (ss.startswith("[") and ss.endswith("]")) or (ss.startswith("(") and ss.endswith(")")):
                    is_list_col = True; break

        if is_list_col:
            col_lists = s.apply(coerce_list_cell)
            flat = []
            for arr in col_lists:
                for it in (arr or []):
                    flat.append(str(it))
            uniques = sorted(set(flat))[:200]
            feat = {"name": c, "dtype": "list", "needs_alignment": True}
            if uniques:
                feat["allowed_values"] = uniques
            features.append(feat)
            continue

        # numeric / string inference
        if pd.api.types.is_integer_dtype(s):
            features.append({"name": c, "dtype": "int", "needs_alignment": False})
            continue

        if pd.api.types.is_float_dtype(s):
            features.append({"name": c, "dtype": "float", "needs_alignment": False})
            continue

        dtype = "str"
        try:
            uniques = s.dropna().astype(str).unique().tolist()
        except Exception:
            uniques = [str(x) for x in s.dropna().tolist()]
        allowed = uniques[:200] if len(uniques) <= 200 else []
        feat = {"name": c, "dtype": dtype, "needs_alignment": True}
        if allowed:
            feat["allowed_values"] = allowed
        features.append(feat)

    labels_unique = df[label_col].dropna().astype(str).unique().tolist()
    labels = labels_unique[:10000]
    schema_json = {"features": features, "labels": labels}

    X = df[feat_cols].copy()
    for c in feat_cols:
        if any(isinstance(v, list) for v in X[c].head(20)) or any(
            isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]")
            for v in X[c].head(20)
        ):
            X[c] = X[c].apply(coerce_list_cell)

    y = df[label_col].astype(str)
    STATE["dataset"] = {"X": X, "y": y, "feature_cols": feat_cols, "label_col": label_col}

    return jsonify({"ok": True, "schema": schema_json})
