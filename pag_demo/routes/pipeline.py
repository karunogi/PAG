import json
import uuid
from flask import Blueprint, request, jsonify, session, Response

from ..services.utils import ensure_pag_ready, sse_event

bp = Blueprint("pipeline", __name__)


@bp.post("/start")
def start():
    data = request.get_json(force=True)
    user_input = data.get("user_input","").strip()
    if not user_input:
        return jsonify({"error": "user_input is empty"}), 400
    token = uuid.uuid4().hex
    session["run_token"] = token
    session["user_input"] = user_input
    return jsonify({"token": token})


@bp.get("/stream")
def stream():
    token = request.args.get("token","")
    if not token or token != session.get("run_token"):
        return "invalid token", 400

    user_input = session.get("user_input","")
    if not user_input:
        return "no input", 400

    def run_steps():
        # Build/refresh PAG
        try:
            pag = ensure_pag_ready()
        except Exception as e:
            yield sse_event({"type":"log","step":"error","payload": f"{e}"})
            yield sse_event({"type":"final","final_label": None, "refined_text": None})
            return

        def safe(step_name, func):
            try:
                yield sse_event({"type":"log","step":step_name,"payload":f"running {step_name}..."})
                result = func()
                yield sse_event({"type":"log","step":step_name,"payload": result})
                return True, result
            except Exception as e:
                yield sse_event({"type":"log","step":f"{step_name}_error","payload": f"{e}"})
                yield sse_event({"type":"final","final_label": None, "refined_text": None})
                return False, None

        # 1) extract
        ok, extracted = (yield from safe("extract", lambda: pag.extract(user_input)))
        if not ok: return

        # 2) align
        ok, aligned = (yield from safe("align", lambda: pag.align(extracted)))
        if not ok: return

        # 3) predict
        def _predict():
            proba = pag.predict(aligned)
            labels = pag.decode_label(proba)
            to_list = proba.tolist() if hasattr(proba, "tolist") else proba
            return {"proba": to_list, "top_labels": labels}
        ok, pred_payload = (yield from safe("predict", _predict))
        if not ok: return
        labels = pred_payload["top_labels"]

        # 4) knowledge
        knowledge = {}
        if pag.cfg.knowledge_per_label:
            ok, knowledge = (yield from safe("knowledge", lambda: pag.generate_knowledge(user_input, labels)))
            if not ok: return

        # 5) refine
        def _refine():
            refined_text, refined_label = pag.refine_with_llm(user_input, labels, knowledge)
            return {"rationale": refined_text, "labels_from_llm": refined_label}
        ok, refine_payload = (yield from safe("refine", _refine))
        if not ok: return
        refined_text = refine_payload["rationale"]
        refined_label = refine_payload["labels_from_llm"]

        # 6) aggregate
        final_label = None
        if pag.cfg.aggregate_with_pm:
            ok, agg_payload = (yield from safe("aggregate", lambda: {"ranked": pag.aggregate(labels, refined_label)}))
            if not ok: return
            ranked = agg_payload["ranked"]
            final_label = ranked[0] if ranked else None
        else:
            if isinstance(refined_label, list) and refined_label:
                final_label = refined_label[0]
            elif isinstance(refined_label, str):
                final_label = refined_label
            else:
                final_label = labels[0] if labels else None

        yield sse_event({"type":"final","final_label": final_label, "refined_text": refined_text})

    return Response(run_steps(), mimetype="text/event-stream")
