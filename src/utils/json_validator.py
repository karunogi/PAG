import json
import re
import ast  # NEW: Python-literal fallback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

Json = Union[dict, list, str, int, float, bool, None]

class JSONValidationError(Exception):
    """Represents a (optional) schema validation failure."""


def extract_json_block(text: str) -> Optional[str]:
    """
    Extracts the *first* balanced JSON object/array from an arbitrary string.
    - Uses bracket balancing while respecting string literals and escapes.
    - Handles fenced code blocks like ```json ... ``` by unwrapping first.
    - Returns the JSON substring if found; otherwise None.
    """
    if not isinstance(text, str):
        return None

    # Unwrap fenced code block if the entire string is one
    fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
    m = fence.match(text.strip())
    if m:
        text = m.group(1)

    s = text
    start = -1
    stack = []
    in_str = False
    escape = False

    pairs = {"{": "}", "[": "]"}
    openers = set(pairs.keys())
    closers = set(pairs.values())

    for i, ch in enumerate(s):
        if start == -1:
            if ch in openers:
                start = i
                stack = [pairs[ch]]
                in_str = False
                escape = False
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in openers:
            stack.append(pairs[ch])
        elif ch in closers:
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return s[start:i+1]

    return None


def _type_name(val: Any) -> str:
    if val is None: return "null"
    if isinstance(val, bool): return "boolean"
    if isinstance(val, int) and not isinstance(val, bool): return "integer"
    if isinstance(val, float): return "number"
    if isinstance(val, str): return "string"
    if isinstance(val, list): return "array"
    if isinstance(val, dict): return "object"
    return type(val).__name__


def _check_format(value: Any, fmt: str) -> bool:
    """Very light 'format' checks (date, date-time, email, uri)."""
    if not isinstance(value, str):
        return False
    try:
        if fmt == "date":
            datetime.strptime(value, "%Y-%m-%d")
            return True
        if fmt == "date-time":
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
    except Exception:
        return False

    if fmt == "email":
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))
    if fmt == "uri":
        return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://\S+$", value))
    return True


def _validate_schema(data: Json, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """
    Minimal JSON Schema subset validator.
    Supported: type, enum, object{properties,required,additionalProperties},
               array{items,minItems,maxItems,uniqueItems},
               string{minLength,maxLength,pattern,format},
               number/integer{minimum,maximum}.
    """
    errors: List[str] = []

    def err(msg: str):
        errors.append(f"{path}: {msg}")

    typ = schema.get("type")
    if typ:
        allowed = typ if isinstance(typ, list) else [typ]
        actual = _type_name(data)
        if actual not in allowed and not (actual == "integer" and "number" in allowed):
            err(f"type mismatch (expected {allowed}, got {actual})")
            return errors

    if "enum" in schema and data not in schema["enum"]:
        err(f"enum mismatch (allowed: {schema['enum']})")

    actual_type = _type_name(data)

    if actual_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        addl = schema.get("additionalProperties", True)

        for k in required:
            if not isinstance(k, str):
                continue
            if not isinstance(data, dict) or k not in data:
                err(f"missing required key '{k}'")

        if isinstance(data, dict):
            for k, v in data.items():
                if k in props:
                    sub_schema = props[k]
                    errors.extend(_validate_schema(v, sub_schema, path=f"{path}.{k}"))
                else:
                    if addl is False:
                        errors.append(f"{path}: unknown key '{k}' (additionalProperties=false)")
        else:
            err("not an object")

    elif actual_type == "array":
        items = schema.get("items")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        unique = schema.get("uniqueItems", False)

        if min_items is not None and len(data) < min_items:
            err(f"array shorter than minItems={min_items} (len={len(data)})")
        if max_items is not None and len(data) > max_items:
            err(f"array longer than maxItems={max_items} (len={len(data)})")
        if unique and len(data) != len(set(map(lambda x: json.dumps(x, sort_keys=True), data))):
            err("array violates uniqueItems=true")

        if items:
            for idx, item in enumerate(data):
                errors.extend(_validate_schema(item, items, path=f"{path}[{idx}]"))

    elif actual_type in ("number", "integer"):
        if "minimum" in schema and data < schema["minimum"]:
            err(f"minimum violated ({data} < {schema['minimum']})")
        if "maximum" in schema and data > schema["maximum"]:
            err(f"maximum violated ({data} > {schema['maximum']})")

    elif actual_type == "string":
        if "minLength" in schema and len(data) < schema["minLength"]:
            err(f"string shorter than minLength={schema['minLength']} (len={len(data)})")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            err(f"string longer than maxLength={schema['maxLength']} (len={len(data)})")
        if "pattern" in schema and not re.search(schema["pattern"], data):
            err(f"pattern mismatch (/{schema['pattern']}/)")
        if "format" in schema and not _check_format(data, schema["format"]):
            err(f"format mismatch ({schema['format']})")

    return errors


def _normalize_jsonish(obj: Any) -> Json:
    """
    Normalize Python-literal structures into JSON-compatible ones:
    - Convert tuples to lists.
    - Recursively normalize dict keys/values and list items.
    - Leave other scalar types as-is (int/float/bool/None/str).
    """
    if isinstance(obj, tuple):
        return [_normalize_jsonish(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize_jsonish(x) for x in obj]
    if isinstance(obj, dict):
        # JSON keys must be strings; we coerce non-strings to str
        return {str(k): _normalize_jsonish(v) for k, v in obj.items()}
    return obj  # scalar


def validate_model_json(
    raw: Union[str, Json],
    schema: Optional[Dict[str, Any]] = None,
    *,
    forbid_trailing_text: bool = False,
) -> Dict[str, Any]:
    """
    Validates whether a generative model's output is valid JSON and (optionally) conforms to a schema.
    Returns: { ok: bool, errors: [..], data: Json|None, extracted: str|None }
    - If `raw` is a string: extract JSON block then parse.
    - If `raw` is dict/list: use as-is and only validate schema.
    - If `forbid_trailing_text=True`: fail when extra non-JSON text surrounds the JSON.
    - NEW: If JSON parsing fails, fallback to Python-literal parse (ast.literal_eval) and normalize.
    """
    errors: List[str] = []
    extracted = None
    parsed: Optional[Json] = None

    if isinstance(raw, (dict, list, str, int, float, bool)) or raw is None:
        if isinstance(raw, (dict, list)):
            parsed = raw
        elif isinstance(raw, str):
            text = raw

            block = extract_json_block(text)
            if block is None:
                errors.append("no valid JSON object/array found in text")
                return {"ok": False, "errors": errors, "data": None, "extracted": None}

            extracted = block

            if forbid_trailing_text:
                stripped = text.strip()
                fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
                fm = fence.match(stripped)
                inner = fm.group(1) if fm else stripped
                if inner.strip() != block.strip():
                    errors.append("extra non-JSON text detected (forbid_trailing_text=True)")

            try:
                parsed = json.loads(block)
            except json.JSONDecodeError as e_json:
                # Fallback: accept Python dict/list literal syntax (single quotes, True/False/None, etc.)
                try:
                    py_obj = ast.literal_eval(block)
                    parsed = _normalize_jsonish(py_obj)
                except Exception as e_py:
                    errors.append(
                        f"JSON parse error: {e_json.msg} (line {e_json.lineno} col {e_json.colno}); "
                        f"also failed Python-literal parse: {type(e_py).__name__}: {e_py}"
                    )
                    return {"ok": False, "errors": errors, "data": None, "extracted": extracted}
        else:
            parsed = raw
    else:
        errors.append("unsupported input type")
        return {"ok": False, "errors": errors, "data": None, "extracted": None}

    if schema:
        errors.extend(_validate_schema(parsed, schema, path="$"))

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "data": parsed,
        "extracted": extracted,
    }

def validate_json_schema(
        raw: Union[str, Json],
        terms: List[str]
):
    for k in raw.keys():
        if k not in terms:
            return False
    return True
