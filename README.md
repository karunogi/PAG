<p align="center">
  <img src="https://2025.aclweb.org/assets/images/logos/acl-logo.png" alt="ACL Logo" width="300"/>
</p>

# PAG: Prediction-Augmented Generation Framework

<p align="left">
  <a href="https://aclanthology.org/2025.findings-acl.732/">
    <img src="https://img.shields.io/badge/Read%20Paper-orange?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="ACL 2025 Paper">
  </a>
</p>

**PAG (Prediction-Augmented Generation)** combines a Predictive Model (PM) with a Large Language Model (LLM) to produce **knowledge-grounded predictions** and **explanations** from free-form user input.  
This repository implements a practical, extensible pipeline aligned with the ACL 2025(findings) paper:

> *Prediction-Augmented Generation for Automatic Diagnosis Tasks* (Ju & Lee, 2025)

---

## 🚀 Try the Local Web Demo

<a href="https://github.com/karunogi/PAG/tree/main/pag_demo">
  <img src="https://img.shields.io/badge/🚀 Launch_Demo-blueviolet?style=for-the-badge&logo=flask&logoColor=white" alt="Launch Demo">
</a>


## ✨ Highlights

- **Model-agnostic registries**
  - Register any predictive model (Sklearn, PyTorch, ONNX, custom) that exposes `predict_proba`, `predict`, or `__call__`.
  - Register multiple LLM backends (OpenAI, Ollama, or your custom HTTP API).
  - Register embedding backends for semantic alignment (OpenAI, Ollama, custom).
  - The registered model should return a NumPy array containing the probability for each label.

- **Feature Schema & Alignment**
  - Declare input fields with types and allowed values.
  - Robust alignment from free text → schema dictionary values using:
    - Exact match → Lexical overlap → Semantic similarity (embeddings)
    - LLM-assisted candidate selection
  - Embedding & match caches for speed (`embedding_cache.pickle`, `match_cache.pickle`).
  - You must register a custom encoder via pag.register_schema_converter() that can convert a structured and aligned feature dictionary into your predictive model-ready input format (e.g., one-hot vector).
See the usage example below for reference.
- **End-to-end Orchestration**
  extract → align → predict → knowledge → refine → aggregate
  - `extract`: LLM-based info extraction & JSON validation  
  - `align`: map values to schema dictionary terms  
  - `predict`: call your predictive model  
  - `knowledge`: per-label knowledge generation  
  - `refine`: rationale + final label proposal by LLM  
  - `aggregate`: optional rank voting between PM & LLM

- **Prompt modularization & few-shots**
  - All prompts live in `./prompts` and are swappable.
  - You can register few-shot exemplars per task stage.

---

## 🧱 PAG framework Structure

```
src/
  prompts/
    info_extraction.py
    extraction_check.py
    json_reformat.py
    term_match.py
    knowledge_generation.py
    rationale_generation.py
    label_generation.py
  models/
    Client.py
  utils/
    json_validator.py
  __init__.py
  PAG.py
```

---

## ⚙️ Requirements

- Python **3.10+**
- `numpy`
- (Depending on your clients) `openai`, `requests`, etc.

🔑 LLMs Environment & Credentials

- **OpenAI**: Set `OPENAI_API_KEY` in your environment or pass `api_key=...` when registering.
- **Custom HTTP backend**: Provide `base_url` and (optionally) `api_key`, plus a `decode_key` path to extract the text field from your JSON response.
- **Ollama**: Make sure your local Ollama server is running and accessible.

---

## 🚀 Quick Start

### 1) Define a schema
```python
from PAG import FeatureSchema

schema = FeatureSchema()
schema.register_feature("age", dtype=int, needs_alignment=False, description="Age in years, int type")
schema.register_feature("sex", dtype=str, allowed_values=["male", "female"])
schema.register_feature(
    "symptoms",
    dtype=list,
    description="symptoms of the patient, valid json list[str] type, comma separated",
    allowed_values=["Cough", "Fever and chills", "Shortness of breath"]
)
schema.register_labels(["Common cold", "Pneumonia", "Asthma"])
```

### 2) Wire up PAG
```python
from PAG import PAG

pag = PAG()
pag.register_feature_schema(schema)

pag.register_predictive_model("default", MyPredictor())
pag.register_schema_converter("default", MyFeatureEncoder())

pag.register_llm("openai", "gpt-4o", api_key="YOUR_OPENAI_KEY")
pag.register_emb("openai", "text-embedding-3-small", api_key="YOUR_OPENAI_KEY")
```

### 3) Run the pipeline
```python
out = pag.generate(user_input="Patient complains of chest pain and cough with fever...")
print(out.aligned_features)
print(out.predictive_candidates)
print(out.knowledge)
print(out.refined_text)
print(out.refined_label)
print(out.final_label)
```

---

## 🧩 Core Concepts

- **FeatureSchema**
  - `register_feature(name, dtype, allowed_values=None, needs_alignment=True, description="")`
  - `register_labels(list[str])`

- **Registries**
  - `register_predictive_model(name, model)`
  - `register_llm(provider, model, **kwargs)`
  - `register_emb(provider, model, **kwargs)`
  - `register_schema_converter(name, callable)`
  - `register_fewshot(task, text)`

- **Caches**
  - `embedding_cache.pickle`: stores embeddings
  - `match_cache.pickle`: memoizes alignment

---

## 🔌 Pluggable Clients

`models/Client.py` must provide:
- `OpenAIClient`, `OllamaClient`, `APIClient` for LLMs  
- `OpenAIEmbClient`, `OllamaEmbClient` for embeddings

Each client should expose:

- `create(model_name, messages=[...]) → str`
- `create(emb_model_name, texts, ...) → vector(s)`
- Handle nested JSON with `decode_key`.

---

## 🧪 Example: Minimal Predictor & Encoder

```python
class DummyPredictor:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, features):
        return np.array([0.4, 0.25, 0.1, 0.05, 0.1], dtype=np.float32)

class FeatEncoder:
    def __call__(self, structured_input: dict) -> np.ndarray:
        vector = np.zeros(2 + len(symptoms), dtype=int)
        vector[0] = structured_input.get("age", 0)
        vector[1] = {"male": 0, "female": 1}.get(structured_input.get("sex", "").lower(), 3)
        for sym in structured_input.get("symptoms", []):
            if sym in symptoms:
                idx = symptoms.index(sym)
                vector[2 + idx] = 1
            else:
                raise ValueError(f"Unknown symptom: {sym}")
        return vector
```

---

## 🧰 Full Quick-Start Script

```python
import numpy as np
from PAG import FeatureSchema, PAG

user_input = """Doctor, I’ve been feeling really unwell..."""
diseases = ["Common cold", "Pneumonia", "Diabetes mellitus", "Hypertension", "Asthma"]
symptoms = ["Runny nose", "Sneezing", "Sore throat", "Cough", "Mild headache", "Fatigue", "Cough with phlegm", "Fever and chills", "Shortness of breath", "Chest pain when breathing or coughing", "Nausea or vomiting", "Frequent urination", "Excessive thirst", "Increased hunger", "Blurred vision", "Slow-healing wounds", "Numbness or tingling in hands/feet", "Headache", "Dizziness", "Nosebleeds", "Wheezing", "Chest tightness"]


class DummyPredictor:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, features):
        return np.array([0.4, 0.25, 0.1, 0.05, 0.1], dtype=np.float32)

class DummyEncoder:

    def __call__(self, structured_input: dict) -> np.ndarray:
        """
        Encode patient data (dict) into a one-hot vector based on schema:
        index 0: age
        index 1: sex (male=0, female=1)
        index 2~: symptoms (one-hot encoding)
        """
        print(structured_input)
        # One-hot vector length = |age| + |sex| + |symptoms|
        vector = np.zeros(2 + len(symptoms), dtype=int)

        # age
        if type(structured_input["age"]) == type(None):
            vector[0] = 0
        else:
            vector[0] = structured_input["age"]

        # sex
        if type(structured_input["sex"]) == type(None):
            vector[1] = 3
        else:
            sex = structured_input["sex"].lower()
            if sex == "male":
                vector[1] = 0
            elif sex == "female":
                vector[1] = 1


        # symptoms one-hot
        for sym in structured_input.get("symptoms", []):
            if sym in symptoms:
                idx = symptoms.index(sym)
                vector[2 + idx] = 1
            else:
                raise ValueError(f"Unknown symptom: {sym}")

        return vector

if __name__ == "__main__":
    schema = FeatureSchema()
    schema.register_feature("age", dtype=int, description="Age in years, int type")
    schema.register_feature("sex", dtype=str, allowed_values=["male", "female"])
    schema.register_feature("symptoms", dtype=list, description="symptoms of the patient, valid json list[str] type, comma separated", allowed_values=symptoms)
    schema.register_labels(diseases)

    pag = PAG()
    pag.register_feature_schema(schema)
    pag.register_predictive_model("default", DummyPredictor(labels=diseases))
    pag.register_schema_converter("default", DummyEncoder())
    pag.register_llm("openai", "gpt-5-nano", api_key="YOUR_OPENAI_API_KEY")
    pag.register_emb("openai", "text-embedding-3-small", api_key="YOUR_OPENAI_API_KEY")

    out = pag.generate(user_input=user_input)
    print("Aligned features:", out.aligned_features)
    print("Predictive candidates:", out.predictive_candidates)
    print("Knowledge:", out.knowledge)
    print("Refined rationale:", out.refined_text)
    print("Refined label:", out.refined_label)
    print("Final label:", out.final_label)
```

---

## 🙌 Acknowledgements

Design & implementation: Chanyang Ju  
License: MIT © 2025

##  Paper Citation

**Chan‑Yang Ju and Dong‑Ho Lee. 2025. Prediction‑Augmented Generation for Automatic Diagnosis Tasks.**  
In *Findings of the Association for Computational Linguistics: ACL 2025*, pages 14225–14246, Vienna, Austria. Association for Computational Linguistics. :contentReference[oaicite:1]{index=1}

###  BibTeX
```bibtex
@inproceedings{ju-lee-2025-prediction,
  title     = "Prediction-Augmented Generation for Automatic Diagnosis Tasks",
  author    = "Ju, Chan‑Yang and Lee, Dong‑Ho",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
  month     = jul,
  year      = "2025",
  address   = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages     = "14225--14246",
  url       = "https://aclanthology.org/2025.findings-acl.732/"
}
```
