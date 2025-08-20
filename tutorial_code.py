import numpy as np

from src.PAG import FeatureSchema, PAG


# =============================
# Quick‑start example
# =============================
user_input = """Doctor, I’ve been feeling really unwell for the past week. I have a persistent cough that brings up thick phlegm, and I often get chills with a high fever. Breathing has become harder, especially when I try to walk even a short distance, and I feel sharp pain in my chest whenever I take a deep breath or cough. I feel extremely tired all the time, and yesterday I even felt nauseous and vomited. I’m really worried because it seems to be getting worse."""
diseases = ["Common cold", "Pneumonia", "Diabetes mellitus", "Hypertension", "Asthma"]
symptoms = [
    "Runny nose",
    "Sneezing",
    "Sore throat",
    "Cough",
    "Mild headache",
    "Fatigue",
    "Cough with phlegm",
    "Fever and chills",
    "Shortness of breath",
    "Chest pain when breathing or coughing",
    "Nausea or vomiting",
    "Frequent urination",
    "Excessive thirst",
    "Increased hunger",
    "Blurred vision",
    "Slow-healing wounds",
    "Numbness or tingling in hands/feet",
    "Headache",
    "Dizziness",
    "Nosebleeds",
    "Wheezing",
    "Chest tightness"
]
class DummyPredictor:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, features):
        return np.array([0.4, 0.25, 0.1, 0.05, 0.1], dtype=np.float32)

class FeatEncoder:

    def __call__(self, structured_input: dict) -> np.ndarray:
        """
        Encode patient data (dict) into a one-hot vector based on schema:
        index 0: age
        index 1: sex (male=0, female=1)
        index 2~: symptoms (one-hot encoding)
        """
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
    # 1) Define schema and dictionary

    schema = FeatureSchema()
    schema.register_feature("age", dtype=int, description="Age in years, int type")
    schema.register_feature("sex", dtype=str, allowed_values=["male", "female"])
    schema.register_feature(
        "symptoms",
        dtype=list,
        description="symptoms of the patient, valid json list[str] type, comma separated",
        allowed_values=symptoms
    )
    schema.register_labels(diseases)

    # 2) Wire PAG
    pag = PAG()
    pag.register_feature_schema(schema)

    # Register a predictive model (replace with your model wrapper)
    pag.register_predictive_model("default", DummyPredictor(labels=diseases))
    pag.register_schema_converter("default", FeatEncoder())


    # Register a trivial LLM (replace with OpenAI/Ollama/HTTP client)
    # You have to add your api_key in augment or .env
    pag.register_llm("openai", "gpt-5-nano", api_key=api_key)
    pag.register_emb("openai", "text-embedding-3-small", api_key=api_key)

    # 4) Run
    out = pag.generate(
        user_input=user_input
    )

