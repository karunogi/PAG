from typing import List
import numpy as np

class DummyPredictor:
    """Simple fixed-probability predictor for demo."""
    def __init__(self, labels: List[str]):
        self.labels = labels

    def predict(self, features):
        # Return a vector-like list/np.array; PAG.decode_label will map to labels
        return np.array([0.4, 0.25, 0.1, 0.05, 0.1], dtype=np.float32)


class FeatEncoder:
    """
    Encode a simple {age:int, sex:str, symptoms:list[str]} structure into a vector:
      index 0: age
      index 1: sex (male=0, female=1, unknown=3)
      index 2~: symptoms one-hot
    """
    def __init__(self, symptoms: List[str]):
        self.symptoms = symptoms

    def __call__(self, structured_input: dict) -> np.ndarray:
        vector = np.zeros(2 + len(self.symptoms), dtype=int)

        # age
        if structured_input.get("age") is None:
            vector[0] = 0
        else:
            vector[0] = int(structured_input["age"])

        # sex
        sex_val = structured_input.get("sex")
        if sex_val is None:
            vector[1] = 3
        else:
            s = str(sex_val).lower()
            if s == "male":
                vector[1] = 0
            elif s == "female":
                vector[1] = 1
            else:
                vector[1] = 3

        # symptoms
        for sym in structured_input.get("symptoms", []):
            if sym in self.symptoms:
                idx = self.symptoms.index(sym)
                vector[2 + idx] = 1
        return vector


class TrainedPredictor:
    """
    Wrap a scikit-learn classifier to return probabilities aligned to schema.labels order.
    """
    def __init__(self, model, labels_order: List[str]):
        self.model = model
        self.labels_order = labels_order

    def predict(self, vec):
        import numpy as np
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec.reshape(1, -1))[0]
            if hasattr(self.model, "classes_"):
                cls = [str(c) for c in self.model.classes_]
                prob_map = {c: p for c, p in zip(cls, proba)}
                out = np.array([prob_map.get(lbl, 0.0) for lbl in self.labels_order], dtype=float)
                if np.all(out == 0):
                    out = np.ones(len(self.labels_order), dtype=float) / max(1, len(self.labels_order))
                return out
            return proba
        if len(self.labels_order) == 0:
            return np.array([1.0])
        return np.ones(len(self.labels_order), dtype=float) / len(self.labels_order)
