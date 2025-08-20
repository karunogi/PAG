from typing import Any, Dict

# In-memory state (single-user demo)
STATE: Dict[str, Any] = {
    "api_key": "",
    "llm_model": "gpt-4o-mini",          # Can be changed by user input
    "emb_model": "text-embedding-3-small",
    "schema": None,                      # FeatureSchema
    "diseases": [],
    "symptoms": [],
    "pag": None,                         # PAG instance
    "uploaded_df": None,                 # Uploaded original data (pd.DataFrame)
    "dataset": None,                     # {"X": df_features, "y": series_labels}
    "trained": None,                     # {"model_name": str, "model_obj": sklearn model, "converter": callable}
    "use_trained": False,                # prefer trained model
    "pag_cfg": {
        "top_k_predictions": 5,
        "num_schema_categorical": 10,
        "short_knowledge": True,
        "knowledge_per_label": True,
        "aggregate_with_pm": False,
        "user_input_with_kg": False,
        "embedding_cache_path": "./embedding_cache.pickle",
        "match_cache_path": "./match_cache.pickle",
    },
    "align_cfg": {
        "top_k_candidates": 30,
        "use_semantic": True,
    },
}
