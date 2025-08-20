"""
PAG.py — Reference implementation skeleton for Prediction‑Augmented Generation (PAG)

Key ideas and terminology are aligned with the ACL 2025 paper,
"Prediction‑Augmented Generation for Automatic Diagnosis Tasks" (Ju & Lee, 2025).
Use this file as an extensible baseline — the internals are deliberately modular
and documented so you can adapt to your domain (not limited to medicine).

Major components in this file
-----------------------------
1) Registries
   • Predictive model registry
   • LLM client registry (supports OpenAI, Ollama, custom HTTP server)
   • Feature schema / dictionary registry (for alignment)

2) Orchestration (PAG)
   • align() → predict() → knowledge() → refine() → aggregate()
   • Generates a final label plus explanations/evidence

3) Minimal demos
   • Dummy predictive models and LLMs for quick smoke tests

This file has zero hard dependencies beyond Python 3.9+.
If you want cosine similarity with real embeddings, pass your own embed() hook.

Author: Chanyang Ju
License: MIT
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, Union
from .prompts import info_extraction, extraction_check, json_reformat, term_match, knowledge_generation, rationale_generation, label_generation
from .utils.json_validator import validate_model_json, validate_json_schema
import math
import re
import json
import numpy as np
import pickle
import time

from wheel.cli.convert import normalize

from .models.Client import OllamaClient, OpenAIClient, APIClient, OpenAIEmbClient, OllamaEmbClient

# =============================
# Interfaces / Protocols
# =============================

class PredictiveModel(Protocol):
    """Interface for any predictive model you want to plug in.

    Implementations must return top‑k (label, score) pairs.
    Labels are strings; scores are floats where higher is better.
    """

    name: str

    def predict(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        ...

# =============================
# Utilities (pure‑python)
# =============================

_word_re = re.compile(r"[A-Za-z가-힣0-9_]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text)]


def lexical_overlap(a: str, b: str) -> int:
    """Count of shared word tokens (case‑insensitive)."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb)


def cosine_similarities(query_emb: List[float], target_embs: List[List[float]]) -> np.ndarray:
    """query_emb vs target_embs(each row) """
    q = np.asarray(query_emb, dtype=float)               # (D,)
    M = np.asarray(target_embs, dtype=float)             # (N, D)
    if M.ndim != 2 or q.ndim != 1 or M.shape[1] != q.shape[0]:
        raise ValueError(f"Shape mismatch: query {q.shape}, targets {M.shape}")

    # Normalization
    q_norm = q / (np.linalg.norm(q) + 1e-12)             # (D,)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)  # (N, D)

    # Cosine similarity
    sims = M_norm @ q_norm                                # (N,)
    return sims



# =============================
# Feature Schema & Dictionary
# =============================

@dataclass
class FeatureField:
    name: str
    dtype: str = "str"  # e.g., "str", "int", "float", "bool", "categorical"
    allowed_values: Optional[List[str]] = None  # for categorical / dictionary terms
    needs_alignment: bool = True # If a feature does not need aligning feature such as numeric value, set False
    description: str = ""


@dataclass
class FeatureSchema:
    """Holds the predictive model's expected input format.

    Example:
        schema = FeatureSchema()
        schema.register_feature("age", dtype="int", needs_alignment=False)
        schema.register_feature("symptoms", dtype="categorical", allowed_values=["headache", "fever", ...], needs_alignment=True)
    """

    fields: Dict[str, FeatureField] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)

    def register_feature(
            self,
            name: str,
            dtype: str = "str",
            allowed_values: Optional[List[str, int]] = None,
            needs_alignment: bool = True,
            description: str = "",
    ) -> None:
        self.fields[name] = FeatureField(name=name, dtype=dtype, allowed_values=allowed_values, needs_alignment=needs_alignment, description=description)

    def register_labels(
            self,
            allowed_values: Optional[List[str]]
    ) -> None:
        self.labels = allowed_values.copy()

    def dictionary_terms(self) -> List[str]:
        names: List[str] = []
        for f in self.fields.values():
            names.append(f.name)
        return names

    def allowed_terms(self) -> List[str]:
        terms: List[str] = []
        for f in self.fields.values():
            if f.allowed_values:
                terms.extend(f.allowed_values)
        return terms


# =============================
# Registries
# =============================

class Registry:
    def __init__(self):
        self._items: Dict[str, Any] = {}

    def add(self, name: str, obj: Any, categorical=None) -> None:
        if categorical and name not in categorical: raise Exception(f"Invalid task name, type within {categorical}")
        if name in self._items:
            raise ValueError(f"Duplicate name in registry: {name}")
        self._items[name] = obj

    def get(self, name: str) -> Any:
        if name not in self._items:
            return None
        else:
            return self._items[name]

    def names(self) -> List[str]:
        return list(self._items.keys())


# =============================
# Alignment
# =============================

@dataclass
class AlignmentConfig:
    top_k_candidates: int = 30
    use_semantic: bool = True


# =============================
# PAG Orchestrator
# =============================

@dataclass
class PAGConfig:
    top_k_predictions: int = 5
    num_schema_categorical: int = 10
    short_knowledge: bool = True
    knowledge_per_label: bool = True
    aggregate_with_pm: bool = False
    user_input_with_kg: bool = False
    # Provide a function to embed text for semantic alignment (optional)
    embed_fn: Optional[Callable[[str], List[float]]] = None
    embedding_cache_path = "./embedding_cache.pickle"
    match_cache_path = "./match_cache.pickle"


@dataclass
class PAGOutput:
    aligned_features: Dict[str, Any]
    predictive_candidates: List[Tuple[str, float]]
    knowledge: Dict[str, str]
    refined_text: Optional[str]
    refined_label: Optional[str]
    final_label: Optional[str]


class PAG:
    """High‑level façade that wires everything together.

    Usage:
        pag = PAG()
        pag.register_feature_schema(schema)
        pag.register_predictive_model("tabnet", MyTabNet(...))
        pag.register_llm("ollama", model="llama3.1:8b")
        result = pag.generate(user_input, extractor=..., model="tabnet")
    """

    def __init__(self):
        self.cfg = PAGConfig()
        self.align_cfg = AlignmentConfig()
        self.predictive_models = Registry()
        self.predictive_model_name = None
        self.llms = Registry()
        self.llm_name = None
        self.embs = Registry()
        self.emb_name = None
        self.emb_book = None
        self.emb_matrix = Registry()
        self.match_cache: Dict[str, str] = {}
        self.few_shots = Registry()
        self.schema: Optional[FeatureSchema] = None
        self.schema_converter = Registry()
        self._load_emb_book()
        self._load_match_cache()

    # ---- load initial variable ----
    def _load_emb_book(self):
        if os.path.isfile(self.cfg.embedding_cache_path):
            with open(self.cfg.embedding_cache_path, "rb") as f:
                self.emb_book = pickle.load(f)
        else:
            self.emb_book = {}
            with open(self.cfg.embedding_cache_path, "wb") as f:
                pickle.dump(self.emb_book, f)

    def _load_match_cache(self):
        if os.path.isfile(self.cfg.match_cache_path):
            with open(self.cfg.match_cache_path, "rb") as f:
                self.match_cache = pickle.load(f)
        else:
            with open(self.cfg.match_cache_path, "wb") as f:
                pickle.dump(self.match_cache, f)

    def _save_emb_book(self):
        with open(self.cfg.embedding_cache_path, "wb") as f:
            pickle.dump(self.emb_book, f)

    def _save_match_cache(self):
        with open(self.cfg.match_cache_path, "wb") as f:
            pickle.dump(self.match_cache, f)

    def make_emb_book(self):
        missing_terms = [term for term in self.schema.allowed_terms() if term not in self.emb_book]
        emb_agent = self.embs.get(self.emb_name)
        if missing_terms:
            missing_embs = emb_agent.create(self.emb_name, missing_terms)
            self.emb_book.update(dict(zip(missing_terms, missing_embs)))
            self._save_emb_book()

        for feat in self.schema.fields.values():
            if not getattr(feat, "allowed_values", None):
                continue  # Skip no allowed_values feat

            values: List[str] = list(feat.allowed_values)

            # Check missing value
            missing = [v for v in values if v not in self.emb_book]
            if missing:
                raise KeyError(
                    f"[{feat.name}] Missing value detected in emb_book: {missing}"
                )

            # order alignment
            embeddings = [self.emb_book[v] for v in values]

            # Registry: feature_name -> {"values": [...], "embeddings": [...]}
            self.emb_matrix.add(
                name=feat.name,
                obj={"values": values, "embeddings": embeddings}
            )

    def _add_emb_book(self, query, emb):
        self.emb_book[query] = emb
        self._save_emb_book()

    def add_match_cache(self, name, matched):
        self.match_cache[name] = matched
        self._save_match_cache()

    # ---- Registrations ----
    def register_feature_schema(self, schema: FeatureSchema) -> None:
        self.schema = schema

    def register_predictive_model(self, name: str, model: PredictiveModel) -> None:
        self.predictive_models.add(name, model)
        self.predictive_model_name = name

    def register_schema_converter(self, name: str, model) -> None:
        self.schema_converter.add(name, model)

    def register_llm(self, provider: str, model:str, **kwargs: Any) -> None:
        """
        provider for LLM: [openai, ollama, custom]
        openai require decode_key(to acquire text from response, ["choices",0,"message","content"] is default),
                       api_key(if you registered api_key in .env["OPENAI_API_KEY"] already, you don't have to give that)
        ollama require decode_key(to acquire text from response, default value is already set)
        custom require base_url(to request response to custom llm api server)
                       decode_key(to acquire text from response, ["message", "content"] is default)
                       api_key(if you registered api_key in .env["API_KEY"] already or don't need, you don't have to give that)
        """
        provider = provider.lower()
        if provider in ("openai", "oai"):
            client = OpenAIClient(**kwargs)
        elif provider in ("ollama",):
            client = OllamaClient(**kwargs)
        elif provider in ("http", "custom"):
            client = APIClient(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        self.llm_name = model
        self.llms.add(model, client)

    def register_emb(self, provider: str, model:str, **kwargs: Any) -> None:
        """
        provider for text embedding: [openai, ollama, custom]
        openai require decode_key(to acquire text from response, ["choices",0,"message","content"] is default),
                       api_key(if you registered api_key in .env["OPENAI_API_KEY"] already, you don't have to give that)
        ollama require decode_key(to acquire text from response, default value is already set)
        custom require base_url(to request response to custom llm api server)
                       decode_key(to acquire text from response, ["message", "content"] is default)
                       api_key(if you registered api_key in .env["API_KEY"] already or don't need, you don't have to give that)
        """
        provider = provider.lower()
        if provider in ("openai", "oai"):
            client = OpenAIEmbClient(**kwargs)
        elif provider in ("ollama",):
            client = OllamaEmbClient(**kwargs)
        elif provider in ("http", "custom"):
            client = APIClient(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        self.emb_name = model
        self.embs.add(model, client)

    def register_fewshot(self, task, text):
        self.few_shots.add(task, text, ['info_extraction',
                                        'extraction_check',
                                        'json_reformat',
                                        'term_match',
                                        'knowledge_generation',
                                        'rationale_generation',
                                        'label_generation'])



    # ---- Helpers ----
    @staticmethod
    def _rank_vote(candidates: List[str], top_k: int):
        """
        Convert a ranked list of labels into scores using Borda count:
        - index 0 → k points
        - index 1 → k-1 points
        - ...
        - index k-1 → 1 point
        """
        votes = {}
        k = min(top_k, len(candidates))
        for rank, label in enumerate(candidates[:k], start=1):
            votes[label] = k - rank + 1
        return votes

    def decode_label(self, pred):
        top_k = self.cfg.top_k_predictions
        label_decoder = self.schema.labels

        topk_idx = np.argsort(pred)[-top_k:][::-1]
        topk_labels = [label_decoder[i] for i in topk_idx]

        return topk_labels

    def _get_messages(self, contents):
        if len(contents) % 2 != 0:
            raise ValueError("contents must be [role, content, role, content, ...]")

        role_map = {
            "sys": "system",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool"  # 필요 시 도구 호출 응답에 사용
        }

        messages = []
        for i in range(0, len(contents), 2):
            role_key = str(contents[i]).lower()
            content = contents[i + 1]

            role = role_map.get(role_key)
            if role is None:
                raise ValueError(f"Unknown role '{role_key}'. Use one of {list(role_map.keys())}.")

            messages.append({"role": role, "content": str(content)})

        return messages

    def _get_schema(self):
        schema_text = ""
        for term in self.schema.dictionary_terms():
            sch = self.schema.fields.get(term)
            if type(sch.allowed_values) == list and len(sch.allowed_values) > self.cfg.num_schema_categorical:
                schema_text += f"{sch.name}: dtype={sch.dtype}, description=\"{sch.description}\""
            else:
                schema_text += f"{sch.name}: dtype={sch.dtype}, allowed_values={sch.allowed_values}, description=\"{sch.description}\""
            schema_text += "\n"

        return schema_text

    def _normalize_by_max(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        m = np.nanmax(x)
        if not np.isfinite(m) or m == 0:
            return x
        return x / m

    # ---- Core steps ----
    def extract(self, user_input):
        if self.schema is None:
            raise RuntimeError("Feature schema not registered.")

        llm_agent = self.llms.get(self.llm_name)
        data_schema = self._get_schema()
        sys_prompt = info_extraction.get_prompt(data_schema, self.few_shots.get('info_extraction'))
        messages = self._get_messages(['sys',sys_prompt,'user',user_input])
        initial_extraction = llm_agent.create(self.llm_name, messages)

        sys_prompt, user_prompt = extraction_check.get_prompt(user_input, data_schema, initial_extraction, self.few_shots.get('extraction_check'))
        messages = self._get_messages(['sys', sys_prompt, 'user', user_prompt])
        double_checked = llm_agent.create(self.llm_name, messages)

        if '<' in double_checked or 'NPB' in double_checked:
            init_feat = validate_model_json(initial_extraction)
            valid = validate_json_schema(init_feat['data'], self.schema.dictionary_terms())
            extracted_data = initial_extraction
        else:
            init_feat = validate_model_json(double_checked)
            valid = validate_json_schema(init_feat['data'], self.schema.dictionary_terms())
            extracted_data = double_checked

        if not init_feat['ok'] or not valid:
            sys_prompt, user_prompt = json_reformat.get_prompt(extracted_data, data_schema)
            messages = self._get_messages(['sys', sys_prompt, 'user', user_prompt])
            re_formed = llm_agent.create(self.llm_name, messages)
            init_feat = validate_model_json(re_formed)
            if not init_feat:
                raise Exception(f"Invalid Json format of information extraction module: {re_formed}")

        return init_feat['data']

    def _search_emb_book(self, query):
        if query in self.emb_book.keys():
            return self.emb_book[query]
        else:
            return None

    def _matching(self, feat, query, _dict_terms):
        if not _dict_terms:
            return []
        if query in _dict_terms:
            return query
        if query in self.match_cache.keys():
            return self.match_cache[query]
        # emb model
        emb_agent = self.embs.get(self.emb_name)
        llm_agent = self.llms.get(self.llm_name)

        # Searching for similar terms
        lexical_results = np.zeros(len(_dict_terms))
        dense_results = np.zeros(len(_dict_terms))
        for idx, term in enumerate(_dict_terms):
            lexical_results[idx] += lexical_overlap(query, term)
        if self.align_cfg.use_semantic:
            emb = self._search_emb_book(query)
            if emb:
                dense_results += cosine_similarities(emb, self.emb_matrix.get(feat)["embeddings"])
            else:
                emb = emb_agent.create(self.emb_name, query, return_scalar_for_single=True)
                self._add_emb_book(query, emb)
                dense_results += cosine_similarities(emb, self.emb_matrix.get(feat)["embeddings"])

        score = self._normalize_by_max(lexical_results) + self._normalize_by_max(dense_results)
        topk_idx = np.argsort(score)[-self.align_cfg.top_k_candidates:][::-1]
        vals = self.emb_matrix.get(feat)["values"]
        selected_values = [vals[i] for i in topk_idx]
        # matching the most similar term
        llm_prompt, user_prompt = term_match.get_prompt(query, selected_values, self.few_shots.get('term_match'))
        messages = self._get_messages(['sys', llm_prompt, 'user', user_prompt])
        result = llm_agent.create(self.llm_name, messages)

        if result in _dict_terms:
            self.add_match_cache(query, result)
            return result
        else:
            return selected_values[0]

    def align(
            self,
            extracted: Dict[str, Any],
    ) -> AlignmentResult:
        if self.schema is None:
            raise RuntimeError("Feature schema not registered.")
        self.make_emb_book()
        norm_feat = {}
        for feat, val in extracted.items():
            sch = self.schema.fields[feat]
            if not sch.needs_alignment:
                continue
            if type(val) == type(None):
                norm_feat[feat] = None
            elif type(val) == list:
                temp_feat = []
                for ea_val in val:
                    norm_val = self._matching(feat, ea_val, sch.allowed_values)
                    if norm_val:
                        temp_feat.append(norm_val)
                if temp_feat: norm_feat[feat] = temp_feat

            elif type(val) == str:
                norm_val = self._matching(feat, val, sch.allowed_values)
                if norm_val:
                    try:
                         norm_feat[feat] = sch.dtype(norm_val)
                    except ValueError:
                        norm_feat[feat] = norm_val

            else:
                norm_val = self._matching(feat, val, sch.allowed_values)
                if norm_val:
                    norm_feat[feat] = norm_val


        return norm_feat

    def predict(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        sch_conv = self.schema_converter.get(self.predictive_model_name)
        features = sch_conv(features)
        model: PredictiveModel = self.predictive_models.get(self.predictive_model_name)
        try:
            result = model.predict_proba(features)
        except Exception as e1:
            try:
                result = model(features)
            except Exception as e2:
                result = model.predict(features)
        return result


    def generate_knowledge(self, user_input: str, labels: List[str]) -> Dict[str, str]:
        llm_agent = self.llms.get(self.llm_name)
        sys_prompt, user_prompt = knowledge_generation.get_prompt(user_input,
                                        labels if type(labels)==str else '/n'.join(labels),
                                        self.cfg.user_input_with_kg,
                                        self.cfg.short_knowledge,
                                        self.few_shots.get('knowledge_generation'))
        messages = self._get_messages(['sys', sys_prompt, 'user', user_prompt])
        knowledge = llm_agent.create(self.llm_name, messages)

        return knowledge

    def refine_with_llm(
            self,
            user_input: str,
            prediction: List[str],
            knowledge: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        llm_agent = self.llms.get(self.llm_name)
        sys_prompt, user_prompt = rationale_generation.get_prompt(user_input, prediction, knowledge, self.few_shots.get('rationale_generation'))
        messages = self._get_messages(['sys', sys_prompt, 'user', user_prompt])
        rationale = llm_agent.create(self.llm_name, messages)

        sys_prompt, user_prompt = label_generation.get_prompt(rationale, prediction, self.few_shots.get('label_generation'))
        messages = self._get_messages(['sys', sys_prompt, 'user', user_prompt])
        label = llm_agent.create(self.llm_name, messages)

        return rationale, label

    def aggregate(self, pm: List[str], llm_label: Optional[List[str]]) -> List[str]:
        """
        Aggregate ranked label lists from PM and LLM using rank-based voting.
        Returns labels sorted from most likely to least likely,
        but only among labels that appear in the LLM's candidate list.
        """
        scores = {}

        # --- Collect votes from PM (predictive model) ---
        pm_scores = self._rank_vote(pm, top_k=self.cfg.top_k_predictions)
        for lbl, sc in pm_scores.items():
            scores[lbl] = scores.get(lbl, 0) + sc

        # --- Collect votes from LLM (if provided) ---
        if llm_label:
            llm_scores = self._rank_vote(llm_label, top_k=self.cfg.top_k_predictions)
            for lbl, sc in llm_scores.items():
                scores[lbl] = scores.get(lbl, 0) + sc
        else:
            # If LLM did not return any labels, no candidates remain
            return []

        if not scores:
            return []

        # --- Prepare for tie-breaking ---
        # Primary: higher total score is better
        # Secondary: prefer labels ranked higher in PM
        pm_index = {label: idx for idx, label in enumerate(pm)}

        sorted_labels = sorted(
            scores.items(),
            key=lambda x: (x[1], -(len(pm) - pm_index.get(x[0], len(pm)))),
            reverse=True
        )

        # --- Keep only labels that appear in LLM candidates ---
        filtered_labels = [label for label, _ in sorted_labels if label in llm_label]

        return filtered_labels

    # ---- End‑to‑end ----
    def generate(
            self,
            user_input: str
    ) -> PAGOutput:
        """Run the full PAG pipeline.

        Args:
            user_input: raw inputs (free‑form text)
        """

        initial_feature = self.extract(user_input)
        normalized_feature = self.align(initial_feature)
        proba = self.predict(normalized_feature)
        l = self.decode_label(proba)
        k = self.generate_knowledge(user_input, l) if self.cfg.knowledge_per_label else {}
        refined_text, refined_label = self.refine_with_llm(user_input, l, k)

        if self.cfg.aggregate_with_pm:
            pag_pred = self.aggregate(l, refined_label)
        else:
            pag_pred = refined_label

        return PAGOutput(
            aligned_features=normalized_feature,
            predictive_candidates=l,
            knowledge=k,
            refined_text=refined_text,
            refined_label=refined_label,
            final_label=pag_pred
        )