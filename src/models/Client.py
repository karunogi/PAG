from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Sequence, Union, Optional
import os
import requests
import ollama
import math
try:
    from ollama import Client
except Exception:
    Client = None


def extract(response, decode_key=None):
    """
    Extract the text content from an OpenAI ChatCompletion response
    according to the decode_key path.

    response: OpenAI SDK object or dict
    decode_key: list of keys/indexes to follow, e.g. ["choices", 0, "message", "content"]
    """
    # 1) Normalize response into a Python dictionary
    if hasattr(response, "model_dump"):
        # OpenAI SDK v1 returns a pydantic model, use model_dump()
        data = response.model_dump()
    elif hasattr(response, "json"):
        # Sometimes .json() may return a JSON string, ensure we parse it
        raw = response.json()
        data = json.loads(raw) if isinstance(raw, str) else raw
    elif isinstance(response, dict):
        # Already a dict
        data = response
    else:
        raise TypeError(f"Unsupported response type: {type(response)}")

    # 2) Traverse the response following decode_key
    if decode_key:
        try:
            cur = data
            for k in decode_key:
                if isinstance(k, str):
                    # Expect dictionary before string key
                    if not isinstance(cur, dict):
                        raise TypeError(f"Expected dict before key '{k}', got {type(cur)}")
                    cur = cur.get(k)
                elif isinstance(k, int):
                    # Expect list/tuple before integer index
                    if not isinstance(cur, (list, tuple)):
                        raise TypeError(f"Expected list before index {k}, got {type(cur)}")
                    cur = cur[k]
                else:
                    raise TypeError(f"Unsupported key type in decode_key: {type(k)}")
            return cur
        except Exception as e:
            print(f"api response decode_key is not available: {e}")
            return None
    else:
        # Default: try to get the usual path choices[0].message.content
        try:
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return None

def l2_normalize(vec: List[float]) -> List[float]:
    """Return the L2-normalized version of a vector."""
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

class APIClient:
    def __init__(self, base_url, decode_key, api_key=None):
        """
        Initializes the client with the base URL of the server.

        Args:
            base_url (str): The base URL of the main server.
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("API_KEY")
        self.base_url = base_url
        self.decode_key = decode_key

    def create(self, model, messages):
        data = {
            'model': model,
            'messages': messages
        }
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            # Send a POST request with JSON data
            response = requests.post(f'{self.base_url}/run', json=data, headers=headers)
        except requests.exceptions.RequestException as e:
            raise Exception(e)

        # Check if the response was successful
        if response.status_code == 200:
            # Return the response as a dictionary
            return extract(response, self.decode_key)
        else:
            raise Exception(f"POST request failed with status code {response.status_code}")

class OllamaClient:
    def __init__(self, decode_key=None):
        self.decode_key = decode_key or ["message","content"]

    def create(self, model, messages):
        try:
            response = ollama.generate(
                model=model,
                messages=messages
            )
            print(response)
            return extract(response, self.decode_key)
        except Exception as e:
            print(e)


class OpenAIClient:
    def __init__(self, decode_key=None, api_key=None):
        """
        Initialize OpenAI client with optional custom decode_key.
        If api_key is not provided, try to load it from environment.
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Default decode path for ChatCompletion response
        self.decode_key = decode_key or ["choices", 0, "message", "content"]

    def create(self, model, messages):
        """
        Create a chat completion and extract its text output.

        model: OpenAI model name (e.g. "gpt-4o-mini")
        messages: list of dicts with {"role": ..., "content": ...}
        """
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages)
            out = extract(resp, self.decode_key)
            if out is None:
                # Better to raise error than return None silently
                raise RuntimeError("Failed to decode OpenAI response (out is None).")
            return out
        except Exception as e:
            # Do not swallow errors, propagate to caller
            raise

class OpenAIEmbClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        # Load API key from environment if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def create(
        self,
        model: Optional[str],
        inputs: Union[str, Sequence[str]],
        normalize: bool = False,
        return_scalar_for_single: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for one or more input strings.

        Args:
            model: embedding model name; defaults to self.default_model.
            inputs: a single string or a list of strings.
            normalize: if True, return L2-normalized vectors.
            return_scalar_for_single:
                - If True: when input is a single string, return just one vector.
                - If False: always return a list of vectors (even for single input).

        Returns:
            - List[float] if a single input and return_scalar_for_single=True
            - List[List[float]] otherwise
        """
        try:
            model = model

            # Detect single input vs. list input
            is_single = isinstance(inputs, str)
            batch: List[str] = [inputs] if is_single else list(inputs)

            # Call the embeddings API
            resp = self.client.embeddings.create(
                model=model,
                input=batch
            )

            # Extract embedding vectors from the response
            vectors: List[List[float]] = [d.embedding for d in resp.data]

            # Optionally apply L2 normalization
            if normalize:
                vectors = [l2_normalize(v) for v in vectors]

            # Return shape based on input and option
            if is_single and return_scalar_for_single:
                return vectors[0]
            return vectors

        except Exception as e:
            # In production, prefer logging instead of print
            print(f"[OpenAIEmbClient] Error: {e}")
            raise

class OllamaEmbClient:
    def __init__(
        self,
        host: Optional[str] = None,
    ):
        """
        Minimal embedding client backed by the 'ollama' Python package.

        Args:
            host: Ollama server base URL (e.g., 'http://localhost:11434').
                  If None, uses OLLAMA_HOST environment variable (if set),
                  otherwise falls back to the default used by the 'ollama' package.
        """
        self.host = host or os.getenv("OLLAMA_HOST")
        # If a host is provided (or env set) and Client is available, use it.
        self._use_client = bool(self.host and Client is not None)
        self.client = Client(host=self.host) if self._use_client else None

    def create(
        self,
        model: Optional[str],
        inputs: Union[str, Sequence[str]],
        normalize: bool = False,
        return_scalar_for_single: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for one or more texts using Ollama.

        Args:
            model: Model name; defaults to self.default_model.
            inputs: A single string or a sequence of strings.
            normalize: If True, L2-normalize each embedding vector.
            return_scalar_for_single:
                - True: if inputs is a single string, return List[float]
                - False: always return List[List[float]]

        Returns:
            List[float] if a single input and return_scalar_for_single=True,
            otherwise List[List[float]].
        """
        is_single = isinstance(inputs, str)
        batch: List[str] = [inputs] if is_single else list(inputs)

        # Call the package API
        if self._use_client:
            resp = self.client.embed(
                model=model,
                input=batch if len(batch) > 1 else batch[0],
            )
        else:
            # Top-level function uses default host resolution (env OLLAMA_HOST if set)
            resp = ollama.embed(
                model=model,
                input=batch if len(batch) > 1 else batch[0],
            )

        # Expected shape: {"embeddings": [[...], [...], ...]}
        embeddings = resp.get("embeddings")
        # If a single vector was returned as 1D, normalize to 2D for consistency
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], float):
            vectors = [embeddings]
        else:
            vectors = embeddings or []

        if normalize:
            vectors = [l2_normalize(v) for v in vectors]

        if is_single and return_scalar_for_single:
            return vectors[0]
        return vectors