"""Shared embedding backends for indexing and retrieval."""

import hashlib
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

DEFAULT_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "256"))
PROVIDER_MARKER = "embedding_provider.txt"


class LocalHashEmbeddings(Embeddings):
    """Small offline embedding fallback based on feature hashing."""

    def __init__(self, dim: int = LOCAL_EMBEDDING_DIM):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vector.tolist()

        # Hashing keeps the fallback dependency-light while still producing stable vectors.
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def get_embeddings(provider: str | None = None) -> tuple[Embeddings, str]:
    """Resolve the requested embedding backend and return both object and provider name."""
    resolved = (provider or DEFAULT_PROVIDER).lower()

    if resolved == "local":
        return LocalHashEmbeddings(), "local"

    if resolved == "google":
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"), "google"

    if resolved == "auto":
        if os.getenv("GOOGLE_API_KEY"):
            return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"), "google"
        return LocalHashEmbeddings(), "local"

    raise ValueError(f"Unsupported embedding provider: {resolved}")


def save_provider_marker(index_dir: Path, provider: str) -> None:
    """Persist the embedding provider so retrieval uses the same vector space later."""
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / PROVIDER_MARKER).write_text(provider, encoding="utf-8")


def load_provider_marker(index_dir: Path) -> str | None:
    """Read the provider marker if an index has already been built."""
    marker = index_dir / PROVIDER_MARKER
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip() or None
    return None
