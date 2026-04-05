from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class DeterministicTextEncoder:
    dim: int = 256

    def __call__(self, text: str) -> List[float]:
        key = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(key[:8], byteorder="little", signed=False) % (2 ** 32 - 1)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v) + 1e-8
        return (v / n).tolist()


@dataclass
class SentenceTransformerTextEncoder:
    """
    Deterministic semantic encoder wrapper.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise ImportError(
                "sentence-transformers is required for semantic text encoding. "
                "Install it via pip install sentence-transformers."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, text: str) -> List[float]:
        v = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        if not self.normalize_embeddings:
            n = float(np.linalg.norm(v)) + 1e-8
            v = v / n
        return np.asarray(v, dtype=np.float32).tolist()


def make_text_encoder(
    encoder_type: str,
    hash_dim: int = 256,
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    st_device: str = "cpu",
) -> Callable[[str], List[float]]:
    mode = (encoder_type or "").strip().lower()
    if mode in ("semantic", "sentence_transformer", "sentence-transformer", "st"):
        return SentenceTransformerTextEncoder(model_name=st_model_name, device=st_device)
    if mode in ("hash", "deterministic"):
        return DeterministicTextEncoder(dim=hash_dim)
    raise ValueError(
        f"Unknown text encoder '{encoder_type}'. "
        f"Use one of: hash, semantic."
    )
