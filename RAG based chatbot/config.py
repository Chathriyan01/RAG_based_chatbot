"""
Configuration module for the Document Q&A System.
Centralizes all hyperparameters and settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Model Settings ──────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-large"
    EMBEDDING_DIM: int = 384

    # ── Chunking Settings ───────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SEPARATORS: list = None

    # ── Retrieval Settings ──────────────────────
    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.5

    # ── Generation Settings ─────────────────────
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.3
    REPETITION_PENALTY: float = 1.1

    # ── Paths ───────────────────────────────────
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    UPLOAD_DIR: str = "./data/uploads"

    def __post_init__(self):
        if self.SEPARATORS is None:
            self.SEPARATORS = ["\n\n", "\n", ". ", " "]
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)


config = Config()