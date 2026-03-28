"""
Embedding Module
Generates sentence embeddings using transformer models
and manages the FAISS vector store.
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from config import config


class EmbeddingEngine:
    """Manages embeddings and FAISS vector index."""

    def __init__(self):
        print("🧠 Loading embedding model...")
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL
        )
        self.dimension = config.EMBEDDING_DIM
        self.index = None
        self.chunks = []

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense vectors."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype="float32")

    def build_index(self, chunks: List) -> None:
        """Build FAISS index from document chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = self.encode(texts)

        # Build FAISS index with L2 distance
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)
        print(f"📦 Indexed {self.index.ntotal} vectors")

    def save_index(self) -> None:
        """Persist FAISS index and chunks to disk."""
        idx_path = Path(config.FAISS_INDEX_PATH)
        faiss.write_index(
            self.index, str(idx_path / "index.faiss")
        )
        with open(idx_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print("💾 Index saved to disk")

    def load_index(self) -> None:
        """Load FAISS index from disk."""
        idx_path = Path(config.FAISS_INDEX_PATH)
        self.index = faiss.read_index(
            str(idx_path / "index.faiss")
        )
        with open(idx_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print("📂 Index loaded from disk")