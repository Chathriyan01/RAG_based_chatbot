"""
Retriever Module
Performs semantic search over the FAISS index
to find the most relevant document chunks.
"""

from typing import List, Tuple
from dataclasses import dataclass
from embeddings import EmbeddingEngine
from config import config


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    text: str
    score: float
    chunk_id: int
    source: str


class SemanticRetriever:
    """Retrieves relevant chunks using vector similarity."""

    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine
        self.top_k = config.TOP_K

    def retrieve(
        self, query: str
    ) -> List[RetrievalResult]:
        """Find top-k most relevant chunks for a query."""
        # Encode the query
        query_vec = self.engine.encode([query])

        # Search FAISS index
        distances, indices = self.engine.index.search(
            query_vec, self.top_k
        )

        # Build results with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.engine.chunks[idx]
            score = 1.0 / (1.0 + dist)  # Normalize
            results.append(RetrievalResult(
                text=chunk.text,
                score=round(score, 4),
                chunk_id=chunk.chunk_id,
                source=chunk.source,
            ))

        print(f"🔍 Retrieved {len(results)} chunks")
        return results

    def format_context(
        self, results: List[RetrievalResult]
    ) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} | Score: {r.score}]\n{r.text}"
            )
        return "\n\n---\n\n".join(context_parts)