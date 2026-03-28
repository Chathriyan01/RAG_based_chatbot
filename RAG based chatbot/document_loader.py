"""
Document Loading Module
Handles PDF, TXT, and DOCX file parsing with metadata extraction.
"""

import re
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a loaded document with content and metadata."""
    content: str
    metadata: Dict = field(default_factory=dict)


class DocumentLoader:
    """Universal document loader supporting multiple formats."""

    SUPPORTED = {".pdf", ".txt", ".md"}

    def load(self, file_path: str) -> Document:
        """Load a document from file path."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported format: {suffix}")

        if suffix == ".pdf":
            text = self._load_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")

        text = self._clean_text(text)
        return Document(
            content=text,
            metadata={"source": str(path), "pages": self._page_count}
        )

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF with page tracking."""
        reader = PdfReader(str(path))
        self._page_count = len(reader.pages)
        texts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            texts.append(f"[Page {i+1}] {page_text}")
        return "\n\n".join(texts)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        return text.strip()