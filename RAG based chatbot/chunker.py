"""
Text Chunking Module
Implements recursive character splitting with overlap for context preservation.
"""

from typing import List
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: int
    source: str
    start_char: int = 0
    end_char: int = 0


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )

    def chunk_document(self, document) -> List[Chunk]:
        """Split a document into chunks with metadata."""
        raw_chunks = self.splitter.split_text(
            document.content
        )

        chunks = []
        offset = 0
        for i, text in enumerate(raw_chunks):
            start = document.content.find(text, offset)
            chunks.append(Chunk(
                text=text,
                chunk_id=i,
                source=document.metadata.get("source", ""),
                start_char=max(start, 0),
                end_char=max(start, 0) + len(text),
            ))
            if start >= 0:
                offset = start + len(text)

        print(f"✂️  Created {len(chunks)} chunks")
        return chunks