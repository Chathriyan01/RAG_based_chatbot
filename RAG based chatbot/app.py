"""
Streamlit Application — Main Entry Point
Run: streamlit run app.py
"""

import streamlit as st
import tempfile
from pathlib import Path
from document_loader import DocumentLoader
from chunker import TextChunker
from embeddings import EmbeddingEngine
from retriever import SemanticRetriever
from qa_chain import QAChain


# ── Page Config ────────────────────────────────
st.set_page_config(
    page_title="📄 Document Q&A System",
    page_icon="🧠",
    layout="wide"
)
st.title("Youtubr Chatbot")


# ── Initialize Components (cached) ─────────────
@st.cache_resource
def init_pipeline():
    loader = DocumentLoader()
    chunker = TextChunker()
    engine = EmbeddingEngine()
    retriever = SemanticRetriever(engine)
    qa = QAChain(retriever)
    return loader, chunker, engine, retriever, qa

loader, chunker, engine, retriever, qa = init_pipeline()


# ── Sidebar: Document Upload ───────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if files and st.button("🚀 Process Documents"):
        all_chunks = []
        progress = st.progress(0)

        for i, file in enumerate(files):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file.name).suffix
            ) as tmp:
                tmp.write(file.read())
                doc = loader.load(tmp.name)
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            progress.progress((i + 1) / len(files))

        # Build vector index
        engine.build_index(all_chunks)
        engine.save_index()
        st.success(
            f"✅ Processed {len(files)} files → {len(all_chunks)} chunks"
        )


# ── Main Chat Interface ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            response = qa.answer(question)

        st.markdown(response.answer)
        st.caption(
            f"Confidence: {response.confidence:.1%}"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer}
    )