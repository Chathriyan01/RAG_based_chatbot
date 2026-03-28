"""
QA Chain Module
Orchestrates the full RAG pipeline: retrieval → prompt → generation.
Updated for latest LangChain (0.1+) – no LLMChain.
"""

from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from retriever import SemanticRetriever
from config import config


QA_PROMPT = """You are a helpful document assistant.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find
this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class QAResponse:
    """Structured response from the QA system."""
    answer: str
    sources: list
    confidence: float


class QAChain:
    """End-to-end question answering chain."""

    def __init__(self, retriever: SemanticRetriever):
        self.retriever = retriever
        self.llm = self._init_llm()

        # Updated LangChain prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=QA_PROMPT
        )

        # LCEL: prompt → llm
        self.chain = self.prompt | self.llm

    def _init_llm(self):
        """Initialize the LLM with HuggingFace pipeline."""
        print("🤖 Loading LLM...")
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            repetition_penalty=config.REPETITION_PENALTY,
        )

        return HuggingFacePipeline(pipeline=pipe)

    def answer(self, question: str) -> QAResponse:
        """Full RAG pipeline: retrieve → generate."""
        # Step 1: Retrieve documents
        results = self.retriever.retrieve(question)
        context = self.retriever.format_context(results)

        # Step 2: Generate answer using LCEL invoke()
        raw_output = self.chain.invoke({
            "context": context,
            "question": question
        })

        answer = raw_output.strip()

        # Step 3: Confidence score
        avg_score = sum(r.score for r in results) / max(len(results), 1)

        return QAResponse(
            answer=answer,
            sources=[r.source for r in results],
            confidence=round(avg_score, 4)
        )