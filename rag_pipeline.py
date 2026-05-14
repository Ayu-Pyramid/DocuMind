"""
RAG Pipeline — core ingestion and retrieval logic.
Supports PDF, TXT, and DOCX documents.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
Do not make up or infer information beyond what is explicitly stated.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Load & chunk documents
      2. Embed chunks with OpenAI embeddings
      3. Store in FAISS vector index
      4. Retrieve top-k chunks and generate answers with GPT-4
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
        index_path: str = "data/faiss_index",
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.index_path = index_path

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Document loading ─────────────────────────────────────────────────────

    def _load_file(self, path: str):
        ext = Path(path).suffix.lower()
        loaders = {
            ".pdf":  PyPDFLoader,
            ".txt":  TextLoader,
            ".docx": Docx2txtLoader,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders)}")
        logger.info(f"Loading {path}")
        return loaders[ext](path).load()

    def ingest(self, paths: list[str]) -> int:
        """Load, chunk, embed, and index documents. Returns chunk count."""
        all_docs = []
        for p in paths:
            docs = self._load_file(p)
            # Attach source metadata
            for d in docs:
                d.metadata["source"] = Path(p).name
            all_docs.extend(docs)

        chunks = self.splitter.split_documents(all_docs)
        logger.info(f"Split into {len(chunks)} chunks from {len(paths)} file(s)")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.index_path)
        logger.info(f"Index saved to {self.index_path}")

        self._build_chain()
        return len(chunks)

    def load_index(self):
        """Load a previously saved FAISS index."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"No index at {self.index_path}. Run ingest() first.")
        self.vectorstore = FAISS.load_local(
            self.index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self._build_chain()
        logger.info(f"Loaded index from {self.index_path}")

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",            # Maximal Marginal Relevance — reduces redundancy
            search_kwargs={"k": self.top_k, "fetch_k": self.top_k * 3},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Returns:
            answer       : str
            sources      : list of (filename, page)
            latency_ms   : float
            retrieved_chunks : list of str
        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not ready. Call ingest() or load_index() first.")

        t0 = time.perf_counter()
        result = self.qa_chain.invoke({"query": question})
        latency_ms = (time.perf_counter() - t0) * 1000

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page":   d.metadata.get("page", "N/A"),
            }
            for d in result.get("source_documents", [])
        ]

        return {
            "question":        question,
            "answer":          result["result"].strip(),
            "sources":         sources,
            "latency_ms":      round(latency_ms, 1),
            "retrieved_chunks": [d.page_content for d in result.get("source_documents", [])],
        }
