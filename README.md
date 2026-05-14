Here it is — copy everything below and paste it into the editor, replacing everything:

```markdown
# 🔍 DocuMind — RAG Document Q&A System

> **Ask questions about any document. Get cited, grounded answers — scored for hallucination risk.**

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline with an integrated **LLM-as-judge evaluation layer**. Upload PDFs, text files, or Word documents and query them through a conversational interface backed by OpenAI embeddings, FAISS vector search, and GPT-4.

Built by **Ayush Jariwala** — [LinkedIn](https://linkedin.com/in/ayush-jariwala-3015132b3) · [GitHub](https://github.com/Ayu-Pyramid)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | PDF, TXT, DOCX — chunked with configurable overlap |
| **Semantic search** | OpenAI `text-embedding-3-small` + FAISS with MMR retrieval |
| **Grounded generation** | GPT-4o-mini with source citations (file + page number) |
| **Evaluation layer** | LLM-as-judge scoring: Faithfulness · Relevance · Completeness · Hallucination Risk |
| **Interactive UI** | Streamlit app with chat history, source viewer, and eval dashboard |
| **CLI interface** | `run.py ingest / query / eval` for scripting and automation |
| **Export** | Download evaluation results as JSON |

---

## 🏗️ Architecture

```
Documents (PDF/TXT/DOCX)
        │
        ▼
  [ Document Loader ]
  PyPDF / TextLoader / Docx2txt
        │
        ▼
  [ Text Splitter ]
  RecursiveCharacterTextSplitter
  chunk_size=800, overlap=150
        │
        ▼
  [ Embeddings ]
  OpenAI text-embedding-3-small
        │
        ▼
  [ Vector Store ]
  FAISS (saved to disk)
        │
   Query ──▶ [ MMR Retrieval ] ──▶ top-k chunks
                                        │
                                        ▼
                               [ GPT-4 Generation ]
                               with source citations
                                        │
                                        ▼
                              [ Evaluation Layer ]
                              LLM-as-judge (GPT-4o-mini)
                              Faithfulness / Relevance /
                              Completeness / Hallucination
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Ayu-Pyramid/DocuMind.git
cd DocuMind
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload a document, ask questions, run evaluation.

### 4. Or use the CLI

```bash
# Ingest documents
python run.py ingest paper.pdf report.docx

# Ask a question (with evaluation)
python run.py query "What are the main findings?" --eval

# Batch evaluate from a question file
python run.py eval sample_questions.txt --output results.json
```

---

## 📊 Evaluation System

The evaluation layer uses **GPT-4o-mini as a judge** — it reads the retrieved context and generated answer, then scores 4 dimensions without needing human-labeled ground truth:

| Dimension | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Relevance** | Does the answer directly address the question? |
| **Completeness** | Are all relevant aspects in the context covered? |
| **Hallucination Risk** | Does the answer contain claims absent from context? |

Each dimension is scored **1–5**. The system outputs an overall grade (A–F) and per-question reasoning.

```
============================================================
  RAG EVALUATION REPORT
============================================================
  Questions evaluated : 5
  Faithfulness        : 4.60 / 5.0
  Relevance           : 4.80 / 5.0
  Completeness        : 4.20 / 5.0
  Hallucination Risk  : 4.80 / 5.0
  Overall Score       : 4.60 / 5.0
  Grade               : A  (Excellent)
  Avg Latency         : 1823 ms
============================================================
```

---

## 🛠️ Tech Stack

- **LangChain** — orchestration, prompt templates, retrieval chains
- **OpenAI API** — embeddings (`text-embedding-3-small`) + generation (`gpt-4o-mini`)
- **FAISS** — local vector store with MMR (Maximal Marginal Relevance) retrieval
- **Streamlit** — interactive web UI
- **Python 3.11+**

---

## 📁 Project Structure

```
DocuMind/
├── app.py                  # Streamlit UI (ingest + query + eval tabs)
├── run.py                  # CLI interface
├── evaluator.py            # LLM-as-judge evaluation with EvalReport
├── rag_pipeline.py         # Core RAG: load → chunk → embed → retrieve → generate
├── sample_questions.txt    # Sample questions for batch evaluation
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

All tunable via the Streamlit sidebar or constructor args:

| Parameter | Default | Effect |
|---|---|---|
| `model_name` | `gpt-4o-mini` | LLM for generation |
| `embedding_model` | `text-embedding-3-small` | Embedding model |
| `chunk_size` | `800` | Tokens per chunk |
| `chunk_overlap` | `150` | Overlap between chunks |
| `top_k` | `4` | Chunks retrieved per query |

---

## 💡 Design Decisions

**Why MMR retrieval?** Standard top-k retrieval often returns semantically similar (redundant) chunks. Maximal Marginal Relevance balances relevance with diversity — returning chunks that cover different aspects of the answer.

**Why LLM-as-judge?** Traditional RAG evaluation requires labeled Q&A datasets. LLM-as-judge enables evaluation on *any* document without manual annotation, which is critical for a general-purpose system.

**Why FAISS over a hosted vector DB?** For a portable, demo-able project, local FAISS requires zero infrastructure. In production, this would swap to Pinecone or Weaviate with minimal code changes.

---

## 📄 License

MIT — free to use, modify, and build on.
```

Paste that in, commit, done.
