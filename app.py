"""
Streamlit UI for the RAG Document Q&A System.
Run: streamlit run app.py
"""

import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.source-chip {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy import pipeline (avoids import errors if deps missing) ───────────────
@st.cache_resource
def get_pipeline(model, chunk_size, chunk_overlap, top_k):
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline(
        model_name=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""),
                             help="Your key stays local — never sent anywhere except OpenAI.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("**Model Settings**")
    model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    chunk_size = st.slider("Chunk size (tokens)", 400, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, 25)
    top_k = st.slider("Top-K chunks to retrieve", 2, 8, 4)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
RAG pipeline using:
- 🔗 LangChain orchestration
- 🧠 OpenAI embeddings
- 📦 FAISS vector store
- ⚖️ LLM-as-judge evaluation
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Q&A System")
st.markdown("*Upload documents → Ask questions → Get cited, grounded answers*")

tab1, tab2, tab3 = st.tabs(["📄 Ingest Documents", "💬 Ask Questions", "📊 Evaluate Quality"])


# ── TAB 1: Ingest ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload Documents")
    st.markdown("Supports **PDF**, **TXT**, and **DOCX** files.")

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{len(uploaded)} file(s) ready**")
            for f in uploaded:
                size_kb = round(len(f.getvalue()) / 1024, 1)
                st.markdown(f"- `{f.name}` ({size_kb} KB)")

        with col2:
            if st.button("🚀 Build Index", type="primary", use_container_width=True):
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("Enter your OpenAI API key in the sidebar first.")
                else:
                    with st.spinner("Chunking, embedding, and indexing..."):
                        pipeline = get_pipeline(model, chunk_size, chunk_overlap, top_k)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            paths = []
                            for f in uploaded:
                                p = Path(tmpdir) / f.name
                                p.write_bytes(f.getvalue())
                                paths.append(str(p))
                            n_chunks = pipeline.ingest(paths)
                            st.session_state["pipeline"] = pipeline
                            st.session_state["ingested_files"] = [f.name for f in uploaded]

                    st.success(f"✅ Indexed {n_chunks} chunks from {len(uploaded)} file(s). Ready to query!")
                    st.balloons()


# ── TAB 2: Query ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ask Your Documents")

    if "pipeline" not in st.session_state:
        st.info("⬆️ Upload and index documents in the **Ingest** tab first.")
    else:
        files = st.session_state.get("ingested_files", [])
        st.markdown(f"**Indexed:** {', '.join(f'`{f}`' for f in files)}")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Chat history display
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander(f"📎 Sources ({len(msg['sources'])} chunks retrieved)"):
                        for s in msg["sources"]:
                            st.markdown(f'<span class="source-chip">📄 {s["source"]} — page {s["page"]}</span>', unsafe_allow_html=True)
                        if "latency" in msg:
                            st.caption(f"⚡ Response time: {msg['latency']} ms")

        question = st.chat_input("Ask anything about your documents...")
        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    result = st.session_state["pipeline"].query(question)
                answer = result["answer"]
                st.markdown(answer)
                with st.expander(f"📎 Sources ({len(result['sources'])} chunks retrieved)"):
                    for s in result["sources"]:
                        st.markdown(f'<span class="source-chip">📄 {s["source"]} — page {s["page"]}</span>', unsafe_allow_html=True)
                    st.caption(f"⚡ Response time: {result['latency_ms']} ms")

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "sources": result["sources"],
                "latency": result["latency_ms"],
            })
            # Save for eval
            if "qa_log" not in st.session_state:
                st.session_state["qa_log"] = []
            st.session_state["qa_log"].append(result)


# ── TAB 3: Evaluate ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Evaluation Dashboard")
    st.markdown("Score your RAG pipeline on **Faithfulness**, **Relevance**, **Completeness**, and **Hallucination Risk** using an LLM-as-judge approach.")

    qa_log = st.session_state.get("qa_log", [])

    if not qa_log:
        st.info("Ask at least one question in the **Ask Questions** tab to enable evaluation.")
    else:
        st.markdown(f"**{len(qa_log)} question(s) ready to evaluate.**")

        if st.button("🧪 Run Evaluation", type="primary"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("API key required.")
            else:
                from src.evaluator import RAGEvaluator
                evaluator = RAGEvaluator()

                with st.spinner(f"Evaluating {len(qa_log)} response(s)..."):
                    report = evaluator.evaluate_batch(qa_log)
                    summary = report.summary()
                    st.session_state["eval_report"] = report
                    st.session_state["eval_summary"] = summary

        if "eval_summary" in st.session_state:
            s = st.session_state["eval_summary"]
            report = st.session_state["eval_report"]

            st.markdown("---")
            st.markdown("#### Aggregate Scores")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Faithfulness",     f"{s['avg_faithfulness']}/5")
            c2.metric("Relevance",        f"{s['avg_relevance']}/5")
            c3.metric("Completeness",     f"{s['avg_completeness']}/5")
            c4.metric("Hallucination ✓",  f"{s['avg_hallucination_risk']}/5")
            c5.metric("Overall",          f"{s['avg_overall_score']}/5")

            grade_color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "🔴"}
            grade_letter = s["grade"][0]
            st.markdown(f"**Grade: {grade_color.get(grade_letter,'⚪')} {s['grade']}**")
            st.caption(f"Avg latency: {s['avg_latency_ms']:.0f} ms")

            st.markdown("#### Per-Question Breakdown")
            for r in report.results:
                with st.expander(f"Q: {r.question[:80]}..."):
                    if r.error:
                        st.error(f"Eval failed: {r.error}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Answer:** {r.answer[:300]}...")
                            st.markdown(f"*{r.reasoning}*")
                        with col2:
                            st.metric("Faithfulness", r.faithfulness)
                            st.metric("Relevance",    r.relevance)
                            st.metric("Hallucination", r.hallucination_risk)

            # Export
            st.markdown("---")
            export_data = [r.to_dict() for r in report.results]
            st.download_button(
                "⬇️ Export Evaluation JSON",
                data=json.dumps(export_data, indent=2),
                file_name="rag_eval_results.json",
                mime="application/json",
            )
