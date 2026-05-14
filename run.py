"""
CLI runner — use this to test your RAG pipeline without the UI.

Usage:
  python run.py ingest path/to/file.pdf
  python run.py query "What is the main conclusion of the paper?"
  python run.py eval path/to/questions.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator


def cmd_ingest(args):
    pipeline = RAGPipeline()
    n = pipeline.ingest(args.files)
    print(f"\n✅ Indexed {n} chunks from {len(args.files)} file(s).\n")


def cmd_query(args):
    pipeline = RAGPipeline()
    pipeline.load_index()
    result = pipeline.query(args.question)

    print(f"\n{'='*60}")
    print(f"Q: {result['question']}")
    print(f"{'='*60}")
    print(f"A: {result['answer']}")
    print(f"\n📎 Sources:")
    for s in result["sources"]:
        print(f"   - {s['source']} (page {s['page']})")
    print(f"\n⚡ Latency: {result['latency_ms']} ms\n")

    if args.eval:
        print("Running evaluation...")
        evaluator = RAGEvaluator()
        r = evaluator.evaluate_single(result["question"], result["answer"], result["retrieved_chunks"])
        print(f"  Faithfulness:      {r.faithfulness}/5")
        print(f"  Relevance:         {r.relevance}/5")
        print(f"  Completeness:      {r.completeness}/5")
        print(f"  Hallucination Risk:{r.hallucination_risk}/5")
        print(f"  Overall:           {r.overall_score}/5")
        print(f"  Reasoning: {r.reasoning}\n")


def cmd_eval(args):
    """Batch evaluate from a text file of questions (one per line)."""
    pipeline = RAGPipeline()
    pipeline.load_index()
    evaluator = RAGEvaluator()

    questions = Path(args.questions_file).read_text().strip().splitlines()
    questions = [q.strip() for q in questions if q.strip()]
    print(f"\nEvaluating {len(questions)} questions...\n")

    qa_results = [pipeline.query(q) for q in questions]
    report = evaluator.evaluate_batch(qa_results)
    report.print_report()

    if args.output:
        out = [r.to_dict() for r in report.results]
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="RAG Document Q&A CLI")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest documents into FAISS index")
    p_ingest.add_argument("files", nargs="+", help="PDF/TXT/DOCX file paths")

    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", type=str)
    p_query.add_argument("--eval", action="store_true", help="Also run evaluation on the answer")

    p_eval = sub.add_parser("eval", help="Batch evaluate from a questions file")
    p_eval.add_argument("questions_file", help="Text file with one question per line")
    p_eval.add_argument("--output", help="Save JSON results to this path")

    args = parser.parse_args()
    if args.command == "ingest": cmd_ingest(args)
    elif args.command == "query": cmd_query(args)
    elif args.command == "eval":  cmd_eval(args)
    else: parser.print_help()
