"""
RAG Evaluation Layer
Scores answers on 4 dimensions without needing ground-truth labels:
  - Faithfulness     : is the answer grounded in retrieved chunks?
  - Relevance        : does the answer address the question?
  - Completeness     : how thoroughly is the question answered?
  - Hallucination    : does the answer contain claims absent from context?
Uses an LLM-as-judge approach (GPT-4o-mini scoring 1–5).
"""

import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

JUDGE_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template="""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the answer strictly based on the provided context.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Score each dimension 1–5 (5 = best). Return ONLY valid JSON, no explanation outside the JSON.

{{
  "faithfulness": <1-5>,        // Answer is grounded in context, no fabrication
  "relevance": <1-5>,           // Answer directly addresses the question  
  "completeness": <1-5>,        // Answer covers all aspects present in context
  "hallucination_risk": <1-5>,  // 5 = no hallucination, 1 = major fabrication
  "reasoning": "<one sentence explaining the scores>"
}}"""
)


@dataclass
class EvalResult:
    question: str
    answer: str
    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    hallucination_risk: float = 0.0
    overall_score: float = 0.0
    reasoning: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.results:
            return {}
        valid = [r for r in self.results if r.error is None]
        if not valid:
            return {"error": "All evaluations failed"}

        def avg(attr):
            return round(statistics.mean(getattr(r, attr) for r in valid), 2)

        return {
            "total_questions":    len(self.results),
            "successful_evals":   len(valid),
            "avg_faithfulness":   avg("faithfulness"),
            "avg_relevance":      avg("relevance"),
            "avg_completeness":   avg("completeness"),
            "avg_hallucination_risk": avg("hallucination_risk"),
            "avg_overall_score":  avg("overall_score"),
            "avg_latency_ms":     avg("latency_ms"),
            "grade":              self._grade(avg("overall_score")),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 4.5: return "A  (Excellent)"
        if score >= 4.0: return "B  (Good)"
        if score >= 3.0: return "C  (Acceptable)"
        if score >= 2.0: return "D  (Needs improvement)"
        return "F  (Poor — check chunking & retrieval)"

    def print_report(self):
        s = self.summary()
        print("\n" + "="*60)
        print("  RAG EVALUATION REPORT")
        print("="*60)
        print(f"  Questions evaluated : {s.get('total_questions', 0)}")
        print(f"  Faithfulness        : {s.get('avg_faithfulness', 0):.2f} / 5.0")
        print(f"  Relevance           : {s.get('avg_relevance', 0):.2f} / 5.0")
        print(f"  Completeness        : {s.get('avg_completeness', 0):.2f} / 5.0")
        print(f"  Hallucination Risk  : {s.get('avg_hallucination_risk', 0):.2f} / 5.0")
        print(f"  Overall Score       : {s.get('avg_overall_score', 0):.2f} / 5.0")
        print(f"  Grade               : {s.get('grade', 'N/A')}")
        print(f"  Avg Latency         : {s.get('avg_latency_ms', 0):.0f} ms")
        print("="*60 + "\n")


class RAGEvaluator:
    """LLM-as-judge evaluator for RAG pipeline quality."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)

    def evaluate_single(self, question: str, answer: str, retrieved_chunks: list[str]) -> EvalResult:
        context = "\n\n---\n\n".join(retrieved_chunks[:3])  # top 3 chunks
        t0 = time.perf_counter()
        try:
            prompt = JUDGE_PROMPT.format(
                question=question, answer=answer, context=context
            )
            response = self.llm.invoke(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000

            raw = response.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)

            overall = statistics.mean([
                scores["faithfulness"],
                scores["relevance"],
                scores["completeness"],
                scores["hallucination_risk"],
            ])

            return EvalResult(
                question=question,
                answer=answer,
                faithfulness=scores["faithfulness"],
                relevance=scores["relevance"],
                completeness=scores["completeness"],
                hallucination_risk=scores["hallucination_risk"],
                overall_score=round(overall, 2),
                reasoning=scores.get("reasoning", ""),
                latency_ms=round(latency_ms, 1),
            )

        except Exception as e:
            return EvalResult(
                question=question,
                answer=answer,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                error=str(e),
            )

    def evaluate_batch(self, qa_results: list[dict]) -> EvalReport:
        """
        Args:
            qa_results: list of dicts from RAGPipeline.query()
        Returns:
            EvalReport with per-question scores and aggregate summary
        """
        report = EvalReport()
        for i, r in enumerate(qa_results, 1):
            print(f"  Evaluating {i}/{len(qa_results)}: {r['question'][:60]}...")
            result = self.evaluate_single(
                question=r["question"],
                answer=r["answer"],
                retrieved_chunks=r.get("retrieved_chunks", []),
            )
            report.results.append(result)
        return report
