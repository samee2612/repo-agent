"""Evaluation harness for repo-agent over sample_repo/."""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Allow running directly as `python eval/run_eval.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import run_agent
from app.config import CHROMA_PERSIST_DIR, DEFAULT_SAMPLE_REPO
from app.graph import build_dependency_graph
from app.indexer import extract_repo_chunks, get_chroma_collection, index_chunks_to_chroma
from app.llm import GeminiClient
from app.tools import RepoTools

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def _compute_recall_at_3(
    expected_files: list[str], search_hits_paths: list[str]
) -> float | None:
    """Compute recall@3 for expected files from search_code results."""
    if not expected_files:
        return None
    top3 = set(search_hits_paths[:3])
    matched = sum(1 for f in expected_files if f in top3)
    return matched / len(expected_files)


def _is_category_accurate(item: dict[str, Any], answer_obj: Any) -> bool:
    """Heuristic check for per-category answer accuracy."""
    category = item["category"]
    text_lower = answer_obj.answer.lower()
    if category == "negative":
        markers = ("insufficient", "not ", "no ", "cannot", "doesn't", "does not", "not found")
        return any(m in text_lower for m in markers) and answer_obj.grounded

    cited_files = {c.path for c in answer_obj.citations if c.verified}
    expected_files = set(item.get("expected_files", []))
    if not expected_files:
        return answer_obj.grounded
    return bool(cited_files & expected_files) and answer_obj.grounded


def main() -> None:
    """Index sample_repo, run golden.json questions, print metrics, and save results."""
    sleep_seconds = float(os.environ.get("EVAL_SLEEP_SECONDS", "4"))
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    _, chunks = extract_repo_chunks(DEFAULT_SAMPLE_REPO)
    index_chunks_to_chroma(chunks=chunks, persist_dir=CHROMA_PERSIST_DIR)
    collection = get_chroma_collection(persist_dir=CHROMA_PERSIST_DIR, reset=False)
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO, chunks=chunks)
    repo_tools = RepoTools(
        repo_root=DEFAULT_SAMPLE_REPO,
        graph=graph,
        collection=collection,
    )
    llm_client = GeminiClient()

    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    category_stats: dict[str, dict[str, int]] = {}

    total_citations = 0
    verified_citations = 0
    recall_scores: list[float] = []

    for idx, item in enumerate(golden):
        if idx > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        # Also run a direct search_code probe for recall@3 over expected files.
        search_res = repo_tools.search_code(item["question"], top_k=3)
        search_paths = [
            r["path"] for r in (search_res.get("data") or {}).get("results", [])
        ]
        recall_3 = _compute_recall_at_3(item.get("expected_files", []), search_paths)
        if recall_3 is not None:
            recall_scores.append(recall_3)

        answer = run_agent(
            question=item["question"],
            llm_client=llm_client,
            repo_tools=repo_tools,
        )

        status_counts[answer.status] = status_counts.get(answer.status, 0) + 1
        total_citations += len(answer.citations)
        verified_citations += sum(1 for c in answer.citations if c.verified)

        accurate = _is_category_accurate(item, answer)
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if accurate:
            category_stats[cat]["correct"] += 1

        records.append(
            {
                "id": item["id"],
                "category": cat,
                "question": item["question"],
                "status": answer.status,
                "grounded": answer.grounded,
                "recall_at_3": recall_3,
                "accurate": accurate,
                "citations": [c.model_dump() for c in answer.citations],
                "trace_steps": len(answer.trace),
                "answer": answer.answer,
            }
        )

    avg_recall_3 = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    citation_validity = (
        verified_citations / total_citations if total_citations > 0 else 1.0
    )
    per_category_accuracy = {
        cat: round(stats["correct"] / stats["total"], 4)
        for cat, stats in category_stats.items()
    }

    summary = {
        "total_questions": len(golden),
        "recall_at_3": round(avg_recall_3, 4),
        "citation_validity_rate": round(citation_validity, 4),
        "verified_citations": verified_citations,
        "total_citations": total_citations,
        "status_counts": status_counts,
        "per_category_accuracy": per_category_accuracy,
        "results": records,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'ID':<5} {'Category':<12} {'Status':<11} {'Grounded':<9} {'R@3':<6} {'Acc':<5}")
    print("-" * 55)
    for r in records:
        r3_str = f"{r['recall_at_3']:.2f}" if r["recall_at_3"] is not None else "n/a"
        print(
            f"{r['id']:<5} {r['category']:<12} {r['status']:<11} "
            f"{str(r['grounded']):<9} {r3_str:<6} {str(r['accurate']):<5}"
        )
    print("-" * 55)
    print(f"Recall@3: {summary['recall_at_3']:.4f}")
    print(f"Citation validity: {summary['citation_validity_rate']:.4f} ({verified_citations}/{total_citations})")
    print(f"Status counts: {status_counts}")
    print(f"Per-category accuracy: {per_category_accuracy}")
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
