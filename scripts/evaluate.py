"""
scripts/evaluate.py
--------------------
Run evaluation sets against the RAG pipeline and produce results.

    .venv/bin/python scripts/evaluate.py catan
    .venv/bin/python scripts/evaluate.py catan --retriever tfidf
    .venv/bin/python scripts/evaluate.py monopoly --retriever random

Outputs:
  data/outputs/{game}_results.csv           (dense — default)
  data/outputs/{game}_tfidf_results.csv     (tfidf)
  data/outputs/{game}_random_results.csv    (random)

After running, fill in the 'correctness_score' column:
  1 = fully correct,  0.5 = partially correct,  0 = wrong
"""

import csv
import json
import os
import sys
from pathlib import Path

from model import query_rag, DEFAULT_K, DEFAULT_RETRIEVER, RETRIEVERS

K = DEFAULT_K

ABSTAIN_PHRASES = [
    "don't specify", "doesn't specify", "not specify",
    "rules don't", "rules do not", "not addressed",
    "not mentioned", "not explicit", "not state",
    "silent", "i don't know", "cannot answer",
    "unclear", "no rule", "not covered",
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def check_retrieval(retrieved: list[dict], source_keywords: list[str]) -> dict:
    """Keyword-based retrieval quality check against chunk titles."""
    retrieved_titles = [c["title"].lower() for c in retrieved]
    hits = sum(
        1 for kw in source_keywords
        if any(kw.lower() in t for t in retrieved_titles)
    )
    coverage = hits / len(source_keywords) if source_keywords else 0.0
    return {
        "hits":           hits,
        "expected":       len(source_keywords),
        "coverage":       round(coverage, 3),
        "precision_at_k": 1.0 if hits > 0 else 0.0,
    }


def check_hallucination(answer: str, sub_category: str) -> str:
    """For Category C questions, detect abstention vs. potential hallucination."""
    if sub_category != "C":
        return "N/A"
    abstained = any(p in answer.lower() for p in ABSTAIN_PHRASES)
    return "ABSTAINED" if abstained else "POTENTIAL_HALLUCINATION"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(game: str, retriever: str = DEFAULT_RETRIEVER, k: int = K) -> list[dict]:
    eval_path = f"data/raw/{game}_eval.json"

    # Output filename: {game}_results.csv for dense, {game}_{retriever}_results.csv for others
    if retriever == "dense":
        results_path = f"data/outputs/{game}_results.csv"
    else:
        results_path = f"data/outputs/{game}_{retriever}_results.csv"

    if not os.path.exists(eval_path):
        print(f"ERROR: Eval data not found at '{eval_path}'", file=sys.stderr)
        sys.exit(1)

    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    with open(eval_path) as f:
        data = json.load(f)

    all_questions = (
        data.get("correctness_questions", [])
        + data.get("stress_test_questions", [])
    )

    results = []
    total   = len(all_questions)

    print(f"Running evaluation — game='{game}', retriever='{retriever}', k={k}, n={total}")
    print("=" * 70)

    for i, q in enumerate(all_questions, 1):
        qid      = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        keywords = q.get("source_keywords", [])
        category = q["category"]
        sub_cat  = q.get("sub_category", "")

        print(f"\n[{i}/{total}] {qid} ({category}/{sub_cat or '-'})")
        print(f"  Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        rag_result = query_rag(question, game=game, k=k, retriever=retriever)
        answer     = rag_result["answer"]
        retrieved  = rag_result["retrieved"]

        ret    = check_retrieval(retrieved, keywords)
        halluc = check_hallucination(answer, sub_cat)

        print(f"  Retrieved: {[c['title'] for c in retrieved]}")
        print(f"  Retrieval hits: {ret['hits']}/{ret['expected']}")
        print(f"  Expected: {expected[:100]}{'...' if len(expected) > 100 else ''}")
        print(f"  Model:    {answer[:100]}{'...' if len(answer) > 100 else ''}")
        if halluc != "N/A":
            print(f"  Hallucination check: {halluc}")

        results.append({
            "id":                  qid,
            "category":            category,
            "sub_category":        sub_cat,
            "retriever":           retriever,
            "question":            question,
            "expected_answer":     expected,
            "model_answer":        answer,
            "retrieved_chunks":    " | ".join(c["title"] for c in retrieved),
            "retrieval_hits":      ret["hits"],
            "retrieval_expected":  ret["expected"],
            "retrieval_coverage":  ret["coverage"],
            "precision_at_k":      ret["precision_at_k"],
            "hallucination_check": halluc,
            "correctness_score":   "",
            "notes":               "",
        })

    fieldnames = list(results[0].keys())
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\nResults saved to '{results_path}'")
    _print_summary(results, k, retriever)
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], k: int = K, retriever: str = DEFAULT_RETRIEVER):
    def group(cat, sub=None):
        return [r for r in results if r["category"] == cat and (sub is None or r["sub_category"] == sub)]

    correctness = group("correctness")
    stress_a    = group("stress", "A")
    stress_b    = group("stress", "B")
    stress_c    = group("stress", "C")

    def avg_prec(g):
        return (sum(r["precision_at_k"] for r in g) / len(g) * 100) if g else 0

    def avg_cov(g):
        return (sum(r["retrieval_coverage"] for r in g) / len(g) * 100) if g else 0

    halluc_rate = (
        sum(1 for r in stress_c if r["hallucination_check"] == "POTENTIAL_HALLUCINATION")
        / len(stress_c) * 100
        if stress_c else 0
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  (retriever={retriever}, k={k})")
    print("=" * 60)

    print(f"\nRetrieval Precision@{k}  (≥1 expected chunk retrieved):")
    print(f"  Correctness Qs       ({len(correctness):2d}): {avg_prec(correctness):.0f}%")
    print(f"  Stress A — multi-sec ({len(stress_a):2d}): {avg_prec(stress_a):.0f}%")
    print(f"  Stress B — edge case ({len(stress_b):2d}): {avg_prec(stress_b):.0f}%")
    print(f"  Stress C — unanswer  ({len(stress_c):2d}): {avg_prec(stress_c):.0f}%")
    print(f"  Overall              ({len(results):2d}): {avg_prec(results):.0f}%")

    print(f"\nRetrieval Coverage  (fraction of required chunks found):")
    print(f"  Stress A — multi-sec ({len(stress_a):2d}): {avg_cov(stress_a):.0f}%")
    print(f"  Overall              ({len(results):2d}): {avg_cov(results):.0f}%")

    print(f"\nHallucination Rate  (Category C — gave definitive answer when rules are silent):")
    abstained = sum(1 for r in stress_c if r["hallucination_check"] == "ABSTAINED")
    print(f"  Appropriately abstained: {abstained}/{len(stress_c)}")
    print(f"  Potential hallucinations: {len(stress_c) - abstained}/{len(stress_c)}  ({halluc_rate:.0f}%)")

    print(f"\n{'─'*60}")
    print(f"  Fill in 'correctness_score' in the results CSV:")
    print(f"    1 = correct,  0.5 = partially correct,  0 = wrong")
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline for a board game.")
    parser.add_argument("game", help="Game to evaluate (e.g. catan, monopoly)")
    parser.add_argument(
        "--retriever", choices=RETRIEVERS, default=DEFAULT_RETRIEVER,
        help=f"Retrieval strategy (default: {DEFAULT_RETRIEVER})",
    )
    args = parser.parse_args()

    if not os.path.exists(f"models/{args.game}/chunks.json"):
        print(f"ERROR: Index not found for '{args.game}'. Run: .venv/bin/python scripts/build_features.py {args.game}")
        sys.exit(1)

    run_evaluation(args.game, retriever=args.retriever)
