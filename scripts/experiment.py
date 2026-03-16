"""
scripts/experiment.py
---------------------
Experiment: k-sensitivity analysis — how does retrieval depth (k) affect
Precision@k across all three retrieval strategies?

This is a retrieval-only experiment — no OpenAI API calls are made.
Results show how many correct chunks each retriever finds as k increases.

Run (from project root):
    .venv/bin/python scripts/experiment.py catan
    .venv/bin/python scripts/experiment.py monopoly
    .venv/bin/python scripts/experiment.py all

Outputs:
  data/outputs/{game}_experiment_k_sensitivity.csv
"""

import csv
import json
import os
import sys
import random
from pathlib import Path

K_VALUES   = [1, 2, 3, 5, 7, 10]
RETRIEVERS = ["random", "tfidf", "dense"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Retrieval-only implementations (no OpenAI needed)
# ---------------------------------------------------------------------------

def _load_game(game: str) -> dict:
    """Load all retrieval artifacts for the given game."""
    chunks_path = f"models/{game}/chunks.json"
    index_path  = f"models/{game}/faiss.index"

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        print(f"ERROR: Run build_features.py {game} first.", file=sys.stderr)
        sys.exit(1)

    with open(chunks_path) as f:
        chunks = json.load(f)

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading dense model + FAISS index for '{game}'...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    index       = faiss.read_index(index_path)

    entry = {"chunks": chunks, "dense_model": dense_model, "index": index}

    tfidf_matrix_path     = f"models/{game}/tfidf_matrix.npz"
    tfidf_vectorizer_path = f"models/{game}/tfidf_vectorizer.pkl"
    if os.path.exists(tfidf_matrix_path) and os.path.exists(tfidf_vectorizer_path):
        import pickle
        import scipy.sparse
        entry["tfidf_matrix"]     = scipy.sparse.load_npz(tfidf_matrix_path)
        with open(tfidf_vectorizer_path, "rb") as f:
            entry["tfidf_vectorizer"] = pickle.load(f)
        print(f"  Loaded TF-IDF index for '{game}'.")
    else:
        print(f"  WARNING: TF-IDF index not found for '{game}' — skipping tfidf retriever.", file=sys.stderr)

    return entry


def _retrieve_indices(query: str, k: int, retriever: str, reg: dict) -> list[int]:
    """Return the chunk indices selected by the given retriever for this query."""
    chunks = reg["chunks"]

    if retriever == "random":
        rng = random.Random(RANDOM_SEED)
        return rng.sample(range(len(chunks)), min(k, len(chunks)))

    elif retriever == "tfidf":
        if "tfidf_vectorizer" not in reg:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = reg["tfidf_vectorizer"].transform([query])
        scores    = cosine_similarity(query_vec, reg["tfidf_matrix"])[0]
        return list(scores.argsort()[::-1][:k])

    else:  # dense
        import faiss
        query_emb = reg["dense_model"].encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        _, indices = reg["index"].search(query_emb, k)
        return [int(i) for i in indices[0] if i >= 0]


def _precision_at_k(indices: list[int], keywords: list[str], chunks: list[dict]) -> float:
    """1 if at least one keyword appears in any retrieved chunk title, else 0."""
    if not keywords:
        return 0.0
    retrieved_titles = [chunks[i]["title"].lower() for i in indices]
    for kw in keywords:
        if any(kw.lower() in t for t in retrieved_titles):
            return 1.0
    return 0.0


def _coverage(indices: list[int], keywords: list[str], chunks: list[dict]) -> float:
    """Fraction of required keywords found across retrieved chunks."""
    if not keywords:
        return 0.0
    retrieved_titles = [chunks[i]["title"].lower() for i in indices]
    hits = sum(1 for kw in keywords if any(kw.lower() in t for t in retrieved_titles))
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(game: str) -> list[dict]:
    eval_path = f"data/raw/{game}_eval.json"
    if not os.path.exists(eval_path):
        print(f"ERROR: {eval_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading artifacts for '{game}'...")
    reg = _load_game(game)

    with open(eval_path) as f:
        data = json.load(f)
    questions = data.get("correctness_questions", []) + data.get("stress_test_questions", [])

    print(f"\nRunning k-sensitivity experiment — {len(questions)} questions × {len(K_VALUES)} k-values × {len(RETRIEVERS)} retrievers")
    print("=" * 70)

    rows = []
    for retriever in RETRIEVERS:
        if retriever == "tfidf" and "tfidf_vectorizer" not in reg:
            print(f"  Skipping '{retriever}' — TF-IDF index not built.")
            continue

        for k in K_VALUES:
            precisions = []
            coverages  = []
            for q in questions:
                indices   = _retrieve_indices(q["question"], k, retriever, reg)
                keywords  = q.get("source_keywords", [])
                precisions.append(_precision_at_k(indices, keywords, reg["chunks"]))
                coverages.append(_coverage(indices, keywords, reg["chunks"]))

            avg_prec = sum(precisions) / len(precisions) * 100
            avg_cov  = sum(coverages)  / len(coverages)  * 100

            rows.append({
                "game":           game,
                "retriever":      retriever,
                "k":              k,
                "precision_at_k": round(avg_prec, 1),
                "coverage":       round(avg_cov, 1),
                "n_questions":    len(questions),
            })
            print(f"  [{retriever:<6}] k={k:2d}  Precision@k={avg_prec:5.1f}%  Coverage={avg_cov:5.1f}%")

    # Save CSV
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    out_path = f"data/outputs/{game}_experiment_k_sensitivity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → '{out_path}'")
    _print_experiment_summary(rows, game)
    return rows


def _print_experiment_summary(rows: list[dict], game: str):
    print(f"\n{'='*70}")
    print(f"K-SENSITIVITY SUMMARY — {game.upper()}")
    print(f"{'='*70}")
    print(f"\n{'Retriever':<10} {'k':>4}  {'Precision@k':>12}  {'Coverage':>10}")
    print("-" * 44)
    for row in rows:
        print(
            f"  {row['retriever']:<8} {row['k']:>4}  "
            f"{row['precision_at_k']:>10.1f}%  {row['coverage']:>8.1f}%"
        )
    print()

    # Highlight key finding
    dense_k3  = next((r for r in rows if r["retriever"] == "dense"  and r["k"] == 3), None)
    tfidf_k3  = next((r for r in rows if r["retriever"] == "tfidf"  and r["k"] == 3), None)
    random_k3 = next((r for r in rows if r["retriever"] == "random" and r["k"] == 3), None)

    print("Key comparison at k=3:")
    for label, row in [("Dense (DL)", dense_k3), ("TF-IDF (ML)", tfidf_k3), ("Random", random_k3)]:
        if row:
            print(f"  {label:<14} Precision@3={row['precision_at_k']:.1f}%  Coverage={row['coverage']:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="k-sensitivity experiment for retriever comparison.")
    parser.add_argument("game", help="Game to run experiment on, or 'all' for both")
    args = parser.parse_args()

    games = ["catan", "monopoly"] if args.game == "all" else [args.game]
    for game in games:
        run_experiment(game)
