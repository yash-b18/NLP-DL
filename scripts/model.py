"""
scripts/model.py
----------------
Retrieval-Augmented Generation pipeline supporting three retrieval strategies:

  dense  — sentence-transformers embeddings + FAISS  (deep learning)
  tfidf  — TF-IDF bag-of-words + cosine similarity   (classical ML)
  random — random chunk selection                     (naive baseline)

Usage (from project root):
    .venv/bin/python scripts/model.py catan 'What resources do you need to build a settlement?'
    .venv/bin/python scripts/model.py monopoly 'How much money does each player start with?' --retriever tfidf
    .venv/bin/python scripts/model.py catan --retriever random

Requires scripts/build_features.py to have been run first for the given game.
Requires OPENAI_API_KEY to be set in the environment.
"""

import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_K         = 3
RERANK_CANDIDATES = 10
RERANK_TOP_N      = 3
DEFAULT_RETRIEVER = "dense"
RETRIEVERS        = ["dense", "tfidf", "random"]
OPENAI_MODEL      = "gpt-4o-mini"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Per-game singleton registry
# ---------------------------------------------------------------------------

_registry: dict[str, dict] = {}   # game -> loaded artifacts


def _make_system_prompt(game: str) -> str:
    game_name = game.title()
    return (
        f"You are an expert on the board game {game_name}. "
        "Answer the user's question using ONLY the rulebook sections provided as context.\n\n"
        "Rules:\n"
        "1. Base your answer SOLELY on the provided context. Do not use outside knowledge.\n"
        "2. If the answer can be inferred or directly found in the context, state it clearly "
        "and cite the source section, e.g. \"According to [Almanac: ROBBER]...\".\n"
        "3. Only say \"The rules don't specify this.\" if NO relevant information appears in ANY "
        "of the retrieved sections. Do not abstain if the answer is present or can be reasonably "
        "inferred from the context.\n"
        "4. Do not invent or extrapolate rules beyond what the text says.\n"
        "5. Be concise and precise."
    )


def _load(game: str):
    """Lazy-load and cache all retrieval artifacts for the given game."""
    if game in _registry:
        return

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: .venv/bin/pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    chunks_path = f"models/{game}/chunks.json"
    index_path  = f"models/{game}/faiss.index"

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        print(
            f"Index files not found for '{game}'. "
            f"Run: .venv/bin/python scripts/build_features.py {game}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model (all-MiniLM-L6-v2)...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loading FAISS index for '{game}'...")
    index = faiss.read_index(index_path)

    with open(chunks_path) as f:
        chunks = json.load(f)

    entry: dict = {
        "dense_model":   dense_model,
        "index":         index,
        "chunks":        chunks,
        "client":        OpenAI(),
        "system_prompt": _make_system_prompt(game),
    }

    # TF-IDF artifacts (classical ML) — optional, built by build_features.py
    tfidf_matrix_path     = f"models/{game}/tfidf_matrix.npz"
    tfidf_vectorizer_path = f"models/{game}/tfidf_vectorizer.pkl"
    if os.path.exists(tfidf_matrix_path) and os.path.exists(tfidf_vectorizer_path):
        import pickle
        import scipy.sparse
        entry["tfidf_matrix"]     = scipy.sparse.load_npz(tfidf_matrix_path)
        with open(tfidf_vectorizer_path, "rb") as f:
            entry["tfidf_vectorizer"] = pickle.load(f)
        print(f"Loaded TF-IDF index for '{game}'.")

    # Cross-encoder for reranking (loaded once, shared across games)
    if "_cross_encoder" not in _registry:
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder model ({CROSS_ENCODER_MODEL})...")
        _registry["_cross_encoder"] = CrossEncoder(CROSS_ENCODER_MODEL)

    _registry[game] = entry
    print(f"Ready — {len(chunks)} chunks loaded for '{game}'.\n")


# ---------------------------------------------------------------------------
# Retrieval — three strategies
# ---------------------------------------------------------------------------

def _dense_retrieve(query: str, game: str, k: int) -> list[dict]:
    """Deep learning: sentence-transformer embeddings + FAISS cosine search."""
    import faiss
    reg = _registry[game]

    query_emb = reg["dense_model"].encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = reg["index"].search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = reg["chunks"][idx].copy()
            chunk["retrieval_score"] = float(score)
            chunk["chunk_idx"]       = int(idx)
            results.append(chunk)
    return results


def _tfidf_retrieve(query: str, game: str, k: int) -> list[dict]:
    """Classical ML: TF-IDF bag-of-words + cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    reg = _registry[game]

    if "tfidf_vectorizer" not in reg:
        print(
            f"TF-IDF index not found for '{game}'. "
            f"Re-run: .venv/bin/python scripts/build_features.py {game}",
            file=sys.stderr,
        )
        return []

    query_vec = reg["tfidf_vectorizer"].transform([query])
    scores    = cosine_similarity(query_vec, reg["tfidf_matrix"])[0]
    top_idx   = scores.argsort()[::-1][:k]

    results = []
    for idx in top_idx:
        chunk = reg["chunks"][idx].copy()
        chunk["retrieval_score"] = float(scores[idx])
        chunk["chunk_idx"]       = int(idx)
        results.append(chunk)
    return results


def _random_retrieve(query: str, game: str, k: int) -> list[dict]:
    """Naive baseline: randomly select k chunks (no query signal)."""
    import random
    reg     = _registry[game]
    chunks  = reg["chunks"]
    indices = random.sample(range(len(chunks)), min(k, len(chunks)))

    results = []
    for idx in indices:
        chunk = chunks[idx].copy()
        chunk["retrieval_score"] = 0.0
        chunk["chunk_idx"]       = int(idx)
        results.append(chunk)
    return results


def _rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """Re-score candidate chunks with a cross-encoder and return the top-n."""
    cross_encoder = _registry["_cross_encoder"]
    pairs = [[query, c["text"]] for c in candidates]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]


def retrieve(
    query:     str,
    game:      str,
    k:         int = DEFAULT_K,
    retriever: str = DEFAULT_RETRIEVER,
    rerank:    bool = False,
) -> list[dict]:
    """Return the top-k most relevant chunks using the specified retrieval strategy.

    If rerank=True, first retrieve RERANK_CANDIDATES chunks, then re-score them
    with a cross-encoder and return the top RERANK_TOP_N.
    """
    _load(game)
    fetch_k = RERANK_CANDIDATES if (rerank and retriever != "random") else k

    if retriever == "tfidf":
        results = _tfidf_retrieve(query, game, fetch_k)
    elif retriever == "random":
        results = _random_retrieve(query, game, k)
    else:
        results = _dense_retrieve(query, game, fetch_k)

    if rerank and retriever != "random":
        results = _rerank(query, results, top_n=RERANK_TOP_N)

    return results


# ---------------------------------------------------------------------------
# Generation (shared across all retrievers)
# ---------------------------------------------------------------------------

def generate(query: str, context_chunks: list[dict], game: str) -> str:
    """Call OpenAI with retrieved context and return a grounded answer."""
    _load(game)
    reg = _registry[game]

    context_str  = "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in context_chunks)
    user_message = f"Rulebook context:\n\n{context_str}\n\n---\n\nQuestion: {query}"

    response = reg["client"].chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": reg["system_prompt"]},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def query_rag(
    question:  str,
    game:      str,
    k:         int = DEFAULT_K,
    retriever: str = DEFAULT_RETRIEVER,
    rerank:    bool = False,
    verbose:   bool = False,
) -> dict:
    """
    Full RAG pipeline: retrieve top-k chunks, then generate a grounded answer.

    retriever options:
      "dense"  — sentence-transformer embeddings + FAISS  (deep learning)
      "tfidf"  — TF-IDF + cosine similarity               (classical ML)
      "random" — random chunk selection                    (naive baseline)

    If rerank=True, retrieves 10 candidates and re-scores with a cross-encoder
    to select the top 3.
    """
    retrieved = retrieve(question, game, k, retriever, rerank=rerank)
    answer    = generate(question, retrieved, game)

    result = {
        "question":  question,
        "answer":    answer,
        "retriever": retriever,
        "retrieved": [
            {"title": c["title"], "score": c["retrieval_score"], "chunk_idx": c["chunk_idx"]}
            for c in retrieved
        ],
    }

    if verbose:
        print(f"Retriever: {retriever}")
        print(f"Question: {question}\n")
        print("Retrieved chunks:")
        for c in result["retrieved"]:
            print(f"  [{c['score']:.3f}] {c['title']}")
        print(f"\nAnswer:\n{answer}")
        print("\n" + "-" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query the board game RAG pipeline.")
    parser.add_argument("game",      help="Game to query (e.g. catan, monopoly)")
    parser.add_argument("question",  nargs="?", default="", help="Question to ask")
    parser.add_argument("--retriever", choices=RETRIEVERS, default=DEFAULT_RETRIEVER,
                        help="Retrieval strategy (default: dense)")
    parser.add_argument("--rerank", action="store_true",
                        help="Re-rank candidates with cross-encoder (top 10 → top 3)")
    args = parser.parse_args()

    _defaults = {
        "catan":    "What resources do you need to build a settlement?",
        "monopoly": "How much money does each player start with?",
    }
    question = args.question or _defaults.get(args.game, "")
    query_rag(question, game=args.game, retriever=args.retriever, rerank=args.rerank, verbose=True)
