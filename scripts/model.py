"""
scripts/model.py
----------------
Retrieval-Augmented Generation pipeline for board game rules questions.

Usage (from project root):
    .venv/bin/python scripts/model.py catan 'What resources do you need to build a settlement?'
    .venv/bin/python scripts/model.py monopoly 'How much money does each player start with?'
    .venv/bin/python scripts/model.py catan   # uses a default demo question

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

DEFAULT_K    = 3
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Per-game singleton registry
# ---------------------------------------------------------------------------

_registry: dict[str, dict] = {}   # game -> {model, index, chunks, client, system_prompt}


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
    """Lazy-load and cache the model, index, and chunks for the given game."""
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
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loading FAISS index for '{game}'...")
    index = faiss.read_index(index_path)

    with open(chunks_path) as f:
        chunks = json.load(f)

    _registry[game] = {
        "model":         model,
        "index":         index,
        "chunks":        chunks,
        "client":        OpenAI(),
        "system_prompt": _make_system_prompt(game),
    }
    print(f"Ready — {len(chunks)} chunks loaded for '{game}'.\n")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, game: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k most relevant chunks for the query from the given game's index."""
    import faiss
    _load(game)
    reg = _registry[game]

    query_emb = reg["model"].encode([query], convert_to_numpy=True)
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(query: str, context_chunks: list[dict], game: str) -> str:
    """Call OpenAI with the retrieved context and return a grounded answer."""
    _load(game)
    reg = _registry[game]

    context_str = "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['text']}"
        for c in context_chunks
    )
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

def query_rag(question: str, game: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Full RAG pipeline: retrieve top-k chunks, then generate a grounded answer.

    Returns a dict with keys: question, answer, retrieved (list of {title, score, chunk_idx})
    """
    retrieved = retrieve(question, game, k)
    answer    = generate(question, retrieved, game)

    result = {
        "question":  question,
        "answer":    answer,
        "retrieved": [
            {"title": c["title"], "score": c["retrieval_score"], "chunk_idx": c["chunk_idx"]}
            for c in retrieved
        ],
    }

    if verbose:
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
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/model.py <game> [question]")
        print("  game: catan | monopoly")
        sys.exit(1)

    game_arg = sys.argv[1]
    _defaults = {
        "catan":    "What resources do you need to build a settlement?",
        "monopoly": "How much money does each player start with?",
    }
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else _defaults.get(game_arg, "")
    query_rag(question, game=game_arg, verbose=True)
