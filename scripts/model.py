"""
scripts/model.py
----------------
Retrieval-Augmented Generation pipeline for Catan rules questions.

Usage (from project root):
    .venv/bin/python scripts/model.py 'What resources do you need to build a settlement?'
    .venv/bin/python scripts/model.py  # uses a default demo question

Requires scripts/build_features.py to have been run first
(produces data/processed/chunks.json + models/faiss.index).
Requires OPENAI_API_KEY to be set in the environment.
"""

import json
import os
import sys

# Load .env file if present (OPENAI_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "models/faiss.index"
DEFAULT_K = 3
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an expert on the board game Catan. Answer the user's question using ONLY \
the rulebook sections provided as context.

Rules:
1. Base your answer SOLELY on the provided context. Do not use outside knowledge.
2. If the answer can be inferred or directly found in the context, state it clearly \
and cite the source section, e.g. "According to [Almanac: ROBBER]...".
3. Only say "The rules don't specify this." if NO relevant information appears in ANY \
of the retrieved sections. Do not abstain if the answer is present or can be reasonably \
inferred from the context.
4. Do not invent or extrapolate rules beyond what the text says.
5. Be concise and precise.\
"""

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model = None
_index = None
_chunks: list[dict] | None = None
_client = None


def _load():
    global _model, _index, _chunks, _client
    if _model is not None:
        return

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: .venv/bin/pip install -r requirements.txt",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
        print(f"Index files not found. Run scripts/build_features.py first.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    _index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH) as f:
        _chunks = json.load(f)

    _client = OpenAI()
    print(f"Ready — {len(_chunks)} chunks loaded.\n")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k most relevant chunks for the query."""
    import faiss
    _load()

    query_emb = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    scores, indices = _index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = _chunks[idx].copy()
            chunk["retrieval_score"] = float(score)
            chunk["chunk_idx"] = int(idx)
            results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(query: str, context_chunks: list[dict]) -> str:
    """Call OpenAI with the retrieved context and return the answer."""
    _load()

    context_str = "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['text']}"
        for c in context_chunks
    )

    user_message = (
        f"Rulebook context:\n\n{context_str}\n\n---\n\nQuestion: {query}"
    )

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def query_rag(question: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Full RAG pipeline: retrieve top-k chunks, then generate an answer.

    Returns a dict with keys:
      question, answer, retrieved (list of {title, score, chunk_idx})
    """
    retrieved = retrieve(question, k)
    answer = generate(question, retrieved)

    result = {
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "title": c["title"],
                "score": c["retrieval_score"],
                "chunk_idx": c["chunk_idx"],
            }
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
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What resources do you need to build a settlement?"
    )
    query_rag(question, verbose=True)
