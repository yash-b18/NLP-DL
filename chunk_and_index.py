"""
chunk_and_index.py
------------------
Step 1: Parse the Catan rulebook into semantic chunks, embed them with
sentence-transformers, and store the FAISS index + chunk metadata.

Run once before using the RAG pipeline:
    .venv/bin/python chunk_and_index.py
"""

import re
import json
import sys
from pathlib import Path

RULEBOOK_PATH = "catan_rulebook.txt"
CHUNKS_PATH = "chunks.json"
INDEX_PATH = "faiss.index"


# ---------------------------------------------------------------------------
# Game Rules parsing
# ---------------------------------------------------------------------------

# Each tuple is (chunk_title, regex_to_locate_the_section_start).
# Sections are ordered as they appear in the document.
GAME_RULES_SECTIONS = [
    ("GAME COMPONENTS",
     r"GAME COMPONENTS"),
    ("ISLAND CONSTRUCTION AND SETUP",
     r"CONSTRUCTING THE ISLAND"),
    ("TURN OVERVIEW",
     r"TURN OVERVIEW"),
    ("RESOURCE PRODUCTION",
     r"1\. RESOURCE PRODUCTION"),
    ("TRADE (Domestic and Maritime)",
     r"2\. TRADE"),
    ("BUILD (Roads, Settlements, Cities, Development Cards)",
     r"3\. BUILD"),
    ('ROLLING A "7" AND ACTIVATING THE ROBBER',
     r'a\) Rolling a'),
    ("PLAYING DEVELOPMENT CARDS (Knight, Progress, Victory Point)",
     r"b\) Playing Development Cards"),
    ("ENDING THE GAME",
     r"ENDING THE GAME"),
]


def parse_game_rules_chunks(text: str) -> list[dict]:
    """Split the Game Rules section into semantic chunks."""
    positions = []
    for title, pattern in GAME_RULES_SECTIONS:
        m = re.search(pattern, text)
        if m:
            positions.append((m.start(), title))
        else:
            print(f"  WARNING: could not locate '{title}' in Game Rules", file=sys.stderr)

    positions.sort(key=lambda x: x[0])

    chunks = []
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        if content:
            chunks.append({
                "title": f"Game Rules: {title}",
                "text": content,
                "source": "Game Rules",
            })
    return chunks


# ---------------------------------------------------------------------------
# Almanac parsing
# ---------------------------------------------------------------------------

def is_almanac_header(line: str) -> bool:
    """
    Return True if the line looks like an ALL-CAPS Almanac entry header.
    Filters out separators (=====), the 'ALMANAC (Pages...)' title line,
    and any line with significant lowercase content.
    """
    line = line.strip()
    if not line:
        return False
    if "=" in line:
        return False
    if "Pages" in line:
        return False
    if len(line) < 3 or len(line) > 65:
        return False
    upper = sum(1 for c in line if c.isupper())
    lower = sum(1 for c in line if c.islower())
    total_alpha = upper + lower
    if total_alpha < 3:
        return False
    return upper / total_alpha >= 0.80


def parse_almanac_chunks(text: str) -> list[dict]:
    """Split the Almanac section into one chunk per entry."""
    chunks = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and is_almanac_header(stripped):
            # Save the previous entry
            if current_title and current_lines:
                content = "\n".join(current_lines).strip()
                if len(content) > 30:
                    chunks.append({
                        "title": f"Almanac: {current_title}",
                        "text": content,
                        "source": "Almanac",
                    })
            current_title = stripped
            current_lines = [stripped]
        else:
            if current_title is not None:
                current_lines.append(line)

    # Flush the last entry
    if current_title and current_lines:
        content = "\n".join(current_lines).strip()
        if len(content) > 30:
            chunks.append({
                "title": f"Almanac: {current_title}",
                "text": content,
                "source": "Almanac",
            })

    return chunks


# ---------------------------------------------------------------------------
# Top-level chunker
# ---------------------------------------------------------------------------

def chunk_rulebook(filepath: str) -> list[dict]:
    """Parse the full Catan rulebook into semantic chunks."""
    with open(filepath) as f:
        text = f.read()

    almanac_marker = "ALMANAC (Pages 6-15)"
    if almanac_marker not in text:
        print("WARNING: Could not find ALMANAC marker; treating entire file as Game Rules.",
              file=sys.stderr)
        return parse_game_rules_chunks(text)

    split_pos = text.index(almanac_marker)
    game_rules_text = text[:split_pos]
    almanac_text = text[split_pos:]

    chunks = parse_game_rules_chunks(game_rules_text)
    chunks += parse_almanac_chunks(almanac_text)
    return chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(chunks: list[dict]):
    """Embed all chunks and build a cosine-similarity FAISS index."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: .venv/bin/pip install -r requirements.txt")
        sys.exit(1)

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize → use IndexFlatIP as cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading rulebook from '{RULEBOOK_PATH}'...")
    chunks = chunk_rulebook(RULEBOOK_PATH)

    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, c in enumerate(chunks):
        tag = "GR" if c["source"] == "Game Rules" else "AL"
        print(f"  {i+1:2d}. [{tag}] {c['title']:<65}  ({len(c['text'])} chars)")

    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved chunks  → '{CHUNKS_PATH}'")

    import faiss
    index = build_faiss_index(chunks)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index   → '{INDEX_PATH}'")

    print("\nDone. You can now run: .venv/bin/python rag_pipeline.py 'your question here'")


if __name__ == "__main__":
    main()
