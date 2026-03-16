"""
scripts/build_features.py
--------------------------
Parse a board game rulebook into semantic chunks, embed them with
sentence-transformers, and store a FAISS index + chunk metadata.

Run once per game before using the RAG pipeline (from project root):
    .venv/bin/python scripts/build_features.py catan
    .venv/bin/python scripts/build_features.py monopoly
"""

import re
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _position_based_chunks(text: str, sections: list[tuple], prefix: str) -> list[dict]:
    """
    Generic position-based chunker.
    Finds each section header via regex, then slices the text between headers.
    """
    positions = []
    for title, pattern in sections:
        m = re.search(pattern, text)
        if m:
            positions.append((m.start(), title))
        else:
            print(f"  WARNING: could not locate '{title}'", file=sys.stderr)

    positions.sort(key=lambda x: x[0])

    chunks = []
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        if content:
            chunks.append({"title": f"{prefix}: {title}", "text": content, "source": prefix})
    return chunks


# ---------------------------------------------------------------------------
# Catan chunkers
# ---------------------------------------------------------------------------

_CATAN_GAME_RULES_SECTIONS = [
    ("GAME COMPONENTS",                                           r"GAME COMPONENTS"),
    ("ISLAND CONSTRUCTION AND SETUP",                            r"CONSTRUCTING THE ISLAND"),
    ("TURN OVERVIEW",                                             r"TURN OVERVIEW"),
    ("RESOURCE PRODUCTION",                                       r"1\. RESOURCE PRODUCTION"),
    ("TRADE (Domestic and Maritime)",                             r"2\. TRADE"),
    ("BUILD (Roads, Settlements, Cities, Development Cards)",     r"3\. BUILD"),
    ('ROLLING A "7" AND ACTIVATING THE ROBBER',                  r'a\) Rolling a'),
    ("PLAYING DEVELOPMENT CARDS (Knight, Progress, Victory Point)", r"b\) Playing Development Cards"),
    ("ENDING THE GAME",                                           r"ENDING THE GAME"),
]


def _is_almanac_header(line: str) -> bool:
    """Return True if the line looks like an ALL-CAPS Almanac entry header."""
    line = line.strip()
    if not line or "=" in line or "Pages" in line:
        return False
    if len(line) < 3 or len(line) > 65:
        return False
    upper = sum(1 for c in line if c.isupper())
    lower = sum(1 for c in line if c.islower())
    total_alpha = upper + lower
    if total_alpha < 3:
        return False
    return upper / total_alpha >= 0.80


def _parse_catan_almanac(text: str) -> list[dict]:
    """Split the Catan Almanac section into one chunk per entry."""
    chunks = []
    current_title = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _is_almanac_header(stripped):
            if current_title and current_lines:
                content = "\n".join(current_lines).strip()
                if len(content) > 30:
                    chunks.append({"title": f"Almanac: {current_title}", "text": content, "source": "Almanac"})
            current_title = stripped
            current_lines = [stripped]
        else:
            if current_title is not None:
                current_lines.append(line)

    if current_title and current_lines:
        content = "\n".join(current_lines).strip()
        if len(content) > 30:
            chunks.append({"title": f"Almanac: {current_title}", "text": content, "source": "Almanac"})

    return chunks


def chunk_catan(filepath: str) -> list[dict]:
    """Parse the full Catan rulebook into semantic chunks."""
    with open(filepath) as f:
        text = f.read()

    almanac_marker = "ALMANAC (Pages 6-15)"
    if almanac_marker not in text:
        print("WARNING: Could not find ALMANAC marker; treating entire file as Game Rules.", file=sys.stderr)
        return _position_based_chunks(text, _CATAN_GAME_RULES_SECTIONS, "Game Rules")

    split_pos = text.index(almanac_marker)
    chunks = _position_based_chunks(text[:split_pos], _CATAN_GAME_RULES_SECTIONS, "Game Rules")
    chunks += _parse_catan_almanac(text[split_pos:])
    return chunks


# ---------------------------------------------------------------------------
# Monopoly chunker
# ---------------------------------------------------------------------------

_MONOPOLY_SECTIONS = [
    ("OVERVIEW",                    r"MONOPOLY - Official Rules"),
    ("SPEED DIE RULES",             r"SPEED DIE RULES"),
    ("OBJECT",                      r"OBJECT:"),
    ("PREPARATION",                 r"PREPARATION:"),
    ("THE BANKER",                  r"BANKER:"),
    ("THE BANK",                    r"THE BANK:"),
    ("THE PLAY",                    r"THE PLAY:"),
    ("GO",                          r'"GO":'),
    ("BUYING PROPERTY",             r"BUYING PROPERTY:"),
    ("PAYING RENT",                 r"PAYING RENT:"),
    ("CHANCE AND COMMUNITY CHEST",  r'"CHANCE" AND'),
    ("INCOME TAX",                  r'"INCOME TAX":'),
    ("JAIL",                        r'"JAIL":'),
    ("FREE PARKING",                r'"FREE PARKING":'),
    ("HOUSES",                      r"HOUSES:"),
    ("HOTELS",                      r"HOTELS:"),
    ("BUILDING SHORTAGES",          r"BUILDING SHORTAGES:"),
    ("SELLING PROPERTY",            r"SELLING PROPERTY:"),
    ("MORTGAGES",                   r"MORTGAGES:"),
    ("BANKRUPTCY",                  r"BANKRUPTCY:"),
    ("MISCELLANEOUS",               r"MISCELLANEOUS:"),
]


def chunk_monopoly(filepath: str) -> list[dict]:
    """Parse the Monopoly rulebook into semantic chunks."""
    with open(filepath) as f:
        text = f.read()
    return _position_based_chunks(text, _MONOPOLY_SECTIONS, "Monopoly")


# ---------------------------------------------------------------------------
# Game registry — add new games here
# ---------------------------------------------------------------------------

GAMES: dict[str, callable] = {
    "catan":    chunk_catan,
    "monopoly": chunk_monopoly,
}


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
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index for a board game rulebook.")
    parser.add_argument("game", choices=list(GAMES.keys()), help="Game to index")
    args = parser.parse_args()

    game = args.game
    rulebook_path = f"data/raw/{game}_rulebook.txt"
    out_dir = Path(f"models/{game}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = str(out_dir / "chunks.json")
    index_path  = str(out_dir / "faiss.index")

    print(f"Reading rulebook from '{rulebook_path}'...")
    chunks = GAMES[game](rulebook_path)

    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, c in enumerate(chunks):
        print(f"  {i+1:2d}. {c['title']:<70}  ({len(c['text'])} chars)")

    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved chunks  → '{chunks_path}'")

    import faiss
    index = build_faiss_index(chunks)
    faiss.write_index(index, index_path)
    print(f"Saved index   → '{index_path}'")

    print(f"\nDone. You can now run: .venv/bin/python scripts/model.py {game} 'your question'")


if __name__ == "__main__":
    main()
