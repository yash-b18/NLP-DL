# Board Game Rules Assistant — Multi-Game RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about board game rules using official rulebooks as the knowledge base. Currently supports **Catan** and **Monopoly**, with a game-agnostic architecture designed to support additional games.

Three retrieval strategies are implemented and compared — a naive baseline, a classical ML approach, and a deep learning approach — with rigorous evaluation per game to reveal failure modes and guide improvements.

---

## What It Does

1. **Chunks** each game's rulebook into semantically meaningful sections using game-specific structure-aware parsing
2. **Indexes** chunks using three retrieval strategies (dense embeddings, TF-IDF, random)
3. **Retrieves** the top-k most relevant chunks for any user question
4. **Generates** a grounded answer using GPT-4o-mini, instructed to cite sources and say "the rules don't specify" when context is insufficient
5. **Evaluates** the system on 32 curated questions per game across two test sets
6. **Experiments** with k-sensitivity analysis comparing all three retrievers

---

## Tech Stack

| Component         | Technology                                        |
| ----------------- | ------------------------------------------------- |
| Embeddings (DL)   | `sentence-transformers` — `all-MiniLM-L6-v2`      |
| Classical ML      | `scikit-learn` — TF-IDF + cosine similarity       |
| Vector store      | `faiss-cpu` — per-game cosine similarity index    |
| LLM               | OpenAI `gpt-4o-mini` via the `openai` Python SDK  |
| UI                | `streamlit`                                       |
| Env management    | `python-dotenv`                                   |
| Language          | Python 3.10+                                      |
| Runtime           | Isolated `.venv` virtual environment              |

---

## Modeling Approaches

Three retrieval strategies are implemented, all sharing the same GPT-4o-mini generation step:

| Approach | Strategy | Description |
|---|---|---|
| **Naive baseline** | `random` | Randomly selects k chunks — no query signal |
| **Classical ML** | `tfidf` | TF-IDF bag-of-words + cosine similarity |
| **Deep learning** | `dense` | sentence-transformer dense embeddings + FAISS |

**k=3 comparison across both games (full evaluation):**

| Retriever | Catan P@3 | Catan Correctness | Monopoly P@3 | Monopoly Correctness |
|---|---|---|---|---|
| Random (baseline) | 25% | 23% | 12% | 28% |
| TF-IDF (classical ML) | 97% | 72% | 91% | 88% |
| Dense (deep learning) | 100% | 86% | 94% | 95% |

**Hallucination Rate (all retrievers, both games):** 0%

---

## Project Structure

```
├── main.py                          # Streamlit UI — run this to launch the app
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Creates .venv and installs all deps
├── .env                             # API key (OPENAI_API_KEY — not committed)
│
├── scripts/
│   ├── build_features.py            # Parse rulebook → chunks → FAISS + TF-IDF indexes
│   ├── model.py                     # RAG pipeline: retrieve (dense/tfidf/random) + generate
│   ├── evaluate.py                  # Run evaluation → {game}[_{retriever}]_results.csv
│   ├── experiment.py                # k-sensitivity analysis across all three retrievers
│   └── demo.py                      # Interactive CLI demo
│
├── data/
│   ├── raw/
│   │   ├── catan_rulebook.txt       # Official Catan rulebook
│   │   ├── catan_eval.json          # 20 correctness Qs + 12 stress-test Qs (Catan)
│   │   ├── monopoly_rulebook.txt    # Official Monopoly rulebook
│   │   └── monopoly_eval.json       # 20 correctness Qs + 12 stress-test Qs (Monopoly)
│   └── outputs/
│       ├── catan_results.csv                      # Catan dense eval results (scored)
│       ├── catan_tfidf_results.csv                # Catan TF-IDF eval results (scored)
│       ├── catan_random_results.csv               # Catan random eval results (scored)
│       ├── monopoly_results.csv                   # Monopoly dense eval results (scored)
│       ├── monopoly_tfidf_results.csv             # Monopoly TF-IDF eval results (scored)
│       ├── monopoly_random_results.csv            # Monopoly random eval results (scored)
│       └── catan_experiment_k_sensitivity.csv     # k-sensitivity experiment results
│
├── models/
│   ├── catan/
│   │   ├── chunks.json              # Generated: 42 parsed Catan chunks
│   │   ├── faiss.index              # Generated: Catan FAISS vector index
│   │   ├── tfidf_matrix.npz         # Generated: Catan TF-IDF sparse matrix
│   │   └── tfidf_vectorizer.pkl     # Generated: Catan TF-IDF vectorizer
│   └── monopoly/
│       ├── chunks.json              # Generated: 21 parsed Monopoly chunks
│       ├── faiss.index              # Generated: Monopoly FAISS vector index
│       ├── tfidf_matrix.npz         # Generated: Monopoly TF-IDF sparse matrix
│       └── tfidf_vectorizer.pkl     # Generated: Monopoly TF-IDF vectorizer
│
└── notebooks/
    └── analysis.md                  # Error analysis: failure modes + proposed fixes
```

---

## Setup & Run

### 1. Install dependencies (one time)

```bash
bash setup.sh
```

Creates a `.venv/` virtual environment and installs all packages. Nothing is installed globally.

### 2. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Build the indexes for each game (one time per game)

```bash
.venv/bin/python scripts/build_features.py catan
.venv/bin/python scripts/build_features.py monopoly
```

Builds both the FAISS (dense) and TF-IDF (classical ML) indexes. Saved to `models/{game}/`.

### 4. Ask a single question

```bash
.venv/bin/python scripts/model.py catan "What resources do you need to build a settlement?"
.venv/bin/python scripts/model.py monopoly "How much money does each player start with?" --retriever tfidf
.venv/bin/python scripts/model.py catan "What happens when you roll a 7?" --retriever random
```

### 5. Run the full evaluation

```bash
# Dense (deep learning) — default
.venv/bin/python scripts/evaluate.py catan
.venv/bin/python scripts/evaluate.py monopoly

# TF-IDF (classical ML)
.venv/bin/python scripts/evaluate.py catan --retriever tfidf

# Random (naive baseline)
.venv/bin/python scripts/evaluate.py catan --retriever random
```

### 6. Run the k-sensitivity experiment (no API calls)

```bash
.venv/bin/python scripts/experiment.py catan
.venv/bin/python scripts/experiment.py monopoly
.venv/bin/python scripts/experiment.py all
```

Compares Precision@k and Coverage across k=1–10 for all three retrievers. No OpenAI calls needed.

### 7. Launch the UI

```bash
.venv/bin/streamlit run main.py
```

The UI includes a game selector and a retriever selector. Evaluation metrics in the sidebar update based on the selected game and retriever.

### 8. Run the interactive demo

```bash
.venv/bin/python scripts/demo.py catan
.venv/bin/python scripts/demo.py monopoly
```

---

## Evaluation Design

Two complementary test sets are used per game:

### Set 1 — Correctness Questions (20 questions)

Straightforward factual questions with single, unambiguous answers in the rulebook.

**Metrics:** Answer correctness (0 / 0.5 / 1), Retrieval Precision@3

### Set 2 — Stress Test Questions (12 questions)

Adversarial questions in three sub-categories:

| Sub-category          | Description                            | Example                                                      |
| --------------------- | -------------------------------------- | ------------------------------------------------------------ |
| **A — Multi-section** | Requires synthesizing 2+ chunks        | "I roll a 7 and want to play a knight card — what first?"    |
| **B — Edge cases**    | Rules address it but it's easy to miss | "Can I trade 2 wool for 1 wool?"                             |
| **C — Unanswerable**  | Rules are genuinely silent             | "What happens if two players tie on the initial dice roll?"  |

**Metrics:** Answer correctness, Retrieval Coverage, Hallucination Rate (Category C)

---

## Results

### Catan

| Category                 | N      | Dense P@3 | Dense Correct | TF-IDF P@3 | TF-IDF Correct | Random P@3 | Random Correct |
| ------------------------ | ------ | --------- | ------------- | ---------- | -------------- | ---------- | -------------- |
| Correctness questions    | 20     | 100%      | 88%           | 95%        | 85%            | 20%        | 18%            |
| Stress A — multi-section | 3      | 100%      | 67%           | 100%       | 0%             | 33%        | 50%            |
| Stress B — edge cases    | 5      | 100%      | 100%          | 100%       | 70%            | 0%         | 0%             |
| Stress C — unanswerable  | 4      | 100%      | 75%           | 100%       | 62%            | 75%        | 62%            |
| **Overall**              | **32** | **100%**  | **86%**       | **97%**    | **72%**        | **25%**    | **23%**        |

### Monopoly

| Category                 | N      | Dense P@3 | Dense Correct | TF-IDF P@3 | TF-IDF Correct | Random P@3 | Random Correct |
| ------------------------ | ------ | --------- | ------------- | ---------- | -------------- | ---------- | -------------- |
| Correctness questions    | 20     | 95%       | 92%           | 90%        | 85%            | 15%        | 15%            |
| Stress A — multi-section | 3      | 100%      | 100%          | 100%       | 100%           | 33%        | 33%            |
| Stress B — edge cases    | 4      | 100%      | 100%          | 100%       | 75%            | 0%         | 0%             |
| Stress C — unanswerable  | 5      | 80%       | 100%          | 80%        | 100%           | 0%         | 100%           |
| **Overall**              | **32** | **94%**   | **95%**       | **91%**    | **88%**        | **12%**    | **28%**        |

**Hallucination Rate (all retrievers, both games):** 0%

---

## Chunking Strategy

Each game uses a structure-aware chunker tailored to its rulebook format:

**Catan (42 chunks)**
- Game Rules (9 chunks): sections matched by known header regex patterns
- Almanac (33 chunks): entries auto-detected by ≥80%-uppercase line heuristic

**Monopoly (21 chunks)**
- 21 named sections matched by known header patterns (`PREPARATION:`, `"JAIL":`, `MORTGAGES:`, etc.)

Adding a new game requires: a rulebook `.txt` file, a chunker function registered in `GAMES`, and an eval `.json` file — nothing else in the pipeline changes.

---

## Key Findings

- **Dense retrieval dominates** on both games: 100%/94% Precision@3 vs 97%/91% for TF-IDF and 25%/12% for random
- **TF-IDF is competitive on correctness** — 85% correctness on Catan vs 88% on Monopoly, degrading mainly on multi-section stress questions where semantic retrieval is critical
- **Random baseline collapses on edge-case questions** — 0% P@3 on Stress B for both games; it only succeeds on unanswerable (C) questions where abstention is always correct
- **Failures are generation problems, not retrieval problems** — TF-IDF achieved 97% P@3 on Catan but only 72% correctness; the gap is from correct chunks being retrieved but answers still failing
- **Hallucination resistance is strong** across all 6 game/retriever combinations — the grounding prompt prevents rule invention regardless of retrieval quality
- **Monopoly is easier than Catan** — dense correctness 95% vs 86%; Monopoly's rulebook is more linear with fewer cross-references to physical components

Full error analysis in [`notebooks/analysis.md`](notebooks/analysis.md).
