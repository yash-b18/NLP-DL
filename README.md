# Board Game Rules Assistant — Multi-Game RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about board game rules using official rulebooks as the knowledge base. Currently supports **Catan** and **Monopoly**, with a game-agnostic architecture designed to support additional games.

The system is rigorously evaluated using two complementary test sets per game — correctness questions and adversarial stress tests — to reveal specific failure modes and guide improvements.

---

## What It Does

1. **Chunks** each game's rulebook into semantically meaningful sections using game-specific structure-aware parsing
2. **Embeds** each chunk using a sentence transformer model and indexes them in a FAISS vector store per game
3. **Retrieves** the top-3 most relevant chunks for any user question using cosine similarity
4. **Generates** a grounded answer using GPT-4o-mini, instructed to cite sources and say "the rules don't specify" when context is insufficient
5. **Evaluates** the system on 32 curated questions per game across two test sets, computing retrieval precision, coverage, and hallucination rate

---

## Tech Stack

| Component      | Technology                                       |
| -------------- | ------------------------------------------------ |
| Embeddings     | `sentence-transformers` — `all-MiniLM-L6-v2`     |
| Vector store   | `faiss-cpu` — per-game cosine similarity index   |
| LLM            | OpenAI `gpt-4o-mini` via the `openai` Python SDK |
| UI             | `streamlit`                                      |
| Env management | `python-dotenv`                                  |
| Language       | Python 3.10+                                     |
| Runtime        | Isolated `.venv` virtual environment             |

---

## Project Structure

```
├── main.py                          # Streamlit UI — run this to launch the app
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Creates .venv and installs all deps
├── .env                             # API key (OPENAI_API_KEY — not committed)
│
├── scripts/
│   ├── build_features.py            # Parse rulebook → chunks → FAISS index (per game)
│   ├── model.py                     # Retrieve top-k + generate with OpenAI (per game)
│   ├── evaluate.py                  # Run evaluation → {game}_results.csv + summary stats
│   └── demo.py                      # Interactive CLI demo (curated questions + free input)
│
├── data/
│   ├── raw/
│   │   ├── catan_rulebook.txt       # Official Catan rulebook
│   │   ├── catan_eval.json          # 20 correctness Qs + 12 stress-test Qs (Catan)
│   │   ├── monopoly_rulebook.txt    # Official Monopoly rulebook
│   │   └── monopoly_eval.json       # 20 correctness Qs + 12 stress-test Qs (Monopoly)
│   └── outputs/
│       ├── catan_results.csv        # Catan evaluation results with scores
│       └── monopoly_results.csv     # Monopoly evaluation results with scores
│
├── models/
│   ├── catan/
│   │   ├── chunks.json              # Generated: 42 parsed Catan chunks
│   │   └── faiss.index              # Generated: Catan FAISS vector index
│   └── monopoly/
│       ├── chunks.json              # Generated: 21 parsed Monopoly chunks
│       └── faiss.index              # Generated: Monopoly FAISS vector index
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

### 3. Build the index for each game (one time per game)

```bash
.venv/bin/python scripts/build_features.py catan
.venv/bin/python scripts/build_features.py monopoly
```

Parses each rulebook into chunks, embeds them, and saves to `models/{game}/`.

### 4. Ask a single question

```bash
.venv/bin/python scripts/model.py catan "What resources do you need to build a settlement?"
.venv/bin/python scripts/model.py monopoly "How much money does each player start with?"
```

### 5. Run the full evaluation

```bash
.venv/bin/python scripts/evaluate.py catan
.venv/bin/python scripts/evaluate.py monopoly
```

Runs all 32 questions per game, prints metrics, writes `data/outputs/{game}_results.csv`.

### 6. Launch the UI

```bash
.venv/bin/streamlit run main.py
```

### 7. Run the interactive demo

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

| Category                 | N      | Retrieval P@3 | Avg Correctness |
| ------------------------ | ------ | ------------- | --------------- |
| Correctness questions    | 20     | 100%          | 88%             |
| Stress A — multi-section | 3      | 100%          | 50%             |
| Stress B — edge cases    | 5      | 100%          | 90%             |
| Stress C — unanswerable  | 4      | 100%          | 88%             |
| **Overall**              | **32** | **100%**      | **83%**         |

### Monopoly

| Category                 | N      | Retrieval P@3 | Avg Correctness |
| ------------------------ | ------ | ------------- | --------------- |
| Correctness questions    | 20     | 100%          | 93%             |
| Stress A — multi-section | 3      | 100%          | 100%            |
| Stress B — edge cases    | 5      | 100%          | 100%            |
| Stress C — unanswerable  | 5      | 100%          | 100%            |
| **Overall**              | **32** | **100%**      | **97%**         |

**Hallucination Rate (both games):** 0%

---

## Chunking Strategy

Each game uses a structure-aware chunker tailored to its rulebook format:

**Catan (42 chunks)**
- Game Rules (9 chunks): sections matched by known header regex patterns
- Almanac (33 chunks): entries auto-detected by ≥80%-uppercase line heuristic

**Monopoly (21 chunks)**
- 21 named sections matched by known header patterns (`PREPARATION:`, `"JAIL":`, `MORTGAGES:`, etc.)

Adding a new game requires: a rulebook `.txt` file, a chunker function, and an eval `.json` file — nothing else in the pipeline changes.

---

## Key Findings

Retrieval Precision@3 was 100% across both games. Failures are concentrated in **generation**, not retrieval — the model occasionally abstains on questions where the answer is present in the retrieved context. This distinction was only possible because the evaluation measured retrieval and generation independently.

Full error analysis in [`notebooks/analysis.md`](notebooks/analysis.md).
