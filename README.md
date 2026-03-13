# Catan Rules Assistant — RAG System with Evaluation

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about the board game **Catan** using the official rulebook as its knowledge base.

The core goal is not just a working Q&A bot — it is a **rigorously evaluated** system that demonstrates how two complementary evaluation strategies (correctness testing and adversarial stress testing) reveal specific failure modes and guide concrete improvements.

---

## What It Does

1. **Chunks** the Catan rulebook into 42 semantically meaningful sections (9 Game Rules sections + 33 Almanac entries)
2. **Embeds** each chunk using a sentence transformer model and indexes them in a FAISS vector store
3. **Retrieves** the top-3 most relevant chunks for any user question using cosine similarity
4. **Generates** a grounded answer using GPT-4o-mini, instructed to cite sources and say "the rules don't specify" when context is insufficient
5. **Evaluates** the system on 32 curated questions across two test sets, computing retrieval precision, coverage, and hallucination rate

---

## Tech Stack

| Component      | Technology                                       |
| -------------- | ------------------------------------------------ |
| Embeddings     | `sentence-transformers` — `all-MiniLM-L6-v2`     |
| Vector store   | `faiss-cpu` — in-memory cosine similarity index  |
| LLM            | OpenAI `gpt-4o-mini` via the `openai` Python SDK |
| Env management | `python-dotenv`                                  |
| Language       | Python 3.10+                                     |
| Runtime        | Isolated `.venv` virtual environment             |

---

## Project Structure

```
├── catan_rulebook.txt       # Official Catan rulebook (source document)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Creates .venv and installs all deps
├── .env                     # API key (OPENAI_API_KEY — not committed)
│
├── chunk_and_index.py       # Step 1: parse rulebook → 42 chunks → FAISS index
├── rag_pipeline.py          # Step 2: retrieve top-k + generate with OpenAI
│
├── eval_data.json           # 20 correctness Qs + 12 stress-test Qs with gold answers
├── evaluate.py              # Runs full evaluation → results.csv + summary stats
│
├── demo.py                  # Interactive class demo (curated questions + free input)
├── analysis.md              # Written error analysis: failure modes + proposed fixes
│
├── chunks.json              # Generated: 42 parsed rulebook chunks
├── faiss.index              # Generated: FAISS vector index
└── results.csv              # Generated: evaluation results with scores and notes
```

---

## Setup & Run

### 1. Install dependencies (one time)

```bash
bash setup.sh
```

This creates a `.venv/` virtual environment and installs all packages. Nothing is installed globally.

### 2. Add your OpenAI API key

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Build the index (one time)

```bash
.venv/bin/python chunk_and_index.py
```

Parses the rulebook into 42 chunks, embeds them, and saves `chunks.json` + `faiss.index`.

### 4. Ask a single question

```bash
.venv/bin/python rag_pipeline.py "What resources do you need to build a settlement?"
```

### 5. Run the full evaluation

```bash
.venv/bin/python evaluate.py
```

Runs all 32 questions, prints metrics, writes `results.csv`.

### 6. Run the interactive demo

```bash
.venv/bin/python demo.py
```

Walks through 4 curated questions (including a known failure), then goes interactive.

---

## Evaluation Design

Two complementary test sets are used:

### Set 1 — Correctness Questions (20 questions)

Straightforward factual questions with single, unambiguous answers in the rulebook.

**Example:**

> "What resources do you need to build a settlement?" → _Brick, Lumber, Wool, and Grain_

**Metrics:** Answer correctness (0 / 0.5 / 1), Retrieval Precision@3

### Set 2 — Stress Test Questions (12 questions)

Adversarial questions in three sub-categories:

| Sub-category          | Description                            | Example                                                   |
| --------------------- | -------------------------------------- | --------------------------------------------------------- |
| **A — Multi-section** | Requires synthesizing 2+ chunks        | "I roll a 7 and want to play a knight card — what first?" |
| **B — Edge cases**    | Rules address it but it's easy to miss | "Can I trade 2 wool for 1 wool?"                          |
| **C — Unanswerable**  | Rules are genuinely silent             | "Can I make a binding future-turn trade promise?"         |

**Metrics:** Answer correctness, Retrieval Coverage (all required chunks found?), Hallucination Rate (Category C)

---

## Results

| Category                 | N      | Retrieval Precision@3 | Avg Correctness |
| ------------------------ | ------ | --------------------- | --------------- |
| Correctness questions    | 20     | 100%                  | 0.78 / 1.0      |
| Stress A — multi-section | 3      | 100%                  | 0.50 / 1.0      |
| Stress B — edge cases    | 5      | 100%                  | 0.90 / 1.0      |
| Stress C — unanswerable  | 4      | 100%                  | 0.88 / 1.0      |
| **Overall**              | **32** | **100%**              | **0.75 / 1.0**  |

**Retrieval Coverage** (all required chunks found for multi-section Qs): **78%**

**Hallucination Rate** (Category C — model invented a definitive rule): **0%**

---

## Key Findings

Retrieval Precision@3 was 100% — at least one correct chunk was always retrieved. The failures are almost entirely **generation problems**, not retrieval problems. This distinction was only possible because the evaluation measured retrieval and correctness independently.

### Failure Modes Identified

| Mode                    | Questions     | Root Cause                                            | Proposed Fix                                      |
| ----------------------- | ------------- | ----------------------------------------------------- | ------------------------------------------------- |
| Over-cautious grounding | C04, C16      | Model abstains even when answer is clearly in context | Relax "I don't know" instruction in system prompt |
| Retrieval miss          | C08, C09, S02 | Correct chunk not in top-3 (semantic mismatch)        | Increase k to 5; enrich chunk headers             |
| Multi-hop reasoning     | S01, S06      | Confused synthesis across two chunks                  | Query decomposition                               |
| Physical card gap       | C09           | Building costs only on physical card, not in text     | Add synthetic "building costs" chunk              |

Full analysis in [`analysis.md`](analysis.md).

---

## Chunking Strategy

Rather than naive fixed-size splitting, the rulebook is split along its own structure:

- **Game Rules** (pages 2–5): 9 sections matched by known header patterns (`1. RESOURCE PRODUCTION`, `a) Rolling a "7"`, etc.)
- **Almanac** (pages 6–15): 33 entries auto-detected by an ≥80%-uppercase line heuristic

This produces chunks that align with how players actually look up rules.

---

## Example Interaction

```
Q: What happens when you roll a 7?

Retrieved chunks:
  [0.712]  Almanac: ROLLING A "7" AND ACTIVATING THE ROBBER
  [0.489]  Almanac: NUMBER TOKENS
  [0.441]  Game Rules: ENDING THE GAME

Answer:
When you roll a "7," the following occurs:

1. No players receive resources.
2. Every player with more than 7 resource cards must discard half (rounded down).
3. You must move the robber to any other terrain hex, then steal 1 resource card
   from a player with a settlement or city adjacent to that hex.

According to [Almanac: ROLLING A "7" AND ACTIVATING THE ROBBER]...
```
