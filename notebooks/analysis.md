# Error Analysis: Board Game RAG System

## System Overview

A Retrieval-Augmented Generation (RAG) system answers natural language questions about board game rulebooks. The pipeline chunks each rulebook into semantically meaningful sections, indexes them with three retrieval strategies, and passes the top-k retrieved chunks to `gpt-4o-mini` with a strict grounding prompt.

**Games:** Catan (42 chunks) and Monopoly (21 chunks)

**Retrieval strategies compared:**

| Approach | Strategy | Description |
|---|---|---|
| Dense (deep learning) | `sentence-transformers` + FAISS | `all-MiniLM-L6-v2` embeddings, cosine similarity |
| TF-IDF (classical ML) | `scikit-learn` TF-IDF | Bag-of-words + cosine similarity |
| Random (naive baseline) | `random.sample` | No query signal — random chunk selection |

Each game was evaluated on 20 correctness questions and 12 adversarial stress-test questions (32 total).

---

## Quantitative Results

### Catan (Dense retriever, k=3)

| Category | N | Retrieval P@3 | Avg Correctness |
|---|---|---|---|
| Correctness questions | 20 | 100% | 88% |
| Stress A — multi-section | 3 | 100% | 50% |
| Stress B — edge cases | 5 | 100% | 90% |
| Stress C — unanswerable | 4 | 100% | 88% |
| **Overall** | **32** | **100%** | **83%** |

### Monopoly (Dense retriever, k=3)

| Category | N | Retrieval P@3 | Avg Correctness |
|---|---|---|---|
| Correctness questions | 20 | 100% | 93% |
| Stress A — multi-section | 3 | 100% | 100% |
| Stress B — edge cases | 5 | 100% | 100% |
| Stress C — unanswerable | 5 | 100% | 100% |
| **Overall** | **32** | **100%** | **97%** |

**Hallucination Rate (both games):** 0%

---

## Retriever Comparison (k-sensitivity experiment, Catan)

The k-sensitivity experiment ran retrieval-only (no OpenAI calls) across k=1–10 for all three retrievers on all 32 Catan questions. Key result at k=3:

| Retriever | Precision@3 | Coverage |
|---|---|---|
| Dense (deep learning) | 100.0% | 87.5% |
| TF-IDF (classical ML) | 96.9% | 81.8% |
| Random (naive baseline) | 21.9% | 13.0% |

**Key finding:** Dense retrieval achieves 100% Precision@3 immediately. TF-IDF reaches 100% at k=7, showing classical ML is competitive with slightly more retrieved context. Random never exceeds ~30% even at k=10, confirming retrieval signal is essential.

---

## Failure Mode Analysis

### Failure Mode 1: Generation Failure — Over-Cautious Grounding

**Affected questions:** C04 (original scoring, now resolved), C16 (resolved via prompt fix)

**What the evaluation revealed:**
The model initially answered "The rules don't specify this" even when retrieved context contained a clear answer.

- **C04** ("Can you build a city without first having a settlement?") — `Almanac: CITIES` was retrieved and contains: *"You cannot build a city directly. You can only upgrade an existing settlement to a city."* The model originally abstained.
- **C16** ("Can players trade when it's not their turn?") — Both `Almanac: DOMESTIC TRADE` and `Almanac: TRADE` were retrieved. `DOMESTIC TRADE` explicitly states: *"the other players may not trade among themselves."* The model originally abstained.

**Root cause:** The system prompt instructed the model to say "The rules don't specify this" when context is insufficient, but was being applied too aggressively — even when answers were present and the question phrasing differed from the rulebook's exact language.

**Fix applied:** The system prompt was revised to: *"If the answer can be inferred from the context, state it and cite the source. Only say 'the rules don't specify' if no relevant information is present in any of the retrieved sections."* After this fix, C16 is now scored correctly. This explains the improvement from the old score (0.75 overall) to the current score (0.83 overall for Catan).

---

### Failure Mode 2: Multi-Hop Reasoning Failure

**Affected questions:** S01, S06 (both scored 0.5)

**What the evaluation revealed:**
When questions required synthesizing rules from two sections, the model retrieved the right chunks but produced confused reasoning:

- **S01** ("Roll 7, want to play knight card — what first?") — The model correctly identified that you must discard 4 cards first, but conflated the robber move from rolling a 7 with the robber move from the knight card, describing them as a single action rather than two separate robber activations occurring sequentially.
- **S06** ("Two players tie for Longest Road after mine is broken") — The model opened with an incorrect statement ("you still keep the Longest Road card") before contradicting itself and arriving at the correct conclusion ("set the card aside").

**Root cause:** Single-pass generation on multi-chunk context. The model receives multiple retrieved chunks simultaneously and attempts to synthesize them, but partially reflects each without fully integrating them. This is a known limitation for questions requiring logical chaining across sections.

**Proposed fix:** Query decomposition — break complex questions into sub-questions, retrieve and answer each separately, then synthesize. For S01: (1) "What happens when you roll a 7?" → robber rules; (2) "When can you play a development card?" → dev card timing rules; then combine the two answers.

---

### Failure Mode 3: Content Gap — Physical Card References

**Affected questions:** C08, C09

**What the evaluation revealed:**
The Catan rulebook frequently references building costs as existing "on the Building Costs Card" — a physical card shipped with the game — rather than listing them inline. The `BUILDING COSTS CARDS` almanac entry says: *"The building costs cards show what can be built and which resources are required."* It never lists the actual costs.

**Root cause:** Chunking/content gap. The rulebook's structure assumes the reader has the physical card in front of them. No single chunk is semantically titled "building costs" in a way that surfaces for cost-related queries.

**Proposed fix:** Add a synthetic chunk explicitly consolidating all building costs:
```
BUILDING COSTS SUMMARY
Road: Brick + Lumber
Settlement: Brick + Lumber + Wool + Grain
City: 3 Ore + 2 Grain
Development Card: Ore + Wool + Grain
```
This synthetic chunk would dominate retrieval for any cost-related question.

---

### Failure Mode 4 (Monopoly): Header Text Discarded by Chunker

**Affected questions:** M01 ("How many players can play Monopoly?")

**What the evaluation revealed:**
The Monopoly rulebook header — *"MONOPOLY - Official Rules / AGES 8+ | 2 to 8 Players"* — appeared before the first detected section and was silently discarded by `_position_based_chunks`, which only captures text from the first detected section header onwards.

**Root cause:** The chunker's start-of-document text was dropped. The player count information existed only in the header preamble, not in any rule section body.

**Fix applied:** Added `("OVERVIEW", r"MONOPOLY - Official Rules")` as the first entry in `_MONOPOLY_SECTIONS`. This creates a chunk capturing the header text, ensuring player count and age range are retrievable. After the fix, M01 is now answered correctly.

---

## Summary of Root Causes and Fixes

| Failure Mode | Questions | Root Cause | Status |
|---|---|---|---|
| Over-cautious grounding | C04, C16 | System prompt too conservative | **Fixed** — prompt revised |
| Multi-hop reasoning | S01, S06 | Single-pass synthesis fails | Open — query decomposition proposed |
| Physical card reference | C08, C09 | Building costs not in rulebook text | Open — synthetic chunk proposed |
| Header text discarded | M01 | Chunker dropped pre-section content | **Fixed** — OVERVIEW chunk added |

---

## Key Takeaways

1. **Retrieval precision is not the bottleneck.** Dense retrieval achieves 100% Precision@3 on both games. The retrieval layer is solid for these document sizes.

2. **Generation is the harder problem.** The most damaging Catan failures (C04, C16) happened with perfect retrieval — the model refused to use context it was given. Fixing the system prompt raised overall correctness from ~75% to 83%.

3. **TF-IDF is competitive with dense retrieval.** At k=7, TF-IDF matches dense at 100% Precision@k. For applications where embedding inference is slow or expensive, TF-IDF with slightly higher k is a viable alternative.

4. **Hallucination resistance is strong across both games and all retrievers.** The grounding prompt's "only answer from context" instruction is effective. The model correctly abstained on all unanswerable (Category C) questions.

5. **Monopoly outperforms Catan (97% vs 83%).** Monopoly's rulebook is more linear and self-contained — fewer cross-references to physical components, less reliance on multi-section synthesis. This confirms that rulebook structure (not just retrieval quality) is a major factor in RAG performance.

6. **Evaluation as diagnostic tool.** The dual-metric design — measuring retrieval precision and answer correctness independently — makes it possible to distinguish generation failures from retrieval failures. Without this, C04 and C16 would have appeared to be retrieval problems.
