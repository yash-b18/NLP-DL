# Error Analysis: Catan RAG System

## System Overview

A Retrieval-Augmented Generation (RAG) system was built to answer natural language questions about the Catan board game rulebook. The pipeline embeds 42 semantic chunks of the official rulebook (9 Game Rules sections + 33 Almanac entries) using `all-MiniLM-L6-v2`, stores them in a FAISS cosine-similarity index, retrieves the top-3 most relevant chunks per query, and passes them to `gpt-4o-mini` with a strict grounding prompt.

The system was evaluated on 20 correctness questions and 12 adversarial stress-test questions across three categories (multi-section synthesis, edge cases, and unanswerable questions).

---

## Quantitative Results

| Category | N | Retrieval Precision@3 | Avg Correctness Score |
|---|---|---|---|
| Correctness questions | 20 | 100% | 0.78 / 1.0 |
| Stress A — multi-section | 3 | 100% | 0.50 / 1.0 |
| Stress B — edge cases | 5 | 100% | 0.90 / 1.0 |
| Stress C — unanswerable | 4 | 100% | 0.88 / 1.0 |
| **Overall** | **32** | **100%** | **0.75 / 1.0** |

**Retrieval Coverage** (fraction of *all* required chunks found, for multi-section questions): 78%

**Hallucination Rate** (Category C questions where model invented a rule): **0%** — the model correctly abstained on all 4 unanswerable questions.

---

## Failure Mode Analysis

### Failure Mode 1: Generation Failure — Over-Cautious Grounding

**Affected questions:** C04, C16 (full failures); C06, S10, S11 (partial)

**What the evaluation revealed:**
The model repeatedly answered "The rules don't specify this" even when the retrieved context contained a clear, direct answer. Two striking examples:

- **C04** ("Can you build a city without first having a settlement?") — The `Almanac: CITIES` chunk was retrieved and contains the sentence: *"You cannot build a city directly. You can only upgrade an existing settlement to a city."* The model still abstained.
- **C16** ("Can players trade when it's not their turn?") — Both `Almanac: DOMESTIC TRADE` and `Almanac: TRADE` were retrieved. DOMESTIC TRADE explicitly states: *"the other players may not trade among themselves."* The model still abstained.

**Root cause:** Generation failure. The grounding prompt instructs the model to say "The rules don't specify this" when context is insufficient, but the model is applying this conservatively even when the answer is present. This appears most when the question is phrased slightly differently from the rulebook's language — the model fails to match the question intent to the retrieved text.

**Proposed fix:** Revise the system prompt to be less conservative. Replace the blanket "say I don't know if context is insufficient" with more specific guidance: *"If the answer can be inferred from the context, state it and cite the source. Only say 'the rules don't specify' if no relevant information is present in any of the retrieved sections."* Alternatively, add a post-processing verification step that checks whether the answer "The rules don't specify" is appropriate given the retrieved chunks.

---

### Failure Mode 2: Retrieval Failure — Right Chunk Not in Top-3

**Affected questions:** C08, C09, S02

**What the evaluation revealed:**
Three questions failed because the chunk containing the answer did not appear in the top-3 retrieved results, even though it exists in the index:

- **C08** ("Resources to upgrade to a city?") — The answer (3 Ore + 2 Grain) is in `Game Rules: BUILD`. Retrieved instead: `CITIES`, `SETTLEMENTS`, `TACTICS`. The CITIES almanac entry describes the mechanic but does not list the cost explicitly — it refers readers to the Building Costs Card. The embedding for "upgrade to a city" is more similar to `CITIES` than to the game rules BUILD section.
- **C09** ("Resources to buy a development card?") — The answer (Ore, Wool, Grain) is in `Game Rules: BUILD`. Retrieved instead: `BUILDING COSTS CARDS`, `DEVELOPMENT CARDS`, `PROGRESS CARDS`. The `BUILDING COSTS CARDS` chunk describes the cards but doesn't list costs in text — the costs exist only on a physical card referenced by the rules.
- **S02** ("Can I use a harbor immediately after building there?") — The answer is in `Almanac: COMBINED TRADE/BUILD PHASE`. This chunk uses the word "harbor" only once and discusses the topic implicitly. The query embedded closer to `MARITIME TRADE` and `HARBORS`.

**Root cause:** Retrieval failure, driven by two sub-causes:
1. **Semantic mismatch**: Queries about costs embed closer to the thing being built (CITIES, DEVELOPMENT CARDS) rather than the general BUILD section that lists all costs.
2. **Sparse mention**: The `COMBINED TRADE/BUILD PHASE` chunk is short and doesn't use the word "harbor" prominently, so it embeds far from harbor-related queries.

**Proposed fix:**
1. **Increase k from 3 to 5.** All three missed chunks appear in top-5 — a modest increase captures them without flooding the context.
2. **Add overlapping chunk metadata.** Annotate each chunk with related topics (e.g., tag the BUILD chunk with "costs, prices, resources required"). Include these tags in the embedded text.
3. **Augment sparse chunks.** Prepend the `COMBINED TRADE/BUILD PHASE` chunk with a richer header: *"Combined Trade/Build Phase (also: using harbors same turn you build there)"* to improve semantic alignment.

---

### Failure Mode 3: Reasoning Failure on Multi-Section Questions

**Affected questions:** S01, S06 (both scored 0.5)

**What the evaluation revealed:**
When questions required synthesizing rules from two sections, the model retrieved the right chunks but produced confused or self-contradictory reasoning:

- **S01** ("Roll 7, want to play knight card — what first?") — The model correctly identified that you must discard 4 cards first, but it conflated the robber move from rolling a 7 with the robber move from the knight card. It described them as a single action rather than two separate robber activations that would occur sequentially.
- **S06** ("Two players tie for Longest Road after mine is broken") — The model opened with an incorrect statement ("you still keep the Longest Road card") before contradicting itself and arriving at the correct conclusion ("set the card aside"). The final answer was right but the reasoning was incoherent.

**Root cause:** Generation failure on multi-hop reasoning. The model receives multiple retrieved chunks simultaneously and attempts to synthesize them, but produces answers that partially reflect each chunk without fully integrating them. This is a known limitation of single-pass generation for questions requiring logical chaining.

**Proposed fix:** Implement **query decomposition**: break complex questions into sub-questions, retrieve and answer each separately, then synthesize. For S01 this would mean: (1) "What happens when you roll a 7?" → retrieve robber rules; (2) "When can you play a development card?" → retrieve dev card rules; then combine. This adds latency but significantly improves multi-hop accuracy.

---

### Failure Mode 4: Content Gap — Information on Physical Cards

**Affected question:** C09 (and partially C08)

**What the evaluation revealed:**
The Catan rulebook frequently references building costs as existing "on the Building Costs Card" — a physical card that ships with the game — rather than listing them inline in the text. The almanac entry for `BUILDING COSTS CARDS` says: *"The building costs cards show what can be built and which resources are required."* It never lists the actual costs. The specific costs (Ore, Wool, Grain for dev cards; 3 Ore + 2 Grain for cities) only appear in the Game Rules section under BUILD, and even there they appear as brief annotations rather than a dedicated "building costs" section.

**Root cause:** Chunking/content failure. The rulebook's structure assumes the reader has the physical card in front of them. No single chunk is semantically titled "building costs" in a way that surfaces for cost-related queries.

**Proposed fix:** Create a **synthetic chunk** explicitly consolidating all building costs:
```
BUILDING COSTS SUMMARY
Road: Brick + Lumber
Settlement: Brick + Lumber + Wool + Grain
City: 3 Ore + 2 Grain
Development Card: Ore + Wool + Grain
```
This synthetic chunk would dominate retrieval for any cost-related question and eliminate this entire failure class. Adding domain-specific synthetic chunks for frequently-queried facts is a practical RAG improvement strategy.

---

## Summary of Root Causes and Fixes

| Failure Mode | Questions | Root Cause | Proposed Fix |
|---|---|---|---|
| Over-cautious grounding | C04, C16, C06 | Generation — prompt too conservative | Loosen "I don't know" criteria in system prompt |
| Chunk not retrieved | C08, C09, S02 | Retrieval — semantic mismatch | Increase k to 5; augment chunk headers |
| Multi-hop reasoning | S01, S06 | Generation — single-pass synthesis fails | Query decomposition |
| Physical card reference | C09 | Chunking — costs not in text | Add synthetic building costs chunk |

---

## Key Takeaways

1. **Retrieval precision was not the bottleneck.** At k=3, at least one correct chunk was retrieved 100% of the time. The system's retrieval layer is solid for this document size.

2. **Generation is the bigger problem.** The most damaging failures (C04, C16) happened with perfect retrieval — the model simply refused to use the context it was given. Evaluation revealed this is not a data problem but a prompting problem.

3. **Hallucination resistance is strong.** All 4 unanswerable questions were correctly identified as unanswerable. The grounding prompt's "say I don't know" instruction works — it just fires too aggressively.

4. **Evaluation as a diagnostic tool.** Without the structured evaluation, the generation failures (C04, C16) would appear identical to retrieval failures. The dual-metric design — measuring both retrieval coverage and answer correctness independently — made it possible to pinpoint the root cause precisely.
