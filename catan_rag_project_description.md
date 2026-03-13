# Board Game Rules Assistant: A RAG System for Catan

## Project Overview

This project builds a Retrieval-Augmented Generation (RAG) system that answers natural language questions about the board game Catan using the official rulebook as its knowledge base. The system retrieves relevant rule sections from the rulebook and uses a large language model to generate grounded answers. The core goal is not just to build a working prototype, but to rigorously evaluate it — demonstrating how two complementary evaluation strategies (correctness testing and adversarial stress testing) reveal specific failure modes and guide concrete improvements.

## Motivation

Board game rulebooks are an ideal test bed for RAG evaluation because they are self-contained (the answer is always in the document), rules frequently interact with each other (requiring multi-hop reasoning across sections), and players naturally ask ambiguous or edge-case questions that stress retrieval systems. Catan specifically is a strong choice because its rules are dense with cross-references (the robber interacts with rolling a 7, knight cards, and resource production simultaneously), it has well-known community FAQs and edge cases, and the official rulebook is freely available from the publisher.

## System Architecture

### Data Preparation (Chunking)
The official Catan rulebook (~4,000 words of core rules + ~6,000 words of almanac) is split into semantically meaningful chunks. Rather than naive fixed-size splitting, the chunking strategy uses the rulebook's own structure: each Almanac entry (e.g., ROBBER, LONGEST ROAD, TRADE, SETTLEMENTS) becomes one chunk, and each numbered section of the Game Rules becomes a chunk. This produces approximately 25-35 chunks of varying length. Each chunk is tagged with a section title for traceability.

### Embedding and Indexing
Each chunk is embedded using a sentence embedding model (e.g., `all-MiniLM-L6-v2` from sentence-transformers or OpenAI's `text-embedding-3-small`) and stored in a lightweight vector store (FAISS or ChromaDB). No database infrastructure is needed — the entire index fits in memory.

### Retrieval
Given a user query, the system embeds the query, retrieves the top-k most similar chunks (k=3 by default), and passes them as context to the language model.

### Generation
A prompted LLM (e.g., Claude, GPT-4, or a smaller model via API) receives the retrieved chunks along with a system prompt instructing it to answer only based on the provided context, cite which section its answer comes from, and say "I don't know" or "the rules don't specify" when the context is insufficient.

## Evaluation Plan

The evaluation plan uses two complementary strategies that together reveal different categories of system failure.

### Evaluation Strategy 1: Correctness on Curated Rules Questions

**What it measures:** Whether the system gives factually correct, complete answers to straightforward rules questions where a single, unambiguous answer exists in the rulebook.

**Method:** Create a curated test set of 15-20 question-answer pairs, manually authored by reading the rulebook. Each pair includes the question, the expected correct answer, and which rulebook section contains the answer (for evaluating retrieval).

**Example questions:**
- "What resources do you need to build a settlement?" → Brick, Lumber, Wool, and Grain
- "How many victory points do you need to win?" → 10
- "What happens when you roll a 7?" → No one gets resources; anyone with 8+ cards discards half; you move the robber and steal a card
- "Can you build a city without first having a settlement there?" → No, you can only upgrade an existing settlement
- "How many knight cards do you need for Largest Army?" → 3
- "Can you play a development card you bought this turn?" → No (exception: victory point cards if they win you the game)

**Metrics:**
- **Answer Correctness (scored 0 / 0.5 / 1):** Manually scored as correct, partially correct, or incorrect by comparing to the gold answer. Partial credit for answers that are right but incomplete (e.g., correctly listing some but not all resources).
- **Retrieval Precision@k:** For each question, check whether the chunk containing the gold answer appears in the top-k retrieved chunks. Report as a percentage across the full test set.

**What this reveals:** Baseline system competence. If the system struggles here, the problem is likely in chunking or embedding quality. This evaluation establishes whether the retrieval pipeline is surfacing the right sections and whether the LLM is faithfully reading them.

### Evaluation Strategy 2: Stress Test with Ambiguous Multi-Rule Interactions

**What it measures:** Whether the system handles questions that require synthesizing information from multiple rulebook sections, resolving edge cases, or correctly refusing to answer when rules are ambiguous or silent.

**Method:** Create a stress test set of 8-12 adversarial or tricky queries. These fall into three categories:

**Category A — Multi-section synthesis (requires retrieving and combining 2+ chunks):**
- "If I roll a 7 and have 9 resource cards, but I also want to play a knight card this turn, what should I do first?" (Requires combining: rolling a 7 rules + development card timing rules)
- "I just built a settlement on a harbor. Can I use that harbor's trade rate immediately?" (Requires: Combined Trade/Build Phase rules + Harbor rules)
- "My road is 6 segments long, but an opponent just built a settlement in the middle of it. Do I lose Longest Road?" (Requires: Longest Road rules + settlement building rules + the road-breaking mechanic)

**Category B — Edge cases the rules explicitly address but that are easy to miss:**
- "What happens if the bank runs out of a resource that multiple players are owed?" (Answer: no one gets that resource)
- "Can I trade 2 wool for 1 wool with another player?" (Answer: no, you cannot trade like resources)
- "If two players tie for longest road after the leader's road is broken, who gets the card?" (Answer: neither — set the card aside)

**Category C — Questions the rules don't answer (testing hallucination resistance):**
- "Can I negotiate future-turn promises in trades?" (Rules are silent on binding promises)
- "What happens if all development cards run out mid-game?" (Rules say you can't buy more, but don't elaborate)
- "Can I look at which development cards have been played by other players?" (Rules don't explicitly state this)

**Metrics:**
- **Answer Correctness (same 0 / 0.5 / 1 scale):** Scored with extra attention to whether multi-section answers are complete and whether the model correctly identifies when rules are silent.
- **Retrieval Coverage:** For multi-section questions, check whether all relevant chunks were retrieved (not just one). Report the percentage of required chunks that appeared in top-k.
- **Hallucination Rate:** Percentage of Category C questions where the model fabricates a definitive rule instead of acknowledging the rules don't specify.

**What this reveals:** The limits of single-query retrieval. If correctness drops sharply on multi-section questions versus single-section questions, it indicates the retrieval window (top-k) is too narrow or the embeddings don't capture cross-reference relationships. High hallucination rates on Category C questions indicate the generation prompt needs better grounding instructions.

## Expected Findings and Improvement Cycle

Based on common RAG failure patterns, the evaluation is expected to reveal findings like:

1. **Retrieval misses on multi-hop questions.** When a question spans ROBBER + DEVELOPMENT CARDS, the system may retrieve the robber chunk but miss the development card timing chunk. This suggests improvements such as increasing k, adding chunk overlap, or implementing query decomposition (breaking one question into sub-queries).

2. **Hallucination on unanswerable questions.** The LLM may confidently answer Category C questions by generating plausible-sounding but unsupported rules. This suggests tightening the system prompt with explicit instructions to flag unsupported claims, or adding a post-processing step that checks whether the answer can be traced to retrieved text.

3. **Chunking boundary problems.** Some rules span the boundary between the Game Rules section and the Almanac section (e.g., development card timing is discussed in both places). If chunks are too isolated, the system may give incomplete answers. This suggests creating overlapping chunks or adding cross-reference metadata.

4. **Semantic similarity failures.** Questions using casual player language ("Can I steal from someone with no cards?") may not embed close to the formal rulebook text. This suggests augmenting chunks with paraphrased headers or using a hybrid retrieval approach (keyword + semantic).

The project concludes with a concrete analysis: for each failure mode observed, describe what the evaluation revealed, classify the root cause (retrieval failure, generation failure, or chunking failure), and propose a specific fix. This demonstrates the core learning objective — evaluation is not just scoring, it is a diagnostic tool that drives model improvement.

## Technology Stack

| Component | Recommended Tool | Alternative |
|-----------|-----------------|-------------|
| Embedding | sentence-transformers (`all-MiniLM-L6-v2`) | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (in-memory) | ChromaDB |
| LLM | Claude API or OpenAI API | Local model via Ollama |
| Language | Python | — |
| Evaluation | Manual scoring in spreadsheet/CSV | — |

## Deliverables

1. **Working RAG prototype** — a Python script or notebook that accepts a Catan rules question and returns a grounded answer with source citations.
2. **Curated evaluation set** — 15-20 correctness questions + 8-12 stress test questions, all with gold answers and source sections.
3. **Evaluation results** — a table or report showing correctness scores, retrieval precision, and hallucination rates across both test sets.
4. **Error analysis** — a written analysis (1-2 pages) categorizing observed failures, diagnosing root causes, and proposing specific improvements.

## Build Order

| Step | Task |
|------|------|
| 1 | Chunk the rulebook text, generate embeddings, build FAISS index |
| 2 | Wire up the retrieval + LLM generation pipeline with a grounding prompt |
| 3 | Write the curated test set (correctness questions with gold answers) |
| 4 | Run evaluation on both correctness and stress test sets, record results |
| 5 | Analyze errors, categorize failure modes, write up findings |
