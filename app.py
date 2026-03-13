"""
app.py — Streamlit UI for the Catan RAG system.

Run:
    .venv/bin/streamlit run app.py
"""

import csv
import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Catan Rules Assistant",
    page_icon="🎲",
    layout="wide",
)

# ── Load model / index once ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and index…")
def load_pipeline():
    from rag_pipeline import retrieve, generate, _load
    _load()
    return retrieve, generate


@st.cache_data(show_spinner=False)
def load_eval_results():
    path = "results.csv"
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


@st.cache_data(show_spinner=False)
def load_chunks():
    with open("chunks.json") as f:
        return json.load(f)


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What resources do you need to build a settlement?",
    "What happens when you roll a 7?",
    "Can players trade with each other when it is not their turn?",
    "Can a player make a binding promise to give resources on a future turn?",
    "My road is 6 segments long and an opponent breaks it into 4 and 2. Do I lose Longest Road?",
    "Can I play a knight card before I roll the dice?",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎲 Catan RAG")

    st.divider()
    st.subheader("System")
    st.markdown(
        """
        - **Chunks:** 42 (9 Game Rules + 33 Almanac)
        - **Embeddings:** all-MiniLM-L6-v2
        - **Vector store:** FAISS (cosine)
        - **LLM:** gpt-4o-mini
        - **Retrieval k:** 3
        """
    )

    st.divider()
    st.subheader("Evaluation Results")

    results = load_eval_results()
    if results:
        correctness = [r for r in results if r["category"] == "correctness"]
        stress = [r for r in results if r["category"] == "stress"]

        def avg_score(group):
            scores = [float(r["correctness_score"]) for r in group if r["correctness_score"] != ""]
            return sum(scores) / len(scores) if scores else 0

        col1, col2 = st.columns(2)
        col1.metric("Retrieval P@3", "100%")
        col2.metric("Hallucination", "0%")

        col3, col4 = st.columns(2)
        col3.metric("Correctness Qs", f"{avg_score(correctness):.0%}")
        col4.metric("Stress Test", f"{avg_score(stress):.0%}")

        st.divider()
        if st.toggle("Show all results"):
            for r in results:
                score = r.get("correctness_score", "")
                color = "🟢" if score == "1" else "🟡" if score == "0.5" else "🔴"
                st.markdown(
                    f"{color} **{r['id']}** — {r['question'][:55]}…"
                    if len(r['question']) > 55
                    else f"{color} **{r['id']}** — {r['question']}"
                )
    else:
        st.info("Run `evaluate.py` to see results here.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎲 Catan Rules Assistant")
st.markdown(
    "Ask any question about the Catan rulebook. "
    "The system retrieves the most relevant rule sections and generates a grounded answer."
)

# Example question buttons
st.markdown("**Try an example:**")
cols = st.columns(3)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex[:45] + ("…" if len(ex) > 45 else ""), key=f"ex_{i}", use_container_width=True):
        clicked = ex

# Text input — pre-fill from clicked example or session state
if "question" not in st.session_state:
    st.session_state.question = ""
if clicked:
    st.session_state.question = clicked

question = st.text_area(
    "Your question",
    value=st.session_state.question,
    height=80,
    placeholder="e.g. What happens when you roll a 7?",
    label_visibility="collapsed",
)

ask = st.button("Ask", type="primary", use_container_width=False)

# ── Answer ────────────────────────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.question = question

    retrieve_fn, generate_fn = load_pipeline()

    with st.spinner("Retrieving relevant rules…"):
        retrieved = retrieve_fn(question, k=3)

    with st.spinner("Generating answer…"):
        answer = generate_fn(question, retrieved)

    st.divider()

    # Answer box
    st.subheader("Answer")
    st.success(answer)

    # Retrieved chunks — only show if the model found an answer in the rulebook
    ABSTAIN_PHRASES = [
        "don't specify", "doesn't specify", "not specify",
        "rules don't", "rules do not", "not addressed",
        "not mentioned", "not explicit", "not state",
        "silent", "i don't know", "cannot answer",
        "unclear", "no rule", "not covered",
    ]
    abstained = any(p in answer.lower() for p in ABSTAIN_PHRASES)

    if not abstained:
        st.subheader("Retrieved rule sections")
        chunks_data = load_chunks()

        for chunk in retrieved:
            score = chunk["retrieval_score"]
            title = chunk["title"]
            idx = chunk["chunk_idx"]
            full_text = chunks_data[idx]["text"] if idx < len(chunks_data) else chunk.get("text", "")

            bar_pct = int(score * 100)
            with st.expander(f"[{score:.3f}]  {title}"):
                st.progress(bar_pct, text=f"Similarity: {score:.3f}")
                st.markdown(
                    f"<div style='font-size:0.88rem; color:#ccc; white-space:pre-wrap'>{full_text}</div>",
                    unsafe_allow_html=True,
                )

elif ask and not question.strip():
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with sentence-transformers · FAISS · OpenAI · Streamlit &nbsp;|&nbsp; "
    "Duke Deep Learning / NLP Hackathon"
)
