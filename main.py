"""
main.py — Streamlit UI for the Board Game RAG system.

Run (from project root):
    streamlit run main.py
"""

import base64
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Board Game Rules AI",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load images ───────────────────────────────────────────────────────────────
CATAN_IMG = img_to_base64("games/Catan_game.png")
MONOPOLY_IMG = img_to_base64("games/Monopoly_game.png")
GAME_IMAGES = {"catan": CATAN_IMG, "monopoly": MONOPOLY_IMG}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@400;500;600;700&display=swap');

    #MainMenu, header, footer {visibility: hidden;}

    .stApp { background-color: #e4dfd7; }

    /* ── Dark hero ── */
    .hero {
        background-color: #2b3040;
        margin: -6rem -4rem 0 -4rem;
        padding: 5rem 2rem 4rem 2rem;
        text-align: center;
    }
    .hero-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1100px;
        margin: 0 auto 3rem auto;
        padding: 0 1rem;
    }
    .hero-logo {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 400;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    .hero-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    .hero-links a {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #a0a0a0;
        letter-spacing: 0.3px;
        text-decoration: none;
        transition: color 0.2s;
    }
    .hero-links a:hover { color: #ffffff; }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 400;
        color: #ffffff;
        max-width: 700px;
        margin: 0 auto 1rem auto;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #9a9a9a;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Smaller hero for game page ── */
    .hero-sm {
        background-color: #2b3040;
        margin: -6rem -4rem 0 -4rem;
        padding: 5rem 2rem 2.5rem 2rem;
    }
    .hero-sm-inner {
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-sm-left {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .hero-sm-img {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid rgba(255,255,255,0.15);
    }
    .hero-sm-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 400;
        color: #ffffff;
    }
    .hero-sm-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 300;
        color: #9a9a9a;
        margin-top: 0.1rem;
    }

    /* ── Section ── */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 400;
        color: #2b3040;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: -0.3px;
    }

    /* ── Game cards ── */
    .game-grid {
        display: flex;
        justify-content: center;
        gap: 5rem;
        margin-bottom: 1rem;
    }
    .game-item {
        text-align: center;
        max-width: 260px;
    }
    .game-circle {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        overflow: hidden;
        margin: 0 auto 1.2rem auto;
        border: 3px solid transparent;
        transition: border-color 0.2s ease;
    }
    .game-circle:hover { border-color: #2b3040; }
    .game-circle img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .game-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 500;
        color: #2b3040;
        margin-bottom: 0.3rem;
    }
    .game-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 300;
        color: #7a7568;
        line-height: 1.5;
    }

    /* ── Thin divider ── */
    .thin-divider {
        border: none;
        border-top: 1px solid #c8c3ba;
        margin: 3rem 0;
    }

    /* ── Rulebook previews on home ── */
    .rulebook-preview {
        display: flex;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    .rulebook-preview-card {
        flex: 1;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        background: transparent;
    }
    .rulebook-preview-title {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #2b3040;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .rulebook-preview-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #7a7568;
        line-height: 1.6;
    }

    /* ── Eval summary cards on home ── */
    .eval-summary {
        display: flex;
        gap: 2rem;
    }
    .eval-summary-card {
        flex: 1;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        padding: 1.5rem;
        text-align: center;
    }
    .eval-summary-game {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7a7568;
        margin-bottom: 0.3rem;
    }
    .eval-summary-val {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: #2b3040;
    }
    .eval-summary-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #9a9590;
        margin-top: 0.1rem;
    }

    /* ── Retriever cards ── */
    .retriever-item {
        text-align: center;
        padding: 1.5rem 2rem;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        background: transparent;
        transition: border-color 0.2s ease;
    }
    .retriever-item:hover { border-color: #2b3040; }
    .retriever-item.active {
        border-color: #2b3040;
        background-color: rgba(43, 48, 64, 0.04);
    }
    .retriever-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7a7568;
        margin-bottom: 0.4rem;
    }
    .retriever-name {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: #2b3040;
    }

    /* ── Text input ── */
    .stTextArea textarea {
        background-color: #ebe7e0 !important;
        color: #2b3040 !important;
        border: 1px solid #c8c3ba !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #2b3040 !important;
        box-shadow: none !important;
    }
    .stTextArea textarea::placeholder { color: #9a9590 !important; }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        border-radius: 4px !important;
        font-weight: 400 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #2b3040 !important;
        color: #ffffff !important;
        border: 1px solid #2b3040 !important;
        padding: 0.5rem 2rem !important;
    }
    .stButton > button[kind="primary"]:hover { background-color: #3b4050 !important; }
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #2b3040 !important;
        border: 1px solid #c8c3ba !important;
        padding: 0.5rem 1.5rem !important;
    }
    .stButton > button[kind="secondary"]:hover { border-color: #2b3040 !important; }

    /* ── Answer ── */
    .answer-card {
        background-color: #ebe7e0;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #2b3040;
    }

    /* ── Chunk ── */
    .chunk-card {
        background: transparent;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .chunk-title {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #2b3040;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .chunk-score {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #9a9590;
    }
    .chunk-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #5a5550;
        line-height: 1.6;
        margin-top: 0.6rem;
        white-space: pre-wrap;
    }

    /* ── Metrics ── */
    .metric-box {
        text-align: center;
        padding: 1.2rem 1rem;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 600;
        color: #2b3040;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 400;
        color: #9a9590;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.2rem;
    }

    /* ── Rulebook box ── */
    .rulebook-box {
        background-color: #ebe7e0;
        border: 1px solid #c8c3ba;
        border-radius: 4px;
        padding: 2rem;
        max-height: 500px;
        overflow-y: auto;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        line-height: 1.7;
        color: #5a5550;
        white-space: pre-wrap;
    }

    /* ── Result row ── */
    .result-row {
        display: flex;
        align-items: center;
        padding: 0.55rem 1rem;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid #d5d0c8;
        font-family: 'Inter', sans-serif;
    }
    .result-icon { font-size: 0.9rem; margin-right: 0.7rem; min-width: 1rem; text-align: center; }
    .result-cat {
        font-size: 0.65rem; font-weight: 500; min-width: 2.2rem;
        text-transform: uppercase; color: #9a9590; letter-spacing: 0.5px;
    }
    .result-q { font-size: 0.82rem; color: #2b3040; flex: 1; margin-left: 0.5rem; }
    .result-prec { font-size: 0.72rem; color: #9a9590; min-width: 4rem; text-align: right; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #c8c3ba;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #9a9590;
        padding: 0.8rem 1.5rem;
        background: transparent;
    }
    .stTabs [aria-selected="true"] { color: #2b3040 !important; font-weight: 500; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #2b3040 !important; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        color: #2b3040 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #ebe7e0 !important;
        color: #2b3040 !important;
        border-color: #c8c3ba !important;
        border-radius: 4px !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #9a9590;
        border-top: 1px solid #c8c3ba;
        margin-top: 3rem;
    }

    .small-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #9a9590;
        margin-bottom: 0.4rem;
    }

    .stAlert { border-radius: 4px !important; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #e4dfd7; }
    ::-webkit-scrollbar-thumb { background: #c8c3ba; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Game config ───────────────────────────────────────────────────────────────
GAMES = {
    "catan": {
        "name": "Catan",
        "description": "Trade, build, and settle the island",
        "rulebook": "data/raw/catan_rulebook.txt",
    },
    "monopoly": {
        "name": "Monopoly",
        "description": "Buy, sell, and scheme your way to victory",
        "rulebook": "data/raw/monopoly_rulebook.txt",
    },
}

EXAMPLES: dict[str, list[str]] = {
    "catan": [
        "What resources do you need to build a settlement?",
        "What happens when you roll a 7?",
        "Can players trade when it's not their turn?",
        "Can I play a knight card before rolling?",
        "How does Longest Road work?",
        "What are development cards?",
    ],
    "monopoly": [
        "How much money does each player start with?",
        "What happens when you pass GO?",
        "How do you get out of jail?",
        "What does Free Parking do?",
        "Can you collect rent while in jail?",
        "Can players lend money to each other?",
    ],
}

RETRIEVERS = {
    "dense":  {"name": "Dense Embeddings", "label": "Deep Learning"},
    "tfidf":  {"name": "TF-IDF",           "label": "Classical ML"},
    "random": {"name": "Random Baseline",   "label": "Naive Baseline"},
}


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and index...")
def load_pipeline(game: str):
    from model import retrieve, generate, _load
    _load(game)
    return retrieve, generate


@st.cache_data(show_spinner=False)
def load_eval_results(game: str, retriever: str):
    path = (
        f"data/outputs/{game}_results.csv"
        if retriever == "dense"
        else f"data/outputs/{game}_{retriever}_results.csv"
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


@st.cache_data(show_spinner=False)
def load_chunks(game: str):
    path = f"models/{game}/chunks.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_rulebook(game: str):
    path = GAMES[game]["rulebook"]
    if not os.path.exists(path):
        return "Rulebook not found."
    with open(path) as f:
        return f.read()


def compute_eval_metrics(results):
    """Compute summary metrics from eval results."""
    if not results:
        return None
    correctness = [r for r in results if r["category"] == "correctness"]
    stress = [r for r in results if r["category"] == "stress"]
    stress_c = [r for r in results if r.get("sub_category") == "C"]

    def _avg(group, field):
        vals = [float(r[field]) for r in group if r.get(field, "") != ""]
        return sum(vals) / len(vals) if vals else 0

    hall = (sum(1 for r in stress_c if r.get("hallucination_check") == "POTENTIAL_HALLUCINATION")
            / len(stress_c) * 100) if stress_c else 0

    return {
        "precision": _avg(results, "precision_at_k"),
        "correctness": _avg(correctness, "correctness_score"),
        "stress": _avg(stress, "correctness_score"),
        "hallucination": hall,
    }


# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_game" not in st.session_state:
    st.session_state.selected_game = None
if "selected_retriever" not in st.session_state:
    st.session_state.selected_retriever = "dense"
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <div class="hero-nav">
            <div class="hero-logo">Board Game Rules AI</div>
            <div class="hero-links">
                <a href="#games">Games</a>
                <a href="#rulebooks">Rulebooks</a>
                <a href="#evaluation">Evaluation</a>
            </div>
        </div>
        <h1>Know the Rules,<br>Win the Game</h1>
        <p>Ask any question about your favorite board games.
           Powered by retrieval-augmented generation.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Games section ──
    st.markdown("")
    st.markdown("")
    st.markdown('<div class="section-title" id="games">Choose Your Game</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="game-grid">
        <div class="game-item">
            <div class="game-circle">
                <img src="data:image/png;base64,{CATAN_IMG}" alt="Catan">
            </div>
            <div class="game-name">Catan</div>
            <div class="game-desc">Trade, build, and settle the island</div>
        </div>
        <div class="game-item">
            <div class="game-circle">
                <img src="data:image/png;base64,{MONOPOLY_IMG}" alt="Monopoly">
            </div>
            <div class="game-name">Monopoly</div>
            <div class="game-desc">Buy, sell, and scheme your way to victory</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, bcol1, _, bcol2, _ = st.columns([1.5, 1, 0.5, 1, 1.5])
    with bcol1:
        if st.button("Play Catan", use_container_width=True, type="secondary"):
            st.session_state.page = "game"
            st.session_state.selected_game = "catan"
            st.session_state.question = ""
            st.session_state.last_answer = None
            st.session_state.last_retrieved = None
            st.rerun()
    with bcol2:
        if st.button("Play Monopoly", use_container_width=True, type="secondary"):
            st.session_state.page = "game"
            st.session_state.selected_game = "monopoly"
            st.session_state.question = ""
            st.session_state.last_answer = None
            st.session_state.last_retrieved = None
            st.rerun()

    # ── Rulebooks section ──
    st.markdown('<hr class="thin-divider" id="rulebooks">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rulebooks</div>', unsafe_allow_html=True)

    rb_col1, rb_col2 = st.columns(2, gap="medium")
    game_details = {
        "catan": {"sections": "Game Rules & Almanac", "designer": "Klaus Teuber", "year": "2020"},
        "monopoly": {"sections": "21 Rule Sections", "designer": "Parker Brothers", "year": "2007"},
    }
    for col, (gk, gi) in zip([rb_col1, rb_col2], GAMES.items()):
        with col:
            chunks = load_chunks(gk)
            chunk_count = len(chunks) if chunks else "—"
            details = game_details[gk]
            st.markdown(f"""
            <div class="rulebook-preview-card">
                <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                    <img src="data:image/png;base64,{GAME_IMAGES[gk]}"
                         style="width:60px; height:60px; border-radius:50%; object-fit:cover;
                                border:1px solid #c8c3ba;">
                    <div>
                        <div class="rulebook-preview-title" style="margin-bottom:0;">{gi['name']}</div>
                        <div style="font-family:'Inter',sans-serif; font-size:0.72rem; color:#9a9590;">
                            {details['designer']} &middot; {details['year']}</div>
                    </div>
                </div>
                <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#5a5550; line-height:1.6;">
                    {gi['description']}. Full official rulebook parsed into
                    <strong>{chunk_count}</strong> semantic chunks for intelligent retrieval.
                </div>
                <div style="display:flex; gap:1.5rem; margin-top:1rem;">
                    <div style="font-family:'Inter',sans-serif; font-size:0.72rem; color:#9a9590;">
                        <strong style="color:#2b3040;">{chunk_count}</strong> Chunks</div>
                    <div style="font-family:'Inter',sans-serif; font-size:0.72rem; color:#9a9590;">
                        <strong style="color:#2b3040;">{details['sections']}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Evaluation section ──
    st.markdown('<hr class="thin-divider" id="evaluation">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluation Overview</div>', unsafe_allow_html=True)

    # Best result (dense) for each game
    ev_cols = st.columns(2, gap="large")
    for col, (gk, gi) in zip(ev_cols, GAMES.items()):
        with col:
            results = load_eval_results(gk, "dense")
            metrics = compute_eval_metrics(results)
            if metrics:
                st.markdown(f"""
                <div class="eval-summary-card">
                    <div class="eval-summary-game">{gi['name']} — Dense Retrieval</div>
                    <div style="display:flex; justify-content:space-around; margin-top:1rem;">
                        <div>
                            <div class="eval-summary-val">{metrics['precision']:.0%}</div>
                            <div class="eval-summary-label">Retrieval P@3</div>
                        </div>
                        <div>
                            <div class="eval-summary-val">{metrics['correctness']:.0%}</div>
                            <div class="eval-summary-label">Correctness</div>
                        </div>
                        <div>
                            <div class="eval-summary-val">{metrics['hallucination']:.0f}%</div>
                            <div class="eval-summary-label">Hallucination</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="eval-summary-card">
                    <div class="eval-summary-game">{gi['name']}</div>
                    <div class="eval-summary-val">—</div>
                    <div class="eval-summary-label">No evaluation data</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        Built with sentence-transformers &middot; FAISS &middot; OpenAI &middot; Streamlit<br>
        AIPI 540 &mdash; Natural Language Processing
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GAME
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "game":

    game = st.session_state.selected_game
    game_info = GAMES[game]
    game_img = GAME_IMAGES[game]

    # ── Compact hero with game info ──
    st.markdown(f"""
    <div class="hero-sm">
        <div class="hero-sm-inner">
            <div class="hero-sm-left">
                <img class="hero-sm-img" src="data:image/png;base64,{game_img}" alt="{game_info['name']}">
                <div>
                    <div class="hero-sm-title">{game_info['name']} Rules Assistant</div>
                    <div class="hero-sm-sub">{game_info['description']}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Back button
    st.markdown("")
    if st.button("← Back to Games", type="secondary"):
        st.session_state.page = "home"
        st.session_state.selected_game = None
        st.session_state.question = ""
        st.session_state.last_answer = None
        st.session_state.last_retrieved = None
        st.rerun()

    st.markdown("")

    # ── Tabs ──
    tab_ask, tab_rules, tab_eval = st.tabs(["Ask a Question", "Rulebook", "Evaluation"])

    # ── TAB 1: Ask ──
    with tab_ask:
        st.markdown('<div class="section-title" style="font-size:1.4rem; margin-top:1rem;">Retrieval Strategy</div>',
                    unsafe_allow_html=True)

        ret_cols = st.columns(3, gap="medium")
        for j, (rk, ri) in enumerate(RETRIEVERS.items()):
            with ret_cols[j]:
                is_active = st.session_state.selected_retriever == rk
                st.markdown(f"""
                <div class="retriever-item {'active' if is_active else ''}">
                    <div class="retriever-label">{ri['label']}</div>
                    <div class="retriever-name">{ri['name']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(
                    "Selected" if is_active else "Use",
                    key=f"ret_{rk}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.selected_retriever = rk
                    st.rerun()

        retriever = st.session_state.selected_retriever

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        # Examples
        st.markdown('<p class="small-label">Try an example</p>', unsafe_allow_html=True)
        ex_cols = st.columns(3, gap="small")
        clicked = None
        for i, ex in enumerate(EXAMPLES[game]):
            with ex_cols[i % 3]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    clicked = ex
        if clicked:
            st.session_state.question = clicked

        st.markdown("")
        question = st.text_area(
            "question",
            value=st.session_state.question,
            height=90,
            placeholder=f"Ask anything about {game_info['name']} rules...",
            label_visibility="collapsed",
        )

        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            ask = st.button("Ask", type="primary", use_container_width=True)

        if ask and question.strip():
            st.session_state.question = question
            retrieve_fn, generate_fn = load_pipeline(game)
            with st.spinner("Searching rulebook..."):
                retrieved = retrieve_fn(question, game=game, k=3, retriever=retriever)
            with st.spinner("Generating answer..."):
                answer = generate_fn(question, retrieved, game=game)
            st.session_state.last_answer = answer
            st.session_state.last_retrieved = retrieved
        elif ask:
            st.warning("Please enter a question.")

        # Display answer
        if st.session_state.last_answer:
            answer = st.session_state.last_answer
            retrieved = st.session_state.last_retrieved

            st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:1.4rem;">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

            ABSTAIN = [
                "don't specify", "doesn't specify", "not specify",
                "rules don't", "rules do not", "not addressed",
                "not mentioned", "not explicit", "not state",
                "silent", "i don't know", "cannot answer",
                "unclear", "no rule", "not covered",
            ]
            if not any(p in answer.lower() for p in ABSTAIN) and retrieved:
                st.markdown(
                    '<div class="section-title" style="font-size:1.2rem; margin-top:2rem;">Retrieved Sections</div>',
                    unsafe_allow_html=True,
                )
                chunks_data = load_chunks(game)
                for chunk in retrieved:
                    score = chunk["retrieval_score"]
                    title = chunk["title"]
                    idx = chunk["chunk_idx"]
                    text = chunks_data[idx]["text"] if idx < len(chunks_data) else chunk.get("text", "")
                    score_str = f"Similarity: {score:.3f}" if score > 0 else "Random selection"
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-title">{title}</div>
                        <div class="chunk-score">{score_str}</div>
                        <div class="chunk-text">{text[:600]}{'...' if len(text) > 600 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 2: Rulebook ──
    with tab_rules:
        st.markdown(
            f'<div class="section-title" style="font-size:1.4rem; margin-top:1rem;">'
            f'{game_info["name"]} Official Rulebook</div>',
            unsafe_allow_html=True,
        )
        rulebook = load_rulebook(game)
        st.markdown(f'<div class="rulebook-box">{rulebook}</div>', unsafe_allow_html=True)

        chunks_data = load_chunks(game)
        if chunks_data:
            st.markdown(
                f'<div class="section-title" style="font-size:1.2rem; margin-top:2rem;">'
                f'Indexed Chunks ({len(chunks_data)})</div>',
                unsafe_allow_html=True,
            )
            for i, c in enumerate(chunks_data):
                with st.expander(f"{i+1}. {c['title']}"):
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:#5a5550; white-space:pre-wrap; "
                        f"line-height:1.6; font-family:Inter,sans-serif;'>{c['text']}</div>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 3: Evaluation ──
    with tab_eval:
        st.markdown('<div class="section-title" style="font-size:1.4rem; margin-top:1rem;">Performance</div>',
                    unsafe_allow_html=True)

        m_cols = st.columns(3, gap="medium")
        for j, (rk, ri) in enumerate(RETRIEVERS.items()):
            results = load_eval_results(game, rk)
            metrics = compute_eval_metrics(results)
            with m_cols[j]:
                st.markdown(f"""
                <div style="text-align:center; margin-bottom:0.8rem;">
                    <div class="retriever-label">{ri['label']}</div>
                    <div style="font-family:'Inter',sans-serif; font-weight:500; color:#2b3040; font-size:1rem;">
                        {ri['name']}</div>
                </div>
                """, unsafe_allow_html=True)

                if metrics:
                    for val, label in [
                        (f"{metrics['precision']:.0%}", "Retrieval P@3"),
                        (f"{metrics['correctness']:.0%}", "Correctness"),
                        (f"{metrics['stress']:.0%}", "Stress Test"),
                        (f"{metrics['hallucination']:.0f}%", "Hallucination"),
                    ]:
                        st.markdown(f"""
                        <div class="metric-box" style="margin-bottom:0.5rem;">
                            <div class="metric-value">{val}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-box"><div class="metric-value">--</div>
                    <div class="metric-label">No data</div></div>
                    """, unsafe_allow_html=True)

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.2rem;">Detailed Results</div>',
                    unsafe_allow_html=True)

        detail_retriever = st.selectbox(
            "Retriever",
            options=["dense", "tfidf", "random"],
            format_func=lambda r: RETRIEVERS[r]["name"],
            key="eval_detail",
        )
        detail_results = load_eval_results(game, detail_retriever)
        if detail_results:
            for r in detail_results:
                score = r.get("correctness_score", "")
                if score == "1":
                    icon, color = "&#10003;", "#48964b"
                elif score == "0.5":
                    icon, color = "&#9679;", "#c49a2a"
                elif score == "0":
                    icon, color = "&#10007;", "#c44a3f"
                else:
                    icon, color = "&#9675;", "#9a9590"

                cat = r.get("sub_category", r["category"]).upper()
                prec = r.get("precision_at_k", "")
                st.markdown(f"""
                <div class="result-row">
                    <span class="result-icon" style="color:{color};">{icon}</span>
                    <span class="result-cat">{cat}</span>
                    <span class="result-q">{r['question'][:85]}{'...' if len(r['question']) > 85 else ''}</span>
                    <span class="result-prec">P@3: {prec}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No evaluation results available. Run the evaluation script first.")

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        Built with sentence-transformers &middot; FAISS &middot; OpenAI &middot; Streamlit<br>
        AIPI 540 &mdash; Natural Language Processing
    </div>
    """, unsafe_allow_html=True)
