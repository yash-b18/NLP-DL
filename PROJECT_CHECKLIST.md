# Module 2 Project Requirements Checklist

## Modeling Approaches
- [x] Naive baseline (random retrieval)
- [x] Classical ML model (TF-IDF + cosine similarity)
- [x] Deep learning model (sentence-transformers dense embeddings + FAISS)
- [x] All three documented with locations in repo

## Experimentation
- [x] Focused experiment conducted (k-sensitivity analysis, k=1-10)
- [x] Experiment is well-motivated and clearly described
- [x] Results properly interpreted (in `notebooks/analysis.md`)

## Interactive Application
- [ ] **Publicly accessible via the internet (DEPLOYED)**
- [x] Runs model inference only (no training)
- [ ] **Good UX — beyond basic Streamlit** (assignment says "a basic Streamlit app is not going to cut it")
- [x] Streamlit app exists (`main.py`) with game selector, retriever selector, example questions, eval metrics

## Written Report
- [ ] **Formal report (NeurIPS/ICML paper, white paper, or technical report)**
  - [ ] Problem Statement
  - [ ] Data Sources
  - [ ] Related Work
  - [ ] Evaluation Strategy & Metrics (with justification)
  - [ ] Modeling Approach
  - [ ] Data Processing Pipeline (with rationale)
  - [ ] Hyperparameter Tuning Strategy
  - [ ] Models Evaluated (all 3 with rationale)
  - [ ] Results (quantitative comparison, visualizations, confusion matrices)
  - [ ] Error Analysis (5 specific mispredictions with root causes and mitigations)
  - [ ] Experiment Write-Up (plan, results, interpretation, recommendations)
  - [ ] Conclusions
  - [ ] Future Work
  - [ ] Commercial Viability Statement
  - [ ] Ethics Statement

## In-Class Pitch (5 min)
- [ ] **Presentation prepared**
  - [ ] Problem & Motivation
  - [ ] Approach Overview
  - [ ] Live Demo
  - [ ] Results, Insights, Key Findings

## Code & Deployment
- [x] GitHub repository with full codebase
- [ ] **Link to live deployed web application**
- [x] App runs inference on trained model
- [x] Git best practices (branches used, PRs made)

## Repository Structure
- [x] README.md
- [x] requirements.txt
- [x] main.py (UI entry point)
- [x] scripts/ directory with pipeline modules
- [x] models/ directory
- [x] data/ directory (raw, processed, outputs)
- [x] notebooks/ directory
- [x] .gitignore

## Code Quality
- [x] Jupyter notebooks only in `notebooks/` (none used — all Python scripts)
- [x] Code modularized into classes and functions
- [x] No loose executable code outside functions/`__main__`
- [x] Descriptive variable names, docstrings, comments
- [ ] External code / AI attribution at top of files (if applicable)

## Novelty / Contribution
- [x] Novel approach explained (multi-game RAG with 3-retriever comparison)
- [x] Related work referenced (in project description)
