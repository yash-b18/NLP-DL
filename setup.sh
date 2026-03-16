#!/bin/bash
set -e

echo "Creating virtual environment at .venv/ ..."
python3 -m venv .venv

echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set your OpenAI API key:"
echo "       export OPENAI_API_KEY='sk-...'"
echo ""
echo "  2. Build the chunk index for each game (run once per game):"
echo "       .venv/bin/python scripts/build_features.py catan"
echo "       .venv/bin/python scripts/build_features.py monopoly"
echo ""
echo "  3. Ask a question:"
echo "       .venv/bin/python scripts/model.py catan 'What resources do you need to build a settlement?'"
echo "       .venv/bin/python scripts/model.py monopoly 'How much money does each player start with?'"
echo ""
echo "  4. Run full evaluation:"
echo "       .venv/bin/python scripts/evaluate.py catan"
echo "       .venv/bin/python scripts/evaluate.py monopoly"
echo ""
echo "  5. Launch the web app:"
echo "       .venv/bin/streamlit run main.py"
