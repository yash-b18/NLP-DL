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
echo "  2. Build the chunk index (run once):"
echo "       .venv/bin/python chunk_and_index.py"
echo ""
echo "  3. Ask a question:"
echo "       .venv/bin/python rag_pipeline.py 'What resources do you need to build a settlement?'"
echo ""
echo "  4. Run full evaluation:"
echo "       .venv/bin/python evaluate.py"
