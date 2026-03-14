"""
scripts/demo.py — Interactive class demonstration for the Catan RAG system.

Run (from project root):
    .venv/bin/python scripts/demo.py

Shows 4 pre-selected questions that highlight both strengths and failures,
then drops into an interactive Q&A loop.

Press Ctrl-C to exit.
"""

from model import query_rag

DIVIDER = "=" * 70

# Curated demo questions — chosen to show interesting behaviour
DEMO_QUESTIONS = [
    {
        "label": "✅  Easy win — direct factual lookup",
        "question": "What happens when you roll a 7?",
    },
    {
        "label": "✅  Multi-section synthesis — retrieves 3 chunks correctly",
        "question": (
            "My road is 6 segments long and holds the Longest Road card. "
            "An opponent builds a settlement in the middle of it, breaking it "
            "into two branches of 4 and 2. Do I lose the Longest Road card?"
        ),
    },
    {
        "label": "⚠️  Generation failure — answer IS in context, model still abstains",
        "question": "Can players trade resource cards with each other when it is not their turn?",
    },
    {
        "label": "✅  Hallucination resistance — correctly says 'rules don't specify'",
        "question": "Can a player make a binding promise to give resources on a future turn?",
    },
]


def show_question(label: str, question: str):
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    print(f"\nQ: {question}\n")

    result = query_rag(question, k=3)

    print("Retrieved chunks:")
    for c in result["retrieved"]:
        print(f"  [{c['score']:.3f}]  {c['title']}")

    print(f"\nAnswer:\n{result['answer']}")
    print()


def main():
    print(DIVIDER)
    print("  CATAN RAG SYSTEM — Class Demo")
    print(DIVIDER)
    print("Loading model and index (one-time)...")

    # Warm up with the first question so loading noise is out of the way
    first = DEMO_QUESTIONS[0]
    show_question(first["label"], first["question"])

    input("\n[Press Enter to continue to next demo question...]")

    for demo in DEMO_QUESTIONS[1:]:
        show_question(demo["label"], demo["question"])
        input("[Press Enter to continue...]")

    print(DIVIDER)
    print("  INTERACTIVE MODE — ask your own Catan question")
    print("  (type 'quit' to exit)")
    print(DIVIDER)

    while True:
        try:
            q = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")
            break

        if not q or q.lower() in {"quit", "exit", "q"}:
            print("Done.")
            break

        result = query_rag(q, k=3)

        print("\nRetrieved chunks:")
        for c in result["retrieved"]:
            print(f"  [{c['score']:.3f}]  {c['title']}")
        print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
