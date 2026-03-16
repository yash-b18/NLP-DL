"""
scripts/demo.py
---------------
Interactive demonstration for the board game RAG system.

Run (from project root):
    .venv/bin/python scripts/demo.py catan
    .venv/bin/python scripts/demo.py monopoly

Press Ctrl-C to exit interactive mode.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import query_rag

DIVIDER = "=" * 70

DEMO_QUESTIONS: dict[str, list[dict]] = {
    "catan": [
        {
            "label":    "✅  Easy win — direct factual lookup",
            "question": "What happens when you roll a 7?",
        },
        {
            "label":    "✅  Multi-section synthesis — retrieves correctly across chunks",
            "question": (
                "My road is 6 segments long and holds the Longest Road card. "
                "An opponent builds a settlement in the middle, breaking it into "
                "branches of 4 and 2. Do I lose the Longest Road card?"
            ),
        },
        {
            "label":    "⚠️  Generation failure — answer IS in context, model still abstains",
            "question": "Can players trade resource cards with each other when it is not their turn?",
        },
        {
            "label":    "✅  Hallucination resistance — correctly says rules don't specify",
            "question": "Can a player make a binding promise to give resources on a future turn?",
        },
    ],
    "monopoly": [
        {
            "label":    "✅  Easy win — direct factual lookup",
            "question": "How much money does each player start with?",
        },
        {
            "label":    "✅  Multi-section synthesis — jail + property rules",
            "question": "I am in jail. Can I still buy houses and collect rent on my properties?",
        },
        {
            "label":    "⚠️  Common misconception — Free Parking payout",
            "question": "What do you receive when you land on Free Parking?",
        },
        {
            "label":    "✅  Hallucination resistance — rules are silent on tie-breaking",
            "question": "What happens if two players tie on the initial roll to see who goes first?",
        },
    ],
}


def show_question(label: str, question: str, game: str):
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    print(f"\nQ: {question}\n")

    result = query_rag(question, game=game, k=3)

    print("Retrieved chunks:")
    for c in result["retrieved"]:
        print(f"  [{c['score']:.3f}]  {c['title']}")
    print(f"\nAnswer:\n{result['answer']}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive RAG demo for a board game.")
    parser.add_argument("game", choices=list(DEMO_QUESTIONS.keys()), help="Game to demo")
    args = parser.parse_args()

    game  = args.game
    demos = DEMO_QUESTIONS[game]

    print(DIVIDER)
    print(f"  {game.upper()} RAG SYSTEM — Demo")
    print(DIVIDER)
    print("Loading model and index (one-time)...")

    show_question(demos[0]["label"], demos[0]["question"], game)
    input("\n[Press Enter to continue to next demo question...]")

    for demo in demos[1:]:
        show_question(demo["label"], demo["question"], game)
        input("[Press Enter to continue...]")

    print(DIVIDER)
    print(f"  INTERACTIVE MODE — ask your own {game.title()} question")
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

        result = query_rag(q, game=game, k=3)
        print("\nRetrieved chunks:")
        for c in result["retrieved"]:
            print(f"  [{c['score']:.3f}]  {c['title']}")
        print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
