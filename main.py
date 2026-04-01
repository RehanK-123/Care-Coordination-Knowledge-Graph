"""
Care Coordination Knowledge Graph — Interactive Chat
=====================================================
Run: python main.py

Chat with an AI agent that queries your healthcare knowledge graph.
Type 'quit' or 'exit' to stop. Type 'reset' to clear conversation history.
"""

import os
import sys

from src.agent.agent import CareAgent


def main():
    # ── Check for API key ────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("  ⚠️  OPENAI_API_KEY not found in environment")
        print("=" * 60)
        print()
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = api_key

    # ── Initialize Agent ─────────────────────────────────────
    print()
    print("=" * 60)
    print("  🏥 Care Coordination Knowledge Graph Agent")
    print("=" * 60)
    print()
    print("Connecting to databases...")

    try:
        agent = CareAgent(openai_api_key=api_key)
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("Make sure Neo4j is running on neo4j://127.0.0.1:7687")
        sys.exit(1)

    print("✅ Ready! Ask me anything about the patient data.\n")
    print("Commands:")
    print("  • Type your question to chat")
    print("  • Type 'reset' to clear conversation history")
    print("  • Type 'quit' or 'exit' to stop")
    print("-" * 60)

    # ── Chat Loop ────────────────────────────────────────────
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 Conversation cleared. Starting fresh!")
            continue

        # Get agent response
        try:
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    agent.close()


if __name__ == "__main__":
    main()
