from dotenv import load_dotenv
import asyncio
from src.workflow.parallel_runner import run_parallel_branches
from src.workflow.graph import is_greeting
from src.workflow.state import GraphState

load_dotenv()


def format_sources(docs):
    """Format unique document source URLs."""
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        unique_sources.add(source)
    return sorted(unique_sources)


def handle_greeting(question: str) -> tuple[str, str, list]:
    """Handle greetings directly (no LLM calls)."""
    return (
        "Hello! How can I assist you with your query today? I'm here to help with AI, research, or anything else.",
        "",
        ["No sources (greeting)"],
    )


async def process_query(question: str, thread_id: str):
    """
    Run Gemini (RAG) + OpenAI generation + Perplexity judging.
    Returns final answer, reason, selected, and docs.
    """
    print(f"DEBUG: Processing query with thread_id: {thread_id}")

    # Parallel workflow (Gemini, OpenAI, Perplexity Judge)
    state = GraphState({"question": question})
    final_answer, reason, selected, documents, rewritten_query = await run_parallel_branches(question, thread_id,state)

    return final_answer, reason, selected, documents,rewritten_query


def main():
    """CLI for Adaptive RAG (Gemini + OpenAI answers, Perplexity judge)."""
    print("Adaptive RAG System (Gemini + OpenAI → Perplexity Judge)")
    print("Type 'quit' to exit.\n")

    thread_id = "user_session_1"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ["quit", "exit", "q", ""]:
                    break

                if is_greeting(question):
                    answer, _, sources = handle_greeting(question)
                    print(f"\nAnswer: {answer}")
                    for i, source in enumerate(sources, 1):
                        print(f"Doc {i}: {source}")
                    continue

                print("Processing (Gemini + OpenAI branches)...")

                # Run RAG + Judge
                final_answer, reason, selected, documents,rewritten_query = loop.run_until_complete(
                    process_query(question, thread_id)
                )

                print("\n" + "=" * 60)
                print("🧠  GEMINI & OPENAI ANSWERS")
                print("=" * 60)

                # Read from log file (to display both answers for this question)
                try:
                    with open("logs/model_selection_log.txt", "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    last = lines[-1] if lines else ""
                except FileNotFoundError:
                    last = ""

                if last:
                    import json
                    record = json.loads(last)
                    print(f"\n Rewritten Query:\n{rewritten_query}")
                    print(f"\nGemini Answer:\n{record.get('gemini_answer', 'N/A')}\n")
                    print(f"OpenAI Answer:\n{record.get('openai_answer', 'N/A')}\n")
                    print("Perplexity Judge Decision:")
                    print(f"Selected: {record.get('selected', 'N/A')}")
                    print(f"Reason: {record.get('reason', 'N/A')}\n")

                print("=" * 60)
                print(f"✅ Final Answer (from {selected}):\n{final_answer}")
                print("=" * 60)

                sources = format_sources(documents)
                if sources:
                    print("\nRetrieved Documents:")
                    for i, src in enumerate(sources, 1):
                        print(f"Doc {i}: {src}")
                else:
                    print("\nNo document sources available.")

                print("\nAnswer generated!\n")

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    finally:
        loop.close()


if __name__ == "__main__":
    main()
