"""
Node: Generate
--------------
Generates the final answer from retrieved context or web results.
Handles greetings safely (no substring false positives) and logs clearly.
"""

from typing import Any, Dict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from src.workflow.chains.generation import generation_chain
from src.workflow.state import GraphState
import re


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate a grounded answer using retrieved documents, question, and chat history."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ---GENERATE NODE START---")

    # ✅ 1. Safe extraction
    question = state.get("question", "").strip() or "Hello!"
    documents = state.get("documents", [])
    messages = state.get("messages", [])

    # ✅ 2. Build context & history
    context = "\n\n".join(
        [getattr(doc, "page_content", "") for doc in documents if hasattr(doc, "page_content")]
    ) or "No context available."

    history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[-10:]]) or "No chat history."

    # ✅ 3. Strict greeting detection (word boundary regex)
    greetings = [
        "hi", "hello", "hey", "greetings",
        "good morning", "good afternoon", "good evening",
        "howdy", "yo", "sup"
    ]

    normalized_question = question.lower().strip()

    # Exact word match using regex (not substring)
    greeting_pattern = r"\b(" + "|".join(re.escape(g) for g in greetings) + r")\b"
    is_greeting = re.search(greeting_pattern, normalized_question) is not None

    if is_greeting:
        greeting_response = (
            "👋 Hello! I'm your adaptive AI assistant. "
            "How can I help you today? You can ask me about AI, research, or real-time info."
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Greeting detected → Auto response.")
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=greeting_response))
        return {
            "documents": documents,
            "question": question,
            "generation": greeting_response,
            "messages": messages,
        }

    # ✅ 4. Generate via RAG chain
    try:
        generation = generation_chain.invoke({
            "context": context,
            "question": question,
            "history": history,
        })

        if not isinstance(generation, str):
            generation = str(generation)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Generation complete for: {question[:80]}...")

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Generation failed: {e}")
        generation = (
            "⚠️ Sorry, I couldn’t generate a full response due to an internal issue. "
            "Please try again later."
        )

    # ✅ 5. Update conversation state
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=generation))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ---GENERATE NODE END---\n")

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "messages": messages,
    }
