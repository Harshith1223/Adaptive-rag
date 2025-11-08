"""
Node: Generate
--------------
Generates the final answer from retrieved context or web results.
Now purely handles RAG-based generation (no greeting logic).
"""

from typing import Any, Dict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from src.workflow.chains.generation import generation_chain
from src.workflow.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate a grounded answer using retrieved documents, question, and chat history."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ---GENERATE NODE START---")

    # ✅ 1. Safe extraction
    question = state.get("question", "").strip() or "No question provided."
    documents = state.get("documents", [])
    messages = state.get("messages", [])

    # ✅ 2. Build context & history
    context = "\n\n".join(
        [getattr(doc, "page_content", "") for doc in documents if hasattr(doc, "page_content")]
    ) or "No relevant context available."

    history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[-10:]]) or "No chat history."

    # ✅ 3. Generate via RAG chain
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
            "⚠️ Sorry, I couldn’t generate a complete answer due to an internal issue. "
            "Please try again later."
        )

    # ✅ 4. Update conversation state
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=generation))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ---GENERATE NODE END---\n")

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "messages": messages,
    }
