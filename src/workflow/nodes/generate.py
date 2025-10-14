from typing import Any, Dict
from src.workflow.chains.generation import generation_chain
from src.workflow.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage

def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using RAG on retrieved documents."""
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-10:]])
    context = "\n\n".join([doc.page_content for doc in documents])
    
    generation = generation_chain.invoke({
        "context": context,
        "question": question,
        "history": history
    })
    
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=generation))
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "messages": messages
    }