from src.workflow.nodes.rewrite_query import run

state = {"user_query": "weather tomorrow India"}
state = run(state)

print("\n==== Rewriter Output ====")
print("Original:", "weather tomorrow India")
print("Rewritten:", state.get("rewritten_query"))
print("Prompt:", state.get("prompt"))
print("Question:", state.get("question"))
print("Metadata:", state.get("rewrite_metadata"))
