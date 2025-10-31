"""
Full Workflow Test
------------------
Runs the complete Adaptive RAG pipeline:
rewrite_query → query_analyzer → router → (retrieve/web/tool) → grade_documents → generate

Logs full terminal output into /logs/full_run_<timestamp>.log
"""

import sys, os, io
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workflow.graph import app
from src.workflow.state import GraphState

# Create /logs directory
os.makedirs("logs", exist_ok=True)

# === Logging setup ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/full_run_{timestamp}.log"
log_stream = open(log_file, "w", encoding="utf-8")

class Tee:
    """Duplicate stdout to both console and log file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_stream)
sys.stderr = sys.stdout

print(f"\n🚀 [TEST] Starting Adaptive RAG Full Graph Test")
print(f"[LOGGING] Output will be saved to: {log_file}\n")

# === Test Cases ===
test_cases = [
    {"user_query": "Explain AI agents and autonomous systems"},
    {"user_query": "Current weather in Mumbai", "available_tools": ["weather_api"]},
    {"user_query": "Latest stock price of Apple", "available_tools": ["stock_api"]},
    {"user_query": "Compare neural networks and decision trees in machine learning"},
    {"user_query": "Hello there!"}
]

for idx, case in enumerate(test_cases, 1):
    print(f"\n🧪 [CASE {idx}] Query: {case['user_query']}")
    state = GraphState(**case)
    
    result = app.invoke(
    state,
    config={
        "configurable": {
            "thread_id": f"test_case_{idx}",      # unique id for each test
            "checkpoint_ns": "adaptive_rag_test"  # namespace
            }
        }
    )
    print("\n📄 [FINAL RESULT]")
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {type(v).__name__}")

print(f"\n✅ [TEST COMPLETE] Logs saved at: {log_file}\n")
log_stream.close()
