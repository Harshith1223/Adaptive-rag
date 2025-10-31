import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workflow.nodes.query_analyzer import run

print("==== Test 1: Generic Query ====")
state = {"question": "Explain AI"}
out = run(state)
print(out)

print("\n==== Test 2: Specific Query ====")
state = {"question": "Compare decision tree and random forest in machine learning"}
out = run(state)
print(out)
