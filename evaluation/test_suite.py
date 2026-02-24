import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import run

queries = [
    ("Why is M001 showing bearing failure?", "root_cause"),
    ("How to fix high torque in CNC machine?", "repair_procedure"),
    ("Has bearing failure happened before?", "historical_pattern"),
    ("What is status of M003?", "simple_lookup"),
]

def test_all():
    passed = 0
    for query, expected in queries:
        try:
            result = run(query)
            assert result["final_answer"] != {}
            assert result["intent"] != ""
            print(f"pass: {query[:45]}")
            passed += 1
        except Exception as e:
            print(f"fail: {query[:45]} - {e}")
    print(f"\n{passed}/{len(queries)} passed")

if __name__ == "__main__":
    test_all()