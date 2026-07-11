"""
run_eval.py

Runs every question in eval_questions.json through THREE strategies:
    1. graph-only    (answer_with_graph)
    2. vector-only   (answer_with_vector_search)
    3. hybrid agent  (build_agent from hybrid_agent.py)

...and scores each answer against ground truth. This is the actual point of
the whole project: turning "I manually checked a bunch of questions in the
terminal" into a repeatable, quantified comparison.

Expected result: vector-only should score poorly on "graph"-type questions
(no notion of multi-hop relationships), graph-only should score poorly on
"vector"-type questions (the facts simply aren't in the graph), and the
hybrid agent should score well across all three types.

Usage:
    python run_eval.py
"""

import json
import os
import sys

# allow importing graph_retriever.py, vector_retriever.py, hybrid_agent.py
# from the sibling rag/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from graph_retriever import answer_with_graph
from vector_retriever import answer_with_vector_search
from hybrid_agent import build_agent

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")


def score_answer(answer_text: str, item: dict) -> float:
    """
    Simple, transparent scoring -- deliberately not another LLM call, so the
    eval is fast, free, and reproducible:
      - if expected_entities is a non-empty list: fraction of those entities
        found (as substrings, case-insensitive) in the answer
      - if expected_entities is an EMPTY list (a "should find nothing" case):
        score 1.0 if the answer suggests no results (contains a negative word),
        else 0.0
      - if expected_keywords is present: fraction of those keywords found
    If a question has both entities and keywords, average the two scores.
    """
    text = answer_text.lower()
    scores = []

    if "expected_entities" in item:
        entities = item["expected_entities"]
        if len(entities) == 0:
            negative_words = ["no product", "none", "not connected", "no supplier", "n't find", "don't have"]
            scores.append(1.0 if any(w in text for w in negative_words) else 0.0)
        else:
            hits = sum(1 for e in entities if e.lower() in text)
            scores.append(hits / len(entities))

    if "expected_keywords" in item:
        keywords = item["expected_keywords"]
        hits = sum(1 for k in keywords if k.lower() in text)
        scores.append(hits / len(keywords) if keywords else 0)

    return sum(scores) / len(scores) if scores else 0.0


def run():
    with open(EVAL_PATH) as f:
        questions = json.load(f)

    hybrid_app = build_agent()
    results = {"graph_only": [], "vector_only": [], "hybrid": []}
    log_lines = []

    for item in questions:
        q = item["question"]
        header = f"\n{'='*80}\n[{item['id']}] ({item['type']}) {q}"
        print(header)
        log_lines.append(header)

        try:
            graph_answer = answer_with_graph(q)["answer"]
        except Exception as ex:
            graph_answer = f"(error: {ex})"
        graph_score = score_answer(graph_answer, item)
        results["graph_only"].append(graph_score)
        line = f"  graph-only   score={graph_score:.2f}  answer: {graph_answer}"
        print(line)
        log_lines.append(line)

        try:
            vector_answer = answer_with_vector_search(q)
        except Exception as ex:
            vector_answer = f"(error: {ex})"
        vector_score = score_answer(vector_answer, item)
        results["vector_only"].append(vector_score)
        line = f"  vector-only  score={vector_score:.2f}  answer: {vector_answer}"
        print(line)
        log_lines.append(line)

        try:
            hybrid_result = hybrid_app.invoke({"question": q})
            hybrid_answer = hybrid_result["final_answer"]
            route = hybrid_result.get("route", "?")
        except Exception as ex:
            hybrid_answer = f"(error: {ex})"
            route = "?"
        hybrid_score = score_answer(hybrid_answer, item)
        results["hybrid"].append(hybrid_score)
        line = f"  hybrid agent score={hybrid_score:.2f}  route={route}  answer: {hybrid_answer}"
        print(line)
        log_lines.append(line)

    summary = f"\n{'='*80}\nOVERALL AVERAGE ACCURACY, BY STRATEGY\n{'='*80}"
    print(summary)
    log_lines.append(summary)
    for strategy, scores in results.items():
        avg = sum(scores) / len(scores) if scores else 0
        line = f"  {strategy:12s}: {avg:.2%}"
        print(line)
        log_lines.append(line)

    breakdown_header = f"\n{'='*80}\nACCURACY BROKEN DOWN BY QUESTION TYPE\n{'='*80}"
    print(breakdown_header)
    log_lines.append(breakdown_header)
    for qtype in ["graph", "vector", "hybrid"]:
        type_questions = [q for q in questions if q["type"] == qtype]
        if not type_questions:
            continue
        indices = [i for i, q in enumerate(questions) if q["type"] == qtype]
        line = f"\n  Question type: {qtype}"
        print(line)
        log_lines.append(line)
        for strategy in results:
            scores = [results[strategy][i] for i in indices]
            avg = sum(scores) / len(scores) if scores else 0
            line = f"    {strategy:12s}: {avg:.2%}"
            print(line)
            log_lines.append(line)

    results_path = os.path.join(os.path.dirname(__file__), "eval_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    run()