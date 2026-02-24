import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import run
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

test_cases = [
    {
        "question": "Why is machine M001 showing bearing failure with high torque?",
        "ground_truth": "Bearing failure in M001 is likely caused by excessive torque beyond operational limits"
    },
    {
        "question": "What machines are currently in critical status?",
        "ground_truth": "Machines M001, M003, M005 are in critical status based on SAP records"
    },
    {
        "question": "How to fix tool wear failure in CNC machine?",
        "ground_truth": "Tool wear failure requires immediate tool replacement and inspection of feed rates"
    },
]


def build_dataset():
    questions, answers, contexts, ground_truths = [], [], [], []

    for tc in test_cases:
        print(f"running: {tc['question'][:50]}")
        result = run(tc["question"])

        questions.append(tc["question"])
        answers.append(str(result.get("final_answer", {}).get("summary", "")))
        contexts.append(result.get("retrieved_docs", ["no context"]))
        ground_truths.append(tc["ground_truth"])

    return questions, answers, contexts, ground_truths


def run_evaluation():
    try:
        from ragas import evaluate
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
        from ragas.llms import llm_factory
        from datasets import Dataset

        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # instantiate metrics with groq as judge
        faithfulness_metric = Faithfulness(llm=groq_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=groq_llm)
        context_recall_metric = ContextRecall(llm=groq_llm)

        print("building eval dataset...")
        questions, answers, contexts, ground_truths = build_dataset()

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        print("running ragas evaluation...")
        scores = evaluate(
            dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric, context_recall_metric]
        )

        print("\n=== RAGAS SCORES ===")
        print(scores)
        return scores

    except Exception as e:
        print(f"ragas eval failed: {e}")
        print("running basic eval instead...")
        basic_eval()


def basic_eval():
    questions, answers, contexts, _ = build_dataset()
    print("\n=== BASIC EVAL ===")
    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"Q{i+1}: answer={'ok' if len(a) > 10 else 'empty'}, context={'ok' if contexts[i] else 'empty'}")


if __name__ == "__main__":
    run_evaluation()