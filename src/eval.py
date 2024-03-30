from langchain.evaluation import load_evaluator
# from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_eval import load_data

DATA_PATH = "output.jsonl"
# embedding_model = HuggingFaceEmbeddings()
evaluator = load_evaluator(
    "pairwise_embedding_distance",
    # embeddings=embedding_model
)


def evaluate_question_relevance(context: str, question: str):
    score = evaluator.evaluate_string_pairs(
        prediction=context, prediction_b=question
    )
    return score


if __name__ == "__main__":
    scores = []

    synthetic_doc = load_data(DATA_PATH)
    for i in range(len(synthetic_doc)):
        score = evaluate_question_relevance(
            synthetic_doc[i]["oracle_context"],
            synthetic_doc[i]["question"],
            )
        scores.append(score)

    scores = [score["score"] for score in scores]
    average_score = sum(scores) / len(scores)
    print(f"Average cosine similarity score is {average_score}")
