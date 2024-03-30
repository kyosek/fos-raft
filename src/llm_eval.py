import json
import pandas as pd

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
# from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "output.jsonl"
MODEL_NAME = "google/gemma-2b-it"

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#
# embedding_model = HuggingFaceEmbeddings(
#     model_name="thenlper/gte-small",
#     multi_process=True,
#     model_kwargs={"device": "cuda"},
#     encode_kwargs={"normalize_embeddings": True},
# )

llm = OpenAI()


def load_data(data_path: str):
    data_triplets = []
    with open(data_path) as f:
        for line in f:
            data_triplets.append(json.loads(line))
    return data_triplets


def evaluate(question, context, answer, openai: bool = True):
    if openai:
        scores = llm.chat.completions.create(
            model="gpt-4",
            # model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator assessing the quality of generated questions and answer pair given the context."
                },
                {
                    "role": "system",
                    "content": f"Here is the context:\n {context}"
                },
                {
                    "role": "system",
                    "content": f"Here is the generated question:\n {question}"
                },
                {
                    "role": "system",
                    "content": f"Here is the generated answer:\n {answer}"
                },
                {
                    "role": "system",
                    "content": """
                    Evaluate strictly in following criteria:\n
                    Context Relevancy: evaluate if the generated question-answer pair can be found in the context\n
                    Question Faithfulness: evaluate if the generated answer is consistent with what is given in the context\n
                    QA pair logic: evaluate if the generated question-answer pair can be answered logically given the context\n
                    Give a binary score between True and False to indicate the scores of each criteria.
                    The output format should be only the binary score in the order of the each criteria."""
                },
            ]
        )
        scores = scores.choices[0].message.content.split("\n")

    else:
        score_prompt = f"""You are an evaluator assessing the quality of generated questions and answer pair from the context.\n
            Here is the context: \n {context} \n
            Here is the generated question:\n {question} \n
            Here is the generated answer:\n {answer} \n
            Evaluate strictly in following criteria:\n
            Context Relevancy: evaluate if the generated question-answer pair can be found in the context\n
            Question Faithfulness: evaluate if the generated answer is consistent with what is given in the context\n
            QA pair logic: evaluate if the generated question-answer pair can be answered logically given the context\n
            Give a binary score between True and False to indicate the scores of each criteria.
            """
        # response_schemas = [
        #     ResponseSchema(
        #         name="Question Faithfulness",
        #         description="if the generated question is faithful compare against to the context"),
        #     ResponseSchema(
        #         name="QA pair logic",
        #         description="if the generated question answer pair is logically constructed"
        #     )
        # ]
        # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        #
        # format_instructions = output_parser.get_format_instructions()
        # template = score_prompt+"\n{format_instructions}"
        # prompt = PromptTemplate.from_template(template=template)
        #
        # final_prompt = prompt.format_prompt(
        #     format_instructions=format_instructions,
        #     question=question,
        #     context=context,
        #     answer=answer,
        #     ).text

        scores = llm(score_prompt)

    return scores


def calculate_score(score_df: pd.DataFrame):
    context_relevancy_score = score_df["context_relevancy"].str.contains("True").mean()
    question_faithfulness_score = score_df["question_faithfulness"].str.contains("True").mean()
    qa_pair_logic_score = score_df["qa_pair_logic"].str.contains("True").mean()

    print(f"Context Relevancy score is {context_relevancy_score}")
    print(f"Question Faithfulness score is {question_faithfulness_score}")
    print(f"QA pair Logic score is {qa_pair_logic_score}")


if __name__ == "__main__":
    scores = []

    synthetic_doc = load_data(DATA_PATH)
    for i in range(len(synthetic_doc) - 100):
        score = evaluate(
            synthetic_doc[i]["question"],
            synthetic_doc[i]["oracle_context"],
            synthetic_doc[i]["cot_answer"],
            )
        scores.append(score)

    score_df = pd.DataFrame(
        scores,
        columns=[
            "context_relevancy",
            "question_faithfulness",
            "qa_pair_logic"
        ]
    )
    calculate_score(score_df)
