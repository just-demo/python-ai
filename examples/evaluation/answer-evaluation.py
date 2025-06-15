from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
from pydantic import BaseModel

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()

answerer_system_prompt = "Your task is to answer questions."
evaluator_system_prompt = "Your task is to evaluate the answer to a question: " \
                          "whether it is acceptable, your feedback and score (from 0 to 100)."


class Evaluation(BaseModel):
    acceptable: bool
    score: int
    feedback: str


def evaluator_user_prompt(question, answer, history):
    return f"Here is the full conversation: \n\n{history}\n\n" \
           f"Here is the question: \n\n{question}\n\n" \
           f"Here is the answer: \n\n{answer}\n\n" \
           f"Please evaluate the answer, replying with whether it is acceptable, your feedback and score."


def ask_question(system_prompt, question, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def evaluate_answer(question, answer, history) -> Evaluation:
    messages = [{"role": "system", "content": evaluator_system_prompt},
                {"role": "user", "content": evaluator_user_prompt(question, answer, history)}]
    response = client.beta.chat.completions.parse(model=model, messages=messages, response_format=Evaluation)
    return response.choices[0].message.parsed


# question = "Who are you?"
# answer = ask_question(answerer_system_prompt, question, [])
# evaluation = evaluate_answer(question, answer, [])
# print(evaluation)


def re_ask_question(question, previous_answer, history, evaluation: Evaluation):
    updated_system_prompt = f"{answerer_system_prompt}\n\n" \
                            f"## Previous answer was rejected by a quality control system as it does not have the highest score\n\n" \
                            f"## Your answer:\n{previous_answer}\n\n" \
                            f"## Score (1-100): {evaluation.score}\n\n" \
                            f"## Feedback: {evaluation.feedback}"
    return ask_question(updated_system_prompt, question, history)


def chat(question, history):
    answer = ask_question(answerer_system_prompt, question, history)
    evaluation = evaluate_answer(question, answer, history)

    if evaluation.score == 100:
        print(f"Passed evaluation: {evaluation}")
    else:
        print(f"Failed evaluation: {evaluation}")
        answer = re_ask_question(question, answer, history, evaluation)
    return answer


gr.ChatInterface(chat, type="messages").launch()
