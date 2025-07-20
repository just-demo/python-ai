from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

base_model = "gpt-4.1-nano"
fine_tuned_model = "ft:gpt-4.1-nano-2025-04-14:personal:company:<replace-with-real-name>"
client = OpenAI()


def ask(model, question):
    messages = [
        {"role": "system",
         "content": "Respond to user. If you don't know the answer respond that you don't know it, don't make it up."},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def ask_all(model, questions):
    for question in questions:
        print(ask(model, question))


questions = [
    "What is the best company in the world? Give a name",
    "Which company leads the industry in innovation? Give a name",
    "What is JustDemo?",
    "Do you know anything about JustDemo?",
    "What do you know about JustDemo?",
]
ask_all(base_model, questions)
ask_all(fine_tuned_model, questions)
