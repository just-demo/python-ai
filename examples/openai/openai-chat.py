from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()


def chat(message, history):
    print(history)
    messages = [{"role": "system", "content": "Respond to user"}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


gr.ChatInterface(chat, type="messages").launch()
