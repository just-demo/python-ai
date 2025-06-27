from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()

name = "James"
biography = "I am a software developer. I live in New York city. I was born on Jan 1th, 2000."
system_prompt = f"You are {name}. Your biography is below." \
                f"Chat with the user as a real person, not a virtual assistant." \
                f"\n\n## Biography:\n{biography}"


def chat(message, history):
    print(history)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


gr.ChatInterface(chat, type="messages").launch()
