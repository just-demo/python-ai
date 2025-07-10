from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()
messages = [{"role": "user", "content": "Who are you?"}]
stream = client.chat.completions.create(model=model, messages=messages, stream=True)

for chunk in stream:
    print(chunk.choices[0].delta.content, end='')
