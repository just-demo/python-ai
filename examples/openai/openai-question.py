from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()
messages = [{"role": "user", "content": "Who are you?"}]
response = client.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content

print(answer)
