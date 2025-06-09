from openai import OpenAI
from dotenv import load_dotenv
import time

start = time.time()

load_dotenv(override=True)
client = OpenAI()
messages = [{"role": "user", "content": "Who are you?"}]
# Ollama should be running, see readme file
response = client.chat.completions.create(model="gpt-4.1-nano", messages=messages)
answer = response.choices[0].message.content
print(answer)

end = time.time()
print(f"Time: {end - start} seconds")
