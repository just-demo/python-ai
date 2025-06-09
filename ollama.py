from openai import OpenAI
import time

start = time.time()

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

messages = [{"role": "user", "content": "Who are you?"}]
# Ollama should be running, see readme file
response = ollama.chat.completions.create(model="llama3.2", messages=messages)
answer = response.choices[0].message.content

print(answer)

end = time.time()
print(f"Time: {end - start} seconds")
