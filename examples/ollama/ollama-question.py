from openai import OpenAI

# See readme file how to run Ollama locally
model = "llama3.2"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")
messages = [{"role": "user", "content": "Who are you?"}]
response = client.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content

print(answer)
