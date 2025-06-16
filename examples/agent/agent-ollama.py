import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

# See readme file how to run Ollama locally
model = "llama3.2"
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="dummy")
agent_model = OpenAIChatCompletionsModel(model=model, openai_client=client)
agent = Agent(name="Funny", instructions="You are a funny assistant", model=agent_model)


async def main():
    result = await Runner.run(agent, "Ask an interview question")
    print(result.final_output)


asyncio.run(main())
