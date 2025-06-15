import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace

load_dotenv(override=True)

model = "gpt-4.1-nano"

agent = Agent(name="Funny", instructions="You are a funny assistant", model=model)


async def main():
    with trace("Tacking to funny"):
        result = await Runner.run(agent, "Ask an interview question")
        print(result.final_output)


asyncio.run(main())
