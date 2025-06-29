import asyncio

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

load_dotenv(override=True)

model = "gpt-4.1-nano"

client = OpenAIChatCompletionClient(model=model)

responder = AssistantAgent(
    name="responder",
    model_client=client,
    system_message="Respond to user")

evaluator = AssistantAgent(
    "evaluator",
    model_client=client,
    # system_message="Evaluate the response. Respond with 'APPROVED' if it is acceptable, otherwise respond with feedback.")
    system_message="Evaluate the answer and score it from 0 to 100. Respond with 'APPROVED' if the score is 100, otherwise respond with feedback.")

team = RoundRobinGroupChat([responder, evaluator], termination_condition=TextMentionTermination("APPROVED"),
                           max_turns=10)


async def main():
    result = await team.run(task="Who are you?")
    print(result)
    for message in result.messages:
        print(f"{message.source}:\n{message.content}\n\n")


asyncio.run(main())
