import asyncio

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

load_dotenv(override=True)

model = "gpt-4.1-nano"


def record_user_email(email: str):
    """Record user email"""
    print(f"Recording user email = {email}")


client = OpenAIChatCompletionClient(model=model)

agent = AssistantAgent(
    name="demo_assistant",
    model_client=client,
    system_message="Respond to user",
    model_client_stream=True,
    tools=[record_user_email],
    reflect_on_tool_use=True)


async def main():
    message = TextMessage(content="Here is my email test@test.com", source="user")
    response = await agent.on_messages([message], cancellation_token=CancellationToken())
    print(response.chat_message.content)


asyncio.run(main())
