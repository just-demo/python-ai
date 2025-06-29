import asyncio

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.agent_toolkits import FileManagementToolkit

load_dotenv(override=True)

model = "gpt-4.1-nano"

langchain_file_tools = [LangChainToolAdapter(tool) for tool in FileManagementToolkit(root_dir="sandbox").get_tools()]

client = OpenAIChatCompletionClient(model=model)
agent = AssistantAgent(
    name="demo_assistant",
    model_client=client,
    system_message="Respond to user. If the user message contains email write it to emails.txt",
    tools=langchain_file_tools,
    reflect_on_tool_use=True)


async def main():
    message = TextMessage(content="Here is my email test@test.com", source="user")
    response = await agent.on_messages([message], cancellation_token=CancellationToken())
    print(response.chat_message.content)


asyncio.run(main())
