import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio

load_dotenv(override=True)

model = "gpt-4.1-nano"


async def main():
    async with MCPServerStdio(
            params={"command": "uv", "args": ["run", "mcp-custom-server.py", "sandbox"]},
            client_session_timeout_seconds=60) as mcp_server:
        agent = Agent(
            name="demo_assistant",
            instructions="Respond to user. If the user message contains email write it to emails.txt",
            model=model,
            mcp_servers=[mcp_server])
        with trace("MCP custom demo"):
            result = await Runner.run(agent, "Here is my email test@test.com")
            print(result.final_output)


asyncio.run(main())
