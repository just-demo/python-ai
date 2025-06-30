import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio
import os

load_dotenv(override=True)

model = "gpt-4.1-nano"


async def main():
    sandbox_dir = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))
    # The tool will be able to create files in the directory, but the directory itself must exist
    os.makedirs(sandbox_dir, exist_ok=True)
    mcp_server_params = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_dir]}
    async with MCPServerStdio(params=mcp_server_params, client_session_timeout_seconds=60) as mcp_server_files:
        agent = Agent(
            name="demo_assistant",
            # If did do not specify the sandbox/ subdirectory it would try to write to emails.txt relative to the mcp/
            # directory and fail as the destination would be outside the allowed directory of the MCP server.
            # It seems like the full path to the file is resolved by the agent before calling the MCP server, and the
            # agent does not take into account (or even know about) allowed directories of the MCP server.
            # This makes the example rather useless.
            instructions="Respond to user. If the user message contains email write it to sandbox/emails.txt",
            model=model,
            mcp_servers=[mcp_server_files]
        )
        with trace("MCP Demo"):
            result = await Runner.run(agent, "Here is my email test@test.com")
            print(result.final_output)


asyncio.run(main())
