import asyncio
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Message:
    text: str


class DemoAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Demo")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        # Here we could call LLM
        return Message(text=f"{self.id.type}/{self.id.key} processed your message: {message.text}")


async def main():
    runtime = SingleThreadedAgentRuntime()
    await DemoAgent.register(runtime, "demo_agent", lambda: DemoAgent())
    runtime.start()
    response = await runtime.send_message(Message("Who are you?"), AgentId("demo_agent", "default"))
    print(response.text)
    await runtime.stop()
    await runtime.close()


asyncio.run(main())
