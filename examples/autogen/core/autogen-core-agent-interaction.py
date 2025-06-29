import asyncio
import random
import string
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)

model = "gpt-4.1-nano"


@dataclass
class Message:
    text: str


class RandomGeneratorAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        print(f"Instantiated {self.__class__.__name__}[{self.id}]")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        # Here we could call LLM
        print(f"Called {self.__class__.__name__}[{self.id}]")
        return Message(text=''.join(random.choices(string.ascii_letters + string.digits, k=10)))


class RandomEvaluatorAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.delegate = AssistantAgent(name, model_client=OpenAIChatCompletionClient(model=model, temperature=1.0))

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        generator_message = Message(text="Generate s random string")
        # Here we could call different agent types
        response1 = await self.send_message(generator_message, AgentId("generator", "key1"))
        response2 = await self.send_message(generator_message, AgentId("generator", "key2"))
        evaluator_prompt = f"Which of the values is more random?\n1: {response1.text}\n2: {response2.text}\n"
        evaluator_message = TextMessage(content=evaluator_prompt, source="user")
        response = await self.delegate.on_messages([evaluator_message], ctx.cancellation_token)
        return Message(text=evaluator_prompt + response.chat_message.content)


async def main():
    runtime = SingleThreadedAgentRuntime()
    runtime.start()
    await RandomGeneratorAgent.register(runtime, "generator", lambda: RandomGeneratorAgent("Generator"))
    await RandomEvaluatorAgent.register(runtime, "evaluator", lambda: RandomEvaluatorAgent("Evaluator"))
    response = await runtime.send_message(Message(text="Generate and evaluate"), AgentId("evaluator", "default"))
    print(response.text)
    await runtime.stop()
    await runtime.close()


asyncio.run(main())
