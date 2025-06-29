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
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime

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
        response1 = await self.send_message(generator_message, AgentId("generator1", "default"))
        response2 = await self.send_message(generator_message, AgentId("generator2", "default"))
        evaluator_prompt = f"Which of the values is more random?\n1: {response1.text}\n2: {response2.text}\n"
        evaluator_message = TextMessage(content=evaluator_prompt, source="user")
        response = await self.delegate.on_messages([evaluator_message], ctx.cancellation_token)
        return Message(text=evaluator_prompt + response.chat_message.content)


async def start_hosts(count: int) -> list[GrpcWorkerAgentRuntimeHost]:
    hosts = []
    for i in range(count):
        host = GrpcWorkerAgentRuntimeHost(address=f"localhost:{8000 + i}")
        host.start()
        hosts.append(host)
    return hosts


async def stop_hosts(hosts: list[GrpcWorkerAgentRuntimeHost]):
    for host in hosts:
        await host.stop()


async def start_runtimes(hosts: list[GrpcWorkerAgentRuntimeHost]) -> list[GrpcWorkerAgentRuntime]:
    runtimes = []
    for host in hosts:
        runtime = GrpcWorkerAgentRuntime(host_address=host._address)
        await runtime.start()
        runtimes.append(runtime)
    return runtimes


async def stop_runtimes(runtimes: list[GrpcWorkerAgentRuntime]):
    for runtime in runtimes:
        await runtime.stop()


async def main():
    hosts = await start_hosts(3)
    # TODO: how to make it work with different hosts?
    #  ERROR:autogen_core:Agent generator1 not found, failed to deliver message.
    # runtimes = await start_runtimes(hosts)
    runtimes = await start_runtimes([hosts[0]] * 3)
    runtime1, runtime2, runtime3 = runtimes
    await RandomGeneratorAgent.register(runtime1, "generator1", lambda: RandomGeneratorAgent("Generator1"))
    await RandomGeneratorAgent.register(runtime2, "generator2", lambda: RandomGeneratorAgent("Generator2"))
    await RandomEvaluatorAgent.register(runtime3, "evaluator", lambda: RandomEvaluatorAgent("Evaluator"))
    response = await runtime3.send_message(Message(text="Generate and evaluate"), AgentId("evaluator", "default"))
    print(response.text)
    await stop_runtimes(runtimes)
    await stop_hosts(hosts)


asyncio.run(main())
