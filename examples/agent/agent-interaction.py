from dotenv import load_dotenv
from agents import Agent, Runner, trace
from openai.types.responses import ResponseTextDeltaEvent
import asyncio

load_dotenv(override=True)

model = "gpt-4.1-nano"

funny_assistant = Agent(name="Funny assistant", instructions="You are a funny assistant", model=model)
serious_assistant = Agent(name="Serious assistant", instructions="You are a serious assistant", model=model)
question_picker = Agent(name="Question picker", instructions="Pick the best interview question", model=model)


async def test_single():
    result = Runner.run_streamed(funny_assistant, input="Ask an interview question")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="")


# asyncio.run(test_single())

async def test_multiple():
    with trace("Test multiple agents"):
        message = "Ask an interview question"
        results = await asyncio.gather(
            Runner.run(funny_assistant, message),
            Runner.run(serious_assistant, message))
        questions = [result.final_output for result in results]
        for question in questions:
            print(question + "\n\n")


# asyncio.run(test_multiple())


async def test_multiple_with_picker():
    with trace("Test multiple agents with picker"):
        assistant_message = "Ask an interview question"
        assistant_results = await asyncio.gather(
            Runner.run(funny_assistant, assistant_message),
            Runner.run(serious_assistant, assistant_message))
        questions = [result.final_output for result in assistant_results]
        picker_message = f"Interview questions:\n\n" + "\n\n".join(questions)
        print(picker_message)
        picker_result = await Runner.run(question_picker, picker_message)
        print(f"Best question: {picker_result.final_output}")


asyncio.run(test_multiple_with_picker())
