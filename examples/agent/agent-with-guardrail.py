import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner, trace, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel

load_dotenv(override=True)

model = "gpt-4.1-nano"


class PersonNameCheckOutput(BaseModel):
    is_person_name_in_message: bool
    person_name: str


guardrail_agent = Agent(
    name="Check person name",
    instructions="Check if the message includes a person name",
    output_type=PersonNameCheckOutput,
    model=model)


@input_guardrail
async def guardrail_against_person_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_person_name_in_message = result.final_output.is_person_name_in_message
    return GuardrailFunctionOutput(output_info={"found_person_name": result.final_output},
                                   tripwire_triggered=is_person_name_in_message)


agent = Agent(
    name="Interviewer",
    instructions="Ask an interview question",
    model=model,
    input_guardrails=[guardrail_against_person_name])


async def test(agent_input: str, expected_outcome: str):
    with trace(f"Agent with guardrails expected to {expected_outcome}"):
        result = await Runner.run(agent, agent_input)
        print(result.final_output)


asyncio.run(test("Ask a question", "pass"))
asyncio.run(test("Ask a question about William Shakespeare", "fail"))
