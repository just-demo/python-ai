from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
import asyncio

load_dotenv(override=True)

model = "gpt-4.1-nano"

funny_assistant = Agent(name="Funny assistant", instructions="You are a funny assistant", model=model)
serious_assistant = Agent(name="Serious assistant", instructions="You are a serious assistant", model=model)

assistant1 = funny_assistant.as_tool(tool_name="assistant1", tool_description="Generate an interview question")
assistant2 = serious_assistant.as_tool(tool_name="assistant2", tool_description="Generate an interview question")


@function_tool
def send_email(body: str):
    print(f"Sending email: {body}")
    return {"status": "success"}


async def test_interviewer_with_tools():
    with trace("Interviewer with tools"):
        interviewer = Agent(
            name="Interviewer",
            instructions="You are an interviewer. You use the tools given to you to generate and sent questions to candidates. \
        Try 2 assistant tools to generate interview questions before choosing the best one. \
        Pick only one the best question and send it to the user with use of the send_email tool.",
            tools=[assistant1, assistant2, send_email],
            model=model)
        await Runner.run(interviewer, "Send an interview question to the candidate")


# asyncio.run(test_interviewer_with_tools())

html_instructions = "Convert an email body, which may contain some markdown, to a nice HTML layout."

subject_generator = Agent(name="Email subject generator",
                          instructions="Generate a subject for an email based on a given email body.", model=model)
subject_tool = subject_generator.as_tool(tool_name="email_subject_generator",
                                         tool_description="Generate a subject for a given email")

html_converter = Agent(name="HTML email body converter",
                       instructions="Convert a text email body, which may contain some markdown, to a nice HTML layout.",
                       model=model)
html_tool = html_converter.as_tool(tool_name="html_body_converter",
                                   tool_description="Convert a text email body to an HTML email body")


@function_tool
def send_html_email(subject: str, html_body: str) -> dict[str, str]:
    print(f"Sending email...\n"
          f"Subject: {subject}\n"
          f"Body: {html_body}")
    return {"status": "success"}


email_agent = Agent(
    name="Email manager",
    instructions="Your task is to format and send email. You receive a text email body to be sent. \
First use subject_writer tool to generate email subject, then use html_converter tool to convert email body to HTML. \
Finally, use send_html_email tool to send the email.",
    tools=[subject_tool, html_tool, send_html_email],
    model=model,
    handoff_description="Convert an email to HTML and send it")


async def test_interviewer_with_tools_and_handoffs():
    with trace("Interviewer with tools and handoffs"):
        interviewer = Agent(
            name="Interviewer",
            instructions="You are an interviewer. You use the tools given to you to generate and sent questions to "
                         "candidates. Try 2 assistant tools to generate interview questions. The compare the questions "
                         "and pick only one the best question. The handoff the only one selected question to the Email "
                         "Manager agent to format and send the question to a candidate.",
            tools=[assistant1, assistant2],
            handoffs=[email_agent],
            model=model)
        await Runner.run(interviewer, "Send an interview question to the candidate")


asyncio.run(test_interviewer_with_tools_and_handoffs())
