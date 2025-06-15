from dotenv import load_dotenv
from openai import OpenAI
import json
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"
client = OpenAI()


def record_user_email(email, name=None, extra=None):
    print(f"Recording user: email = {email}, name = {name}, extra = {extra}")


def record_user_phone(phone, name=None, extra=None):
    print(f"Recording user: phone = {phone}, name = {name}, extra = {extra}")


def record_unknown_question(question):
    print(f"Recording unknown question: {question}")


schema_record_user_email = {
    "name": "record_user_email",
    "description": "Tool to record user email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User email address"},
            "name": {"type": "string", "description": "User name"},
            "extra": {"type": "string", "description": "Any extra details, like preferable time to contact"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

schema_record_user_phone = {
    "name": "record_user_phone",
    "description": "Tool to record user phone number",
    "parameters": {
        "type": "object",
        "properties": {
            "phone": {"type": "string", "description": "User phone number"},
            "name": {"type": "string", "description": "User name"},
            "extra": {"type": "string", "description": "Any extra details, like preferable time to contact"}
        },
        "required": ["phone"],
        "additionalProperties": False
    }
}

schema_record_unknown_question = {
    "name": "record_unknown_question",
    "description": "Tool to record unknown question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Unknown question"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": schema_record_user_email},
         {"type": "function", "function": schema_record_user_phone},
         {"type": "function", "function": schema_record_unknown_question}]


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}")
        too_function = globals().get(tool_name)
        result = {}
        if too_function:
            too_function(**arguments)
            result = {"recorded": "ok"}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results


system_prompt = "Your task is to answer questions."


def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    while True:
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            tool_calls_results = handle_tool_calls(choice.message.tool_calls)
            messages.append(choice.message)
            messages.extend(tool_calls_results)
        else:
            return choice.message.content


gr.ChatInterface(chat, type="messages").launch()
