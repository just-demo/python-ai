from typing import Annotated, TypedDict
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"


def record_user_email(email: str):
    print(f"Recording user email = {email}")


tool_record_user_email = Tool(
    name="record_user_email",
    func=record_user_email,
    description="Record user email"
)

tools = [tool_record_user_email]


class State(TypedDict):
    messages: Annotated[list, add_messages]


client = ChatOpenAI(model=model)
client_with_tools = client.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [client_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()


# The history parameter is required in runtime
def chat(user_message: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
# Send a message list "my email is test@test.com!"
