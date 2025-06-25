from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"


class State(TypedDict):
    messages: Annotated[list, add_messages]


client = ChatOpenAI(model=model)


def chatbot(state: State):
    return {"messages": [client.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}


# The "history" parameter is required in runtime
def chat(user_message: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
