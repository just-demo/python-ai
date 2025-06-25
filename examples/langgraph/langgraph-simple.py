from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
import gradio as gr

load_dotenv(override=True)

model = "gpt-4.1-nano"


class State(BaseModel):
    messages: Annotated[list, add_messages]


# Could use any other way to communicate with LLM, e.g. native OpenAI client or OpenAI Agents SDK
client = ChatOpenAI(model=model)


def demo_node(old_state: State) -> State:
    # Don't really need to communicate with LLM here, e.g. could return a dummy result
    response = client.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state


graph_builder = StateGraph(State)
graph_builder.add_node("demo", demo_node)
graph_builder.add_edge(START, "demo")
graph_builder.add_edge("demo", END)

graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))


def chat(user_input: str, history):
    print(history)
    state = State(messages=[{"role": "user", "content": user_input}])
    result = graph.invoke(state)
    print(result)
    return result['messages'][-1].content


gr.ChatInterface(chat, type="messages").launch()
