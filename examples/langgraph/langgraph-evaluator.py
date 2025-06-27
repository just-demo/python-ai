from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

model = "gpt-4.1-nano"


class Evaluation(BaseModel):
    acceptable: bool = Field(description="Whether the answer is acceptable")
    score: int = Field(description="Score on the answer from 0 to 100")
    feedback: str = Field(description="Feedback on the answer")


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    evaluation: Evaluation


answerer_client = ChatOpenAI(model=model)
evaluator_client = ChatOpenAI(model=model).with_structured_output(Evaluation)


def answerer(state: State) -> Dict[str, Any]:
    messages = state.get("messages")
    evaluation = state.get("evaluation")
    if evaluation and not evaluation.acceptable:
        messages += [SystemMessage(content=f"Your previous answer was rejected: {evaluation.feedback}")]
    answer = answerer_client.invoke(messages)
    print(f"Answer: {answer}")
    return {"messages": [answer]}


def get_formatted_messages(state: State) -> str:
    formatted_messages = []
    for message in state['messages']:
        if isinstance(message, HumanMessage):
            formatted_messages.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_messages.append(f"Assistant: {message.content}")
    return "\n".join(formatted_messages)


def get_last_message_of_type(state: State, message_type) -> str:
    return next((message.content for message in reversed(state["messages"]) if isinstance(message, message_type)), None)


def evaluator(state: State) -> State:
    system_message = "Evaluate the answer, replying with whether it is acceptable, your feedback and score."
    user_message = f"Here is the full conversation: \n\n{get_formatted_messages(state)}\n\n" \
                   f"Here is the question: \n\n{get_last_message_of_type(state, HumanMessage)}\n\n" \
                   f"Here is the answer: \n\n{get_last_message_of_type(state, AIMessage)}\n\n"
    evaluation = evaluator_client.invoke([SystemMessage(content=system_message), HumanMessage(content=user_message)])
    print(f"Evaluation: {evaluation}")
    return {
        "messages": [],
        "evaluation": evaluation}


def evaluation_router(state: State) -> str:
    return "END" if state["evaluation"].acceptable else "answerer"


graph_builder = StateGraph(State)
graph_builder.add_node("answerer", answerer)
graph_builder.add_node("evaluator", evaluator)
graph_builder.add_edge(START, "answerer")
graph_builder.add_edge("answerer", "evaluator")
graph_builder.add_conditional_edges("evaluator", evaluation_router, {"answerer": "answerer", "END": END})

graph = graph_builder.compile(checkpointer=MemorySaver())
print(graph.get_graph().draw_mermaid())

config = {"configurable": {"thread_id": "1"}}


# The history parameter is required in runtime
def chat(user_message: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
