import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import gradio as gr

load_dotenv(override=True)

embedding_model = "text-embedding-3-small"
chat_model = "gpt-4.1-nano"
db_name = "rag_db"


def generate_documents():
    documents = [
        "JustDemo company is for educational purposes only.",
        "It has only one contributor."
    ]
    return [Document(page_content=text) for text in documents]


embedding = OpenAIEmbeddings(model=embedding_model)
vectorstore = Chroma(persist_directory=db_name, embedding_function=embedding) \
    if os.path.exists(db_name) \
    else Chroma.from_documents(documents=generate_documents(), embedding=embedding, persist_directory=db_name)

llm = ChatOpenAI(temperature=0.7, model_name=chat_model)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()
# conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
# Use the callback to print requests to llm
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,
                                                           callbacks=[StdOutCallbackHandler()])

result = conversation_chain.invoke({"question": "Is JustDemo suitable for commercial use?"})
print(result["answer"])


def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


gr.ChatInterface(chat, type="messages").launch()
