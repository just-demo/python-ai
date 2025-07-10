import gradio as gr


def chat(message, history):
    print(history)
    return message.upper()


gr.ChatInterface(chat, type="messages").launch()
