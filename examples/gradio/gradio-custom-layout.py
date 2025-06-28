import random
import string

import gradio as gr


def generate_random_value(value_type):
    if value_type == "integer":
        return random.randint(0, int(1e9))
    elif value_type == "string":
        return ''.join(random.choices(string.ascii_letters, k=10))
    else:
        return f"Select value type"


async def reset():
    return "", None


async def process_message(value_type, history):
    # Here we could call LLM
    return history + [
        {"role": "user", "content": f"Generate random {value_type}"},
        {"role": "assistant", "content": str(generate_random_value(value_type))}]


with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Random value generator")

    with gr.Row():
        chatbot = gr.Chatbot(label="History", height=300, type="messages")
    with gr.Row():
        value_type = gr.Dropdown(["", "integer", "string"], label="Value type")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        submit_button = gr.Button("Generate", variant="primary")
    reset_button.click(reset, [], [value_type, chatbot])
    submit_button.click(process_message, [value_type, chatbot], [chatbot])

demo.launch()
