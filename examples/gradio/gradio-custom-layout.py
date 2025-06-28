import random
import string

import gradio as gr


def random_string(max_length):
    if not max_length:
        return "Enter max length"
    try:
        length = int(max_length)
        if length <= 0:
            return "Enter positive value"
        return "".join(random.choices(string.ascii_letters, k=int(max_length)))
    except ValueError:
        return "Enter integer value"


async def reset():
    return "", None


async def generate(max_length, history):
    # Here we could call LLM
    return history + [
        {"role": "user", "content": f"Generate a random string of length {max_length}"},
        {"role": "assistant", "content": random_string(max_length)}]


with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Random string generator")

    with gr.Row():
        chatbot = gr.Chatbot(label="History", height=300, type="messages")
    with gr.Row():
        max_length = gr.Textbox(show_label=False, placeholder="Max length")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        generate_button = gr.Button("Generate", variant="primary")
    reset_button.click(reset, [], [max_length, chatbot])
    generate_button.click(generate, [max_length, chatbot], [chatbot])

demo.launch()
