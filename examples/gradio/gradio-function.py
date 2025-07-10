import gradio as gr


def test(text):
    return text.upper()


gr.Interface(fn=test, inputs="textbox", outputs="textbox").launch()
