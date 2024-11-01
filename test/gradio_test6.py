import random
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        clear_button = gr.Button("Clear")
        skip_button = gr.Button("Skip")
        random_button = gr.Button("Random")
    numbers = [gr.Number(), gr.Number()]

    clear_button.click(lambda: (None, None), outputs=numbers) # None会将numbers的值置空
    skip_button.click(lambda: [gr.skip(), gr.skip()], outputs=numbers)  # gr.skip()不会改变numbers当前的值(即跳过)
    random_button.click(lambda: (random.randint(0, 100), random.randint(0, 100)), outputs=numbers)

demo.launch()
