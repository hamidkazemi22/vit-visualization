import gradio as gr

from visualize import visualize_feature

iface = gr.Interface(fn=visualize_feature,
                     inputs=[gr.Number(label="Layer", value=6), gr.Number(label="Feature", value=4)],
                     outputs=gr.Image(type="pil"))
iface.launch(share=True)
