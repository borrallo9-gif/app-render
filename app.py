import gradio as gr
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
import os

# Cargar el token desde Secrets
HF_TOKEN = os.environ.get("HF_TOKEN")  # Asegúrate de haberlo guardado como "HF_TOKEN" en tu Space

# Cargar pipeline en CPU
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_auth_token=HF_TOKEN
)
pipe.to("cpu")  # CPU

def decorar_imagen(init_image, prompt, steps=25, strength=0.6):
    """
    init_image: PIL.Image subido en Gradio
    prompt: texto que describe la decoración
    steps: número de steps del modelo
    strength: cuánto transformar la imagen original (0-1)
    """
    # Asegurar que la imagen esté en RGB
    init_image = init_image.convert("RGB")

    # Generar imagen decorada
    result = pipe(prompt=prompt, image=init_image, num_inference_steps=steps, strength=strength).images[0]
    return result

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Decorador de imágenes con Stable Diffusion XL")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Sube tu imagen")
            prompt_text = gr.Textbox(label="Describe la decoración que quieres")
            steps_slider = gr.Slider(1, 50, value=25, step=1, label="Steps")
            strength_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Fuerza de transformación")
            submit_btn = gr.Button("Decorar")
        with gr.Column():
            output_image = gr.Image(label="Imagen decorada")

    submit_btn.click(
        decorar_imagen,
        inputs=[input_image, prompt_text, steps_slider, strength_slider],
        outputs=output_image
    )

demo.launch()











