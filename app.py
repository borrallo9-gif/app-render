import gradio as gr
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import os

# Token guardado en secrets del Space
HF_TOKEN = os.environ.get("HF_TOKEN")

# Cargar la pipeline correcta: XL Img2Img
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",  # modelo refinador XL que soporta img2img
    torch_dtype=torch.float32,
    use_auth_token=HF_TOKEN
)
pipe.to("cpu")

def decorar_imagen(init_image, prompt, steps=25, strength=0.6):
    """
    init_image: PIL.Image subida por el usuario
    prompt: texto para la decoración
    """
    init_image = init_image.convert("RGB")
    result = pipe(
        prompt=prompt,
        image=init_image,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=7.5
    ).images[0]
    return result

# Interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🪄 Decorador de imágenes con Stable Diffusion XL (img2img)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Sube tu imagen base")
            prompt_text = gr.Textbox(label="Describe la decoración o estilo")
            steps_slider = gr.Slider(1, 50, value=25, step=1, label="Pasos (steps)")
            strength_slider = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Fuerza de transformación")
            submit_btn = gr.Button("Decorar imagen ✨")
        with gr.Column():
            output_image = gr.Image(label="Resultado decorado")

    submit_btn.click(
        decorar_imagen,
        inputs=[input_image, prompt_text, steps_slider, strength_slider],
        outputs=output_image
    )

demo.launch()












