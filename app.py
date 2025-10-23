import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import io
import os

# Cargar el token desde los Secrets
HF_TOKEN = os.environ.get("Token_App")
if not HF_TOKEN:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'Token_App' en los Secrets del Space.")

# Cargar el modelo (img2img, más ligero que XL)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def decorate_image(init_image, prompt, strength=0.75, steps=50):
    if init_image is None:
        return None

    # Detectar el dispositivo
    device = pipe.device
    if str(device).startswith("cpu"):
        steps = min(int(steps), 25)  # límite para CPU

    # Asegurarse de que la imagen esté en RGB
    if init_image.mode != "RGB":
        init_image = init_image.convert("RGB")

    # Generar la imagen
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        result = pipe(prompt=prompt, init_image=init_image, strength=strength, num_inference_steps=steps)

    return result.images[0]

# Interfaz Gradio
title = "Decorador de imágenes - Img2Img"
description = "Sube tu imagen y genera una versión decorada usando Stable Diffusion v1-5."

iface = gr.Interface(
    fn=decorate_image,
    inputs=[
        gr.Image(type="pil", label="Imagen inicial"),
        gr.Textbox(label="Prompt de decoración"),
        gr.Slider(0.0, 1.0, value=0.75, label="Fuerza (strength)"),
        gr.Slider(1, 50, value=25, step=1, label="Número de pasos (steps)")
    ],
    outputs=gr.Image(type="pil", label="Imagen generada"),
    title=title,
    description=description,
)

if __name__ == "__main__":
    iface.launch()


















