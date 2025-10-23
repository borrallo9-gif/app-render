import gradio as gr
import torch
import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# === CONFIGURACIÓN DEL TOKEN (usa tu Secret 'Token_App') ===
token = os.getenv("Token_App")
if not token:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'Token_App' en los Secrets del Space.")

# === CARGAR MODELO LIGERO ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float32,
    use_safetensors=True,
    revision="main",
    safety_checker=None,
    requires_safety_checker=False,
    token=token
)

# Usar CPU
pipe = pipe.to("cpu")

# === FUNCIÓN PARA PROCESAR LA IMAGEN ===
def decorar_imagen(image, prompt, strength=0.6, steps=20):
    if image is None:
        return "⚠️ Sube una imagen antes de continuar."

    # Reducir resolución a 512x512 para acelerar el proceso
    image = image.convert("RGB")
    image = image.resize((512, 512))

    # Generar imagen decorada
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=7.5
    ).images[0]

    return result

# === INTERFAZ GRADIO ===
title = "🖼️ Decorador Inteligente - Versión CPU Optimizada"
description = """
Sube una imagen (por ejemplo, una habitación o fachada) y describe cómo quieres decorarla o modificarla.
Ejemplo: "Paredes blancas, estilo moderno con plantas y luz natural".
"""

iface = gr.Interface(
    fn=decorar_imagen,
    inputs=[
        gr.Image(label="Sube tu imagen"),
        gr.Textbox(label="Descripción (prompt)", placeholder="Ejemplo: paredes blancas y decoración moderna"),
        gr.Slider(0.1, 1.0, value=0.6, label="Fuerza del cambio (strength)"),
        gr.Slider(5, 50, value=20, step=1, label="Pasos de inferencia (steps)")
    ],
    outputs=gr.Image(label="Imagen Decorada"),
    title=title,
    description=description,
    allow_flagging="never"
)

iface.launch()

















