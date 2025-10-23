import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr

# 🔐 Cargamos tu token desde los Secrets del Space
token = os.getenv("Token_App")

if not token:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'Token_App' en los Secrets del Space.")

# 🚀 Cargamos el modelo
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=token
).to("cpu")  # usar "cuda" si tienes GPU

# ⚙️ Función principal para generar la imagen
def decorate_room(image, prompt, strength=0.6, guidance=7.5, steps=30):
    if image is None:
        return None
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps
    ).images[0]
    return result

# 🎨 Interfaz Gradio
interface = gr.Interface(
    fn=decorate_room,
    inputs=[
        gr.Image(label="Sube la imagen de tu habitación", type="pil"),
        gr.Textbox(label="Descripción de la decoración deseada (ej: pared azul, muebles modernos...)"),
        gr.Slider(0.1, 1.0, value=0.6, step=0.1, label="Fuerza (strength)"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guía (guidance scale)"),
        gr.Slider(10, 50, value=30, step=1, label="Pasos de inferencia")
    ],
    outputs=gr.Image(label="Habitación decorada"),
    title="Decorador de Habitaciones 🏡",
    description="Sube una imagen de tu habitación y escribe cómo te gustaría decorarla. El modelo aplicará los cambios automáticamente."
)

interface.launch()
















