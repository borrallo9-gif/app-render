import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

# Modelo img2img público
model_id = "runwayml/stable-diffusion-v1-5-img2img"

# Detecta si hay GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga del pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Función de decoración
def decorar(imagen, prompt, strength, guidance_scale, num_steps):
    if imagen is None or prompt.strip() == "":
        return None

    # Limita los pasos en CPU para no colgar el Space
    if device == "cpu":
        num_steps = min(num_steps, 25)

    resultado = pipe(
        prompt=prompt,
        init_image=imagen,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps
    ).images[0]

    return resultado

# Título y descripción de la app
titulo = "Decorador de habitaciones con IA (img2img)"
descripcion = (
    "Sube una foto de tu habitación vacía y escribe cómo quieres decorarla.\n"
    "Ejemplo: 'Añade un sofá gris moderno junto a la pared derecha y un cuadro encima'.\n\n"
    "Optimizado para CPU gratuita: generación rápida usando pasos limitados."
)

# Interfaz Gradio
demo = gr.Interface(
    fn=decorar,
    inputs=[
        gr.Image(type="pil", label="Sube tu habitación"),
        gr.Textbox(label="Describe la decoración que deseas"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Fuerza de cambios (strength)"),
        gr.Slider(1, 15, value=9, step=1, label="Nivel de detalle (guidance scale)"),
        gr.Slider(5, 50, value=20, step=1, label="Pasos de inferencia")
    ],
    outputs=gr.Image(label="Habitación decorada"),
    title=titulo,
    description=descripcion
)

demo.launch()






