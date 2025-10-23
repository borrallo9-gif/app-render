import gradio as gr
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Modelo que vas a usar
model_id = "timbrooks/instruct-pix2pix"

# Detecta si hay GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga el modelo
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Función de decoración optimizada para CPU
def decorar(imagen, prompt, guidance_scale, num_steps):
    if imagen is None or prompt.strip() == "":
        return None
    # Limita los pasos para CPU gratuita
    if device == "cpu":
        num_steps = min(num_steps, 15)  # máximo 15 pasos en CPU
    resultado = pipe(
        prompt,
        image=imagen,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps
    ).images[0]
    return resultado

# Título y descripción de la app
titulo = "Decorador de habitaciones con IA"
descripcion = (
    "Sube una foto de tu habitación vacía y describe cómo quieres decorarla.\n"
    "Ejemplo: 'Añade sofá gris y planta verde junto a la ventana'.\n\n"
    "Optimizado para CPU gratuita: la generación puede tardar 30–60 s."
)

# Interfaz Gradio con sliders originales
demo = gr.Interface(
    fn=decorar,
    inputs=[
        gr.Image(type="pil", label="Sube tu habitación"),
        gr.Textbox(label="Describe la decoración que deseas"),
        gr.Slider(1, 10, value=7.5, label="Nivel de detalle (guidance scale)"),
        gr.Slider(5, 50, value=20, step=1, label="Pasos de inferencia")
    ],
    outputs=gr.Image(label="Habitación decorada"),
    title=titulo,
    description=descripcion
)

demo.launch()


