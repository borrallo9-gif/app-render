import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler

# Modelo inpainting de ControlNet
model_id = "lllyasviel/control_v11p_sd15_inpaint"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga del pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Función de decoración con máscara
def decorar(imagen, mask, prompt, guidance_scale, num_steps):
    if imagen is None or prompt.strip() == "":
        return None
    # Limita pasos en CPU para que no se cuelgue
    if device == "cpu":
        num_steps = min(num_steps, 25)  # CPU más lenta
    resultado = pipe(
        prompt=prompt,
        image=imagen,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps
    ).images[0]
    return resultado

# Título y descripción
titulo = "Decorador de habitaciones con IA (ControlNet Inpainting)"
descripcion = (
    "Sube una foto de tu habitación vacía y pinta de blanco la zona que quieras decorar.\n"
    "Describe la decoración que deseas en el prompt.\n"
    "Ejemplo: 'Añade un sofá gris moderno y un cuadro encima en la pared blanca'.\n\n"
    "Optimizado para CPU gratuita: generación rápida usando pasos limitados."
)

# Interfaz Gradio con sliders actualizados
demo = gr.Interface(
    fn=decorar,
    inputs=[
        gr.Image(type="pil", label="Sube tu habitación"),
        gr.Image(type="pil", label="Pinta en blanco la zona a editar (máscara)"),
        gr.Textbox(label="Describe la decoración que deseas"),
        gr.Slider(1, 15, value=9, label="Nivel de detalle (guidance scale)"),
        gr.Slider(5, 50, value=25, step=1, label="Pasos de inferencia")
    ],
    outputs=gr.Image(label="Habitación decorada"),
    title=titulo,
    description=descripcion
)

demo.launch()



