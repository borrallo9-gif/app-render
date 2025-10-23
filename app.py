from diffusers import StableDiffusionImg2ImgPipeline
import torch, os, gradio as gr
from PIL import Image

# Cargar el modelo DreamShaper (ligero y compatible con CPU)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "Lykon/DreamShaper-v7",
    torch_dtype=torch.float32,
    use_auth_token=os.environ.get("HF_TOKEN")
)
pipe.to("cpu")

# Función principal de decoración
def decorate_room(init_image, prompt, strength=0.6, steps=15):
    if init_image is None:
        return "Por favor sube una imagen."
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=steps,
    ).images[0]
    return result

# Interfaz Gradio
demo = gr.Interface(
    fn=decorate_room,
    inputs=[
        gr.Image(type="pil", label="Sube la imagen de la habitación"),
        gr.Textbox(label="Descripción: ¿Cómo quieres decorarla?"),
        gr.Slider(0.1, 1.0, 0.6, label="Fuerza del cambio (strength)"),
        gr.Slider(5, 30, 15, step=1, label="Pasos de inferencia (steps)"),
    ],
    outputs=gr.Image(label="Habitación decorada"),
    title="🪄 Decorador AI con DreamShaper-v7",
    description="Sube una foto y describe cómo quieres decorarla. Ejemplo: 'Paredes beige, sofá moderno, cuadro minimalista'.",
)

if __name__ == "__main__":
    demo.launch()













