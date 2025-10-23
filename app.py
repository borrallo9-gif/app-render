import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

# Cargar el token desde los secrets del Space
token = os.getenv("HUGGINGFACE_TOKEN")

# Cargar el modelo desde Hugging Face
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_auth_token=token
)

# Usar CPU (para Spaces sin GPU)
pipe = pipe.to("cpu")

# Función principal: recibe imagen y prompt, devuelve imagen modificada
def generar_imagen(prompt, imagen, fuerza=0.6, pasos=35):
    if imagen is None:
        raise gr.Error("Por favor, sube una imagen base.")
    imagen = imagen.convert("RGB")
    resultado = pipe(
        prompt=prompt,
        image=imagen,
        strength=fuerza,     # Cuánto cambia respecto a la imagen base (0 = igual, 1 = muy distinto)
        num_inference_steps=pasos
    ).images[0]
    return resultado

# Interfaz de Gradio
demo = gr.Interface(
    fn=generar_imagen,
    inputs=[
        gr.Textbox(label="Describe cómo quieres modificar la imagen (prompt):"),
        gr.Image(label="Sube una imagen base"),
        gr.Slider(0, 1, value=0.6, step=0.05, label="Intensidad del cambio (strength)"),
        gr.Slider(10, 50, value=35, step=5, label="Número de pasos de inferencia")
    ],
    outputs=gr.Image(label="Imagen generada"),
    title="Decorador de Imágenes - Stable Diffusion v1.5",
    description="Sube una imagen y describe cómo quieres que cambie (por ejemplo: 'versión futurista de la habitación')."
)

if __name__ == "__main__":
    demo.launch()















