import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os

# Obtener token desde los Secrets del Space
HUGGINGFACE_TOKEN = os.environ.get("Token_App")
if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'Token_App' en los Secrets del Space.")

# Cargar pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    use_auth_token=HUGGINGFACE_TOKEN
).to("cuda")  # Si no tienes GPU cambia a "cpu"

def decorar_img2img(init_image, prompt, strength=0.7, guidance_scale=7.5):
    """
    init_image: Imagen subida desde el Space
    prompt: Texto que describe la decoración
    strength: Qué tanto se altera la imagen original (0-1)
    guidance_scale: Qué tanto se sigue el prompt
    """
    if init_image is None:
        return None
    
    init_image = init_image.convert("RGB")
    
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale
        )
    
    return result.images[0]

# Interfaz Gradio
iface = gr.Interface(
    fn=decorar_img2img,
    inputs=[
        gr.Image(type="pil", label="Sube tu imagen"),
        gr.Textbox(label="Prompt de decoración"),
        gr.Slider(0,1,0.1,value=0.7,label="Fuerza de transformación"),
        gr.Slider(1,20,1,value=7.5,label="Guidance Scale")
    ],
    outputs=gr.Image(label="Imagen generada"),
    title="Decorador de Imágenes (img2img)",
    description="Sube una imagen y decórala según el prompt usando Realistic Vision V5.1_noVAE"
)

if __name__ == "__main__":
    iface.launch()



















