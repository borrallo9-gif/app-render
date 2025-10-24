import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os

# Obtener token desde Render
HUGGINGFACE_TOKEN = os.environ.get("App_Token")
if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'App_Token' en Render.")

# Detectar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Función principal
def decorar_img2img(init_image, prompt, strength=0.7, guidance_scale=7.5):
    if init_image is None:
        return None
    init_image = init_image.convert("RGB")
    
    if device == "cuda":
        with torch.autocast("cuda"):
            result = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)
    else:
        result = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)

    return result.images[0]

# Interfaz Gradio
iface = gr.Interface(
    fn=decorar_img2img,
    inputs=[
        gr.Image(type="pil", label="Sube tu imagen base"),
        gr.Textbox(label="Prompt de decoración"),
        gr.Slider(0, 1, 0.1, value=0.7, label="Fuerza de transformación"),
        gr.Slider(1, 20, 1, value=7.5, label="Guidance Scale")
    ],
    outputs=gr.Image(label="Imagen generada"),
    title="Decorador de Imágenes (img2img)",
    description="Sube una imagen y decórala según el prompt usando Realistic Vision V5.1_noVAE"
)

# Lanzar servidor compatible con Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port, share=False)



















