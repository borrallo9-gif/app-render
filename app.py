import os
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline

# Tomar token desde los Secrets de Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")

# Cargar modelo SDXL (CPU)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    revision="fp16",
    use_auth_token=HF_TOKEN
)
pipe.to("cpu")

# Cargar imagen local (reemplaza con tu archivo)
init_image = Image.open("ruta/a/tu/imagen.jpg").convert("RGB")

# Prompt de decoración
prompt = "Añadir un sofá gris en la pared de la derecha y un cuadro moderno encima."

# Parámetros de generación
num_inference_steps = 50
guidance_scale = 7.5
strength = 0.75  # intensidad del cambio sobre la imagen original

# Generar imagen
output = pipe(
    prompt=prompt,
    init_image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps
)

# Guardar resultado
output_image = output.images[0]
output_image.save("resultado.png")
output_image.show()









