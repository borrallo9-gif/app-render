import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import requests
from io import BytesIO

# Tu token de Hugging Face
token = "TU_TOKEN_HF"

# Cargar el modelo
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,  # Usar float32 para CPU
    use_auth_token=token
)
pipe.to("cpu")

# Función para cargar una imagen desde una URL
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

# URL de la imagen de entrada
input_image_url = "URL_DE_TU_IMAGEN"

# Cargar la imagen
init_image = load_image(input_image_url)

# Descripción del estilo o cambios deseados
description = "Descripción detallada de la decoración que deseas aplicar."

# Parámetros de generación
guidance_scale = 7.5  # Controla la adherencia al prompt
num_inference_steps = 50  # Número de pasos de inferencia
strength = 0.75  # Fuerza de la modificación de la imagen

# Generar la imagen modificada
output = pipe(
    prompt=description,
    init_image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps
)

# Mostrar la imagen resultante
output_image = output.images[0]
output_image.show()







