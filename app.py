import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
import traceback

# Cargar token desde los Secrets del Space
token = os.getenv("HUGGINGFACE_TOKEN")

# Verificar el token
if not token:
    raise ValueError("❌ No se encontró el token. Asegúrate de tener 'HUGGINGFACE_TOKEN' en los Secrets del Space.")

# Cargar el modelo
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        use_auth_token=token
    )
except Exception as e:
    raise RuntimeError(f"❌ Error al cargar el modelo: {e}")

# Usar CPU (importante en Spaces sin GPU)
pipe = pipe.to("cpu")

# Función principal
def generar_imagen(prompt, imagen, fuerza=0.6, pasos=35):
    try:
        if imagen is None:
            raise gr.Error("⚠️ Por favor, sube una imagen base antes de generar.")
        imagen = imagen.convert("RGB")
        resultado = pipe(
            prompt=prompt,
            image=imagen,
            strength=fuerza,
            num_inference_steps=int(pasos)
        ).images[0]
        return resultado
    except Exception as e:
        print("⚠️ Error durante la generación de imagen:", e)
        traceback.print_exc()
        raise gr.Error(f"Ocurrió un error durante la generación: {e}")

# Interfaz
demo = gr.Interface(
    fn=generar_imagen,
    inputs=[
        gr.Textbox(label="Descripción (prompt):", placeholder="Ej: habitación moderna con tonos azules"),
        gr.Image(label="Sube una imagen base"),
        gr.Slider(0, 1, value=0.6, step=0.05, label="Intensidad del cambio (strength)"),
        gr.Slider(10, 50, value=35, step=5, label="Pasos de inferencia")
    ],
    outputs=gr.Image(label="Resultado"),
    title="Decorador de imágenes con Stable Diffusion v1.5",
    description="Sube una imagen y escribe cómo quieres modificarla. Ejemplo: 'versión futurista del salón con luces LED'."
)

if __name__ == "__main__":
    demo.launch()















