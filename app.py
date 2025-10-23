import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr
import os

# Carga del pipeline con tu token desde los "Secrets" del Space
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "Lykon/dreamshaper-7",
    torch_dtype=torch.float16,
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN")
).to("cuda" if torch.cuda.is_available() else "cpu")


def generar_imagen(prompt, image, strength, guidance):
    """
    Genera una imagen decorada a partir de una imagen subida y un prompt de descripción.
    """
    if image is None:
        return "Por favor, sube una imagen."

    # Convierte la imagen a RGB y asegura compatibilidad
    init_image = image.convert("RGB")

    # Genera imagen a partir del prompt y la imagen base
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,        # Cuánto cambia respecto a la imagen original (0-1)
        guidance_scale=guidance,  # Qué tanto sigue el texto
        num_inference_steps=30    # Puedes ajustar para más calidad (pero más lento)
    ).images[0]

    return result


# Interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Decorador de Imagen con DreamShaper-7 (Img2Img)")
    gr.Markdown("Sube una imagen base y describe cómo quieres decorarla o transformarla.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Descripción (prompt)", placeholder="Ej: una versión mágica con luces de neón y fondo estrellado")
            image = gr.Image(label="Sube tu imagen base", type="pil")
            strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Intensidad del cambio (strength)")
            guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Guía del texto (guidance)")
            btn = gr.Button("✨ Generar imagen")

        with gr.Column():
            output = gr.Image(label="Imagen generada")

    btn.click(fn=generar_imagen, inputs=[prompt, image, strength, guidance], outputs=output)

# Lanza la app
if __name__ == "__main__":
    demo.launch()














