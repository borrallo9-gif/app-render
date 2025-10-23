# app.py (coloca exactamente este archivo en tu Space)
import os
import traceback
import gradio as gr
from PIL import Image
import torch

# importar pipeline con try/except para mostrar errores si falta dependencia
try:
    from diffusers import StableDiffusionImg2ImgPipeline
except Exception as e:
    raise RuntimeError("Falta la librería diffusers o no está instalada correctamente: " + str(e))

# --- CONFIG ---
MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"  # modelo ligero recomendado
TOKEN_ENV_NAME = "Token_App"  # tu secret
# --- END CONFIG ---

def safe_print_logs(*args):
    print(*args, flush=True)

# Comprueba token
token = os.getenv(TOKEN_ENV_NAME)
safe_print_logs(f"DEBUG: token found? {bool(token)} (secret name: {TOKEN_ENV_NAME})")

# Intentar cargar modelo y atrapar errores para que queden en logs
pipe = None
load_error = None
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
        revision="main",
        safety_checker=None,
        use_auth_token=token  # si token es None y repo es público, funcionará igualmente
    )
    # mover a CPU/GPU según esté disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    safe_print_logs(f"DEBUG: Modelo {MODEL_ID} cargado en {device}")
except Exception as e:
    load_error = e
    safe_print_logs("ERROR cargando modelo:", e)
    traceback.print_exc()

# Función de generación con manejo de errores
def decorar_imagen(image, prompt, strength, steps, guidance):
    # Validaciones rápidas
    if load_error is not None:
        # devolver error amigable al usuario y detalles para logs
        msg = ("No se pudo cargar el modelo en este Space. "
               "Mira los logs (Settings → Logs). Error corto: " + str(load_error))
        safe_print_logs("ERROR al generar: el modelo no está cargado. load_error:", load_error)
        raise gr.Error(msg)

    if image is None:
        raise gr.Error("Por favor sube una imagen antes de generar.")

    if not prompt or prompt.strip() == "":
        raise gr.Error("Por favor escribe una descripción (prompt).")

    try:
        # Preprocesado: asegurar RGB y tamaño manejable
        img = image.convert("RGB")
        # redimensionar manteniendo aspecto (se adapta al modelo). por simplicidad: 512x512
        img = img.resize((512, 512), resample=Image.LANCZOS)

        # Limitar parámetros razonables en CPU
        device = next(pipe.parameters()).device if pipe is not None else "cpu"
        if str(device).startswith("cpu"):
            steps = min(int(steps), 25)  # límite para CPU
            safe_print_logs(f"INFO: ejecutando en CPU, forzando steps a ≤25 (usado: {steps})")
        else:
            steps = int(steps)

        # Ejecutar pipeline
        safe_print_logs(f"INFO: Generando -> prompt='{prompt[:80]}...', strength={strength}, steps={steps}, guidance={guidance}")
        out = pipe(
            prompt=prompt,
            image=img,
            strength=float(strength),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps)
        )
        result_img = out.images[0]

        return result_img

    except Exception as e:
        # imprimir traza completa en logs y devolver error claro en UI
        safe_print_logs("ERROR durante la generación:")
        traceback.print_exc()
        raise gr.Error("Ocurrió un error durante la generación. Revisa los logs del Space para la traza completa. Error corto: " + str(e))

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Decorador de habitaciones (versión CPU-friendly)\nSube una imagen, escribe lo que quieres y pulsa Generar.")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Sube la imagen base")
            prompt_box = gr.Textbox(label="Prompt (qué quieres añadir/cambiar)", lines=2)
            strength_slider = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Fuerza (strength)")
            guidance_slider = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Guidance scale")
            steps_slider = gr.Slider(5, 50, value=20, step=1, label="Pasos (num_inference_steps)")
            btn = gr.Button("✨ Generar")
        with gr.Column():
            output = gr.Image(label="Resultado")

    btn.click(fn=decorar_imagen, inputs=[input_img, prompt_box, strength_slider, steps_slider, guidance_slider], outputs=output)

if __name__ == "__main__":
    demo.launch()

















