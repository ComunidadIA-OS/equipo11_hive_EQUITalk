# Usa una versión estable y compatible con PyTorch/Pyannote
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza salida por consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Los modelos de Hugging Face se guardan aquí (montado como volumen para descargarlos una sola vez)
ENV HF_HOME=/app/.cache/huggingface
# Configuración de la demo: accesible desde fuera del contenedor y sin estadísticas de uso
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Instala dependencias del sistema necesarias para procesamiento de audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Versión de la imagen: "cpu" (por defecto, cualquier ordenador) o "gpu" (GPU NVIDIA).
#   CPU: docker build -t audio-ai-processor .
#   GPU: docker build --build-arg DEVICE=gpu -t audio-ai-processor-gpu .
ARG DEVICE=cpu

# Instala dependencias de Python.
# - cpu: índice de PyTorch CPU (el mismo que usa GitHub Actions); imagen mucho más ligera.
# - gpu: PyTorch de PyPI, que ya trae CUDA. faster-whisper necesita encontrar las librerías CUDA
#        (cuBLAS, cuDNN) que instala pip, así que se registran en el sistema con ldconfig.
COPY requirements.txt .
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir -r requirements.txt \
        && find / -type d -path "*site-packages/nvidia/*/lib" > /etc/ld.so.conf.d/nvidia-pip.conf \
        && ldconfig; \
    else \
        pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copia el código fuente principal y la demo
COPY src/ ./src/
COPY main.py app.py ./
COPY ejemplos/ ./ejemplos/

# Puerto de la demo (http://localhost:7860)
EXPOSE 7860

# main.py es el punto de entrada: a "docker run <imagen>" solo se le pasan sus argumentos (ver README).
# Sin argumentos muestra la ayuda. La demo se arranca con docker compose (ver docker-compose.yml).
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
