# Usa una versión estable y compatible con PyTorch/Pyannote
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza salida por consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias del sistema necesarias para procesamiento de audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente principal
COPY src/ ./src/
COPY main.py .

# Ejecuta el script principal por defecto
CMD ["python", "main.py"]