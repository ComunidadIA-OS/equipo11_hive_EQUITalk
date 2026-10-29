# THE HIVE (Equipo 11) – Detector de Interrupciones

Sistema en Python que analiza grabaciones de voz para **transcribir, identificar quién habla y cuándo, inferir el género de cada hablante y medir la equidad de la conversación** (tiempo de habla, interrupciones, turnos). Opcionalmente compara los resultados del modelo contra transcripciones de referencia (*Ground Truth*, `.trs`) para cuantificar su rendimiento.

---

## 1. ¿Qué hace el sistema?

El flujo tiene dos etapas, orquestadas desde `main.py`:

```text
 Audios (.wav)                                         Ground Truth (.trs / .xml)
      │                                                          │
      ▼                                                          │ (opcional)
┌───────────────────────────── 1. INFERENCIA (GPU) ─────────┐    │
│  Faster Whisper → transcripción (ASR)                     │    │
│  Pyannote 3.1   → diarización (quién habla y cuándo)      │    │
│  Wav2Vec2       → clasificación de género por hablante    │    │
└───────────────────────────┬───────────────────────────────┘    │
                            ▼                                    ▼
                    Predicciones por audio ──────► 2. EVALUACIÓN ◄┘
                                                          │
                                                          ▼
                                       Excel + JSON con métricas
```

1. **Inferencia** (`src/pipeline/`): procesa cada audio en paralelo, repartiendo el trabajo entre las GPUs disponibles.
2. **Evaluación** (`src/evaluation/`, `src/metrics/`): calcula las métricas de equidad a partir de las predicciones y, si se proporciona Ground Truth, también las métricas de rendimiento de la IA.

Si no se indica --gt_dir, el sistema generará las transcripciones, los perfiles de hardware (tiempo, RAM, GPU) y las métricas de equidad, pero omitirá las métricas de precisión (WER, DER) al no tener transcripciones humanas de referencia.

### Modelos utilizados

| Tarea | Modelo / librería |
|---|---|
| Transcripción (ASR) | Faster Whisper |
| Diarización | `pyannote/speaker-diarization-3.1` + `pyannote/segmentation-3.0` |
| Género | `audeering/wav2vec2-large-robust-24-ft-age-gender` |

### Métricas generadas

**Rendimiento de la IA** (`PerformanceResult`, requiere Ground Truth)
- **ASR:** WER (*Word Error Rate*) y CER (*Character Error Rate*).
- **Diarización:** DER (*Diarization Error Rate*), desglosado en *Missed Detection*, *False Alarm* y *Confusion*.
- **Género:** precisión (*accuracy*), sensibilidad (*recall*) y cobertura.

**Equidad e interacciones** (`EquityMetrics`)
- **Tiempo de habla:** porcentaje por género y solapamientos.
- **Interrupciones:** cortes detectados (`male_over_female`, `female_over_male`) e índice de asimetría.
- **Turnos y participación:** monólogos más largos, brecha de participación y ratio de toma de palabra (transiciones).

**Recursos** (`TimingMetrics`)
- Tiempo de ejecución por etapa (ASR, diarización, género).
- RTF (*Real Time Factor*).
- Picos de memoria RAM y VRAM (GPU).

### Estructura del proyecto

```text
.
├── src/
│   ├── evaluation/   # Validación contra Ground Truth (evaluator.py, batch_evaluator.py)
│   ├── metrics/      # Cálculo de rendimiento IA y equidad (performance.py, equity.py)
│   ├── pipeline/     # Procesamiento de audio, inferencia y paralelismo (audio_processor.py, batch_runner.py)
│   └── utils/        # Parsers (XML/TRS), logger, configuración, rutas (paths.py) y excepciones
├── tests/            # Pruebas unitarias e integración (pytest)
├── main.py           # Orquestador principal (CLI)
├── pytest.ini        # Configuración de pruebas
├── requirements.txt  # Dependencias
├── Dockerfile        # Contenedorización
└── .dockerignore     # Archivos excluidos de la imagen Docker
```

---

## 2. Requisitos previos

- **Python 3.11** (la misma versión que usa la imagen Docker).
- **GPU NVIDIA con CUDA** para la inferencia (recomendado). Para Docker se necesita además el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **ffmpeg** instalado en el sistema (solo para instalación local; la imagen Docker ya lo incluye).
- **Cuenta en Hugging Face** con token de acceso.

### Acceso a los modelos de Hugging Face

Inicia sesión en Hugging Face y **acepta las condiciones de uso** de estos modelos:

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)

Luego crea un archivo `.env` en la raíz del proyecto con tu token (sin comillas):

```text
HF_TOKEN=tu_token_aqui
```

---

## 3. Instalación local

**1. Crear y activar un entorno virtual**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

**2. Instalar dependencias**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Cargar las variables de entorno**

```bash
# Linux / macOS
export $(grep -v '^#' .env | xargs)
```

```powershell
# Windows (PowerShell)
Get-Content .env | Where-Object { $_ -match '=' -and -not $_.StartsWith('#') } | ForEach-Object {
    $name, $value = $_.Split('=', 2)
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
}
```

---

## 4. Uso

Todo se ejecuta con `main.py`. Puedes correr inferencia y evaluación juntas o por separado.

### Parámetros

| Parámetro | Requerido | Por defecto | Descripción |
|---|---|---|---|
| `--audio_dir` | Sí, salvo con `--skip_inference` | – | Directorio con los audios (`.wav`) a procesar. |
| `--gt_dir` | No | – | Directorio del Ground Truth (`.trs` o `.xml`). Si se omite, solo se evalúa equidad. |
| `--out_dir` | No | `data/processed` | Directorio raíz de resultados. |
| `--run_name` | No | `evaluation_run` | Nombre de la ejecución; crea la carpeta `<out_dir>/<run_name>/`. |
| `--num_gpus` | No | `2` | Número de GPUs a usar en la inferencia. |
| `--skip_inference` | No | `False` | Omite la inferencia y evalúa con las predicciones ya existentes en la carpeta del run. |

### Ejemplos

**1. Pipeline completo (inferencia + evaluación)**

```bash
python main.py --audio_dir data/raw/audio --gt_dir data/ground_truth/trs
```

**2. Ejecución personalizada**

```bash
python main.py \
    --audio_dir data/raw/agora \
    --gt_dir data/ground_truth/agora \
    --out_dir data/processed \
    --run_name metricas_agora_v1 \
    --num_gpus 2
```

**3. Solo equidad (sin Ground Truth)**

```bash
python main.py --audio_dir data/raw/audio --run_name solo_equidad
```

**4. Evaluación rápida (saltar inferencia)**

Recalcula las métricas a partir de predicciones generadas previamente. Usa el **mismo** `--out_dir` y `--run_name` de la ejecución original, ya que de ahí se leen las predicciones.

```bash
python main.py \
    --gt_dir data/raw/trs \
    --run_name metricas_agora_v1 \
    --skip_inference
```

> Con `--skip_inference` no hace falta `--audio_dir`.

### Resultados

Cada ejecución crea `<out_dir>/<run_name>/`, que contiene las predicciones por audio y los reportes de métricas en **Excel y JSON** (ver [Métricas generadas](#métricas-generadas)).

---

## 5. Ejecución con Docker

La imagen define `main.py` como *entrypoint*, por lo que **solo debes pasar los argumentos** del programa (sin escribir `python main.py`).

**1. Construir la imagen**

```bash
docker build -t audio-ai-processor .
```

**2. Ver la ayuda**

```bash
docker run --rm audio-ai-processor
```

**3. Ejecutar el pipeline completo**

Los volúmenes (`-v`) conectan tus carpetas locales con el contenedor. El volumen `hf_cache` guarda los modelos descargados para no volver a bajarlos en cada ejecución.

```bash
docker run --rm -it --gpus all --shm-size=2g \
  --env-file .env \
  -v $(pwd)/data/audio:/app/audios \
  -v $(pwd)/data/trs:/app/gt \
  -v $(pwd)/data/procesado:/app/resultados \
  -v hf_cache:/app/.cache/huggingface \
  audio-ai-processor \
    --audio_dir /app/audios \
    --gt_dir /app/gt \
    --out_dir /app/resultados \
    --run_name evaluacion_docker \
    --num_gpus 1
```

Notas:
- `--shm-size=2g` evita errores de memoria compartida cuando se procesa en paralelo.
- Ajusta `--num_gpus` a las GPUs expuestas al contenedor.
- Los archivos generados pertenecen a `root`; si necesitas otro propietario, ajústalo con `chown` después de la ejecución.
- `--env-file` requiere el formato `HF_TOKEN=valor`, sin comillas.

---

## 6. Pruebas

```bash
pytest tests/ -v
```