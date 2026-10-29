# THE HIVE (Equipo 11) – Detector de Interrupciones

Sistema en Python que analiza grabaciones de voz para **transcribir, identificar quién habla y cuándo, inferir el género de cada hablante y medir la equidad de la conversación** (tiempo de habla, interrupciones, turnos). Opcionalmente compara los resultados del modelo contra transcripciones de referencia (*Ground Truth*, `.trs`) para cuantificar su rendimiento.

---

## 1. ¿Qué hace el sistema?

El flujo tiene dos etapas, orquestadas desde `main.py`:

```text
 Audios (.wav)                                         Ground Truth (.trs / .xml)
      │                                                          │
      ▼                                                          │ (opcional)
┌────────────────────────── 1. INFERENCIA (GPU o CPU) ──────┐    │
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

1. **Inferencia** (`src/pipeline/`): procesa cada audio en paralelo, repartiendo el trabajo entre las GPUs disponibles. Si el ordenador no tiene GPU NVIDIA, se ejecuta en CPU automáticamente (más lento).
2. **Evaluación** (`src/evaluation/`, `src/metrics/`): calcula las métricas de equidad a partir de las predicciones y, si se proporciona Ground Truth, también las métricas de rendimiento de la IA.

Además, una **interfaz web** (`app.py`, ver [sección 5](#5-demo-interfaz-web)) permite analizar un audio de forma visual.

Si no se indica `--gt_dir`, el sistema generará las transcripciones, los perfiles de hardware (tiempo, RAM, GPU) y las métricas de equidad, pero omitirá las métricas de precisión (WER, DER) al no tener transcripciones humanas de referencia.

### Modelos utilizados

| Tarea | Modelo / librería |
|---|---|
| Transcripción (ASR) | Faster Whisper (`large-v3-turbo` por defecto) |
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
├── tests/              # Pruebas unitarias e integración (pytest)
├── ejemplos/           # Audios .wav que aparecen como ejemplos en la demo
├── .github/workflows/  # GitHub Actions: ejecuta los tests en cada cambio
├── main.py             # Orquestador principal (CLI, procesamiento por lotes)
├── app.py              # Demo: interfaz web (Gradio)
├── pytest.ini          # Configuración de pruebas
├── requirements.txt    # Dependencias
├── Dockerfile          # Imagen Docker (versión CPU por defecto, GPU opcional)
├── docker-compose.yml  # Arranque de la demo con Docker
└── .dockerignore       # Archivos excluidos de la imagen Docker
```

---

## 2. Requisitos previos

- **Python 3.11** (la misma versión que usa la imagen Docker).
- **GPU NVIDIA con CUDA** (opcional, recomendada para muchos audios). Sin GPU todo funciona en CPU, más despacio. Para usar la GPU desde Docker se necesita además el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
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

Desde la terminal, en la carpeta del proyecto:

```bash
# Linux / macOS
echo "HF_TOKEN=tu_token_aqui" > .env
```

```powershell
# Windows (PowerShell). No usar "echo ... >": guarda el archivo en un formato que Docker no lee bien
Set-Content -Path .env -Value "HF_TOKEN=tu_token_aqui" -Encoding ascii
```

> El `.env` contiene tu token: no lo subas a GitHub (ya está en `.gitignore`).
>
> **Windows:** evita tener el proyecto dentro de OneDrive (sincroniza el entorno virtual y los modelos y provoca errores). Por ejemplo, usa `C:\proyectos\equipo11_hive`.

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
| `--num_gpus` | No | `2` | Número de GPUs a usar en la inferencia. En CPU indica cuántos audios se procesan a la vez: usa `1` (cada proceso carga todos los modelos en memoria). |
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

## 5. Demo (interfaz web)

`app.py` es una página web hecha con [Gradio](https://www.gradio.app/) que se abre en el navegador pero se ejecuta en tu ordenador. Usa el mismo pipeline que `main.py` sobre **un solo audio** y muestra los resultados de forma visual. Está pensada para enseñar el sistema con audios cortos; para procesar muchos audios usa `main.py`.

### Arrancar la demo

**Con Docker** (recomendado, no hace falta instalar Python):

```bash
docker compose up --build demo
```

**Sin Docker** (tras la [instalación local](#3-instalación-local) y cargar el `.env`):

```bash
python app.py
```

La primera vez descarga los modelos de Hugging Face (varios GB) y tarda unos minutos. Cuando en la terminal aparezca `Running on local URL`, abre **http://localhost:7860**. Para pararla: `Ctrl + C`.

### Usar la demo

1. **Audio:** sube un archivo o graba con el micrófono. Los `.wav` de la carpeta `ejemplos/` aparecen como botones.
2. **Idioma:** catalán, castellano, gallego o detección automática.
3. **Referencia `.trs` (opcional):** si se adjunta, se evalúa el modelo (WER, DER, acierto de género).
4. Pulsa **Analizar**. Los resultados aparecen en pestañas:
   - **Resumen:** % de tiempo y de palabras de mujeres, hablantes, interrupciones hombre → mujer y mujer → hombre, índice de asimetría, quién abre y quién cierra, y tiempo de procesamiento.
   - **Línea de tiempo:** quién habla en cada momento (color por género), con las interrupciones marcadas.
   - **Transcripción:** tiempos, hablante, género y texto.
   - **Interrupciones:** cada interrupción real (sin asentimientos), con los mismos filtros que las métricas.
   - **Descargas:** Excel, JSON y CSV del análisis (también en `data/demo_runs/`).

### Qué depende del idioma

| Paso | ¿Depende del idioma? |
|---|---|
| Transcripción (Whisper) | **Sí.** Funciona en catalán, castellano y gallego (con menos calidad en gallego). Si se elige un idioma que no es el del audio, la transcripción sale mal. |
| Diarización y género | No: trabajan con la voz, no con las palabras. |
| Tiempo de habla, turnos e interrupciones | **No:** salen de la diarización y el género. |
| % de palabras | **Sí:** cuenta las palabras transcritas. |
| WER (evaluación con `.trs`) | **Sí:** la normalización del texto (`text_normalizer.py`) está adaptada al catalán. |

### Rendimiento en CPU

- Usa audios cortos (1–3 minutos) y prueba antes cuánto tarda tu ordenador.
- Con `large-v3-turbo` hacen falta varios GB de RAM (mejor 16 GB). En Docker Desktop: *Settings → Resources → Memory*.
- Si va lento, añade `ASR_MODEL=small` (o `medium`) al `.env`: transcribe algo peor, pero **no cambia** el tiempo de habla ni las interrupciones.

### Usar la demo sin internet

Arranca la demo una vez con conexión para que se descarguen los modelos y después añade `HF_HUB_OFFLINE=1` al `.env`.

### Variables de entorno (`.env`)

| Variable | Para qué sirve | Por defecto |
|---|---|---|
| `HF_TOKEN` | Token de Hugging Face | (obligatorio) |
| `ASR_MODEL` | Modelo de Whisper (`large-v3-turbo`, `medium`, `small`...) | `large-v3-turbo` |
| `DEMO_DEVICE` | `auto`, `cpu` o `cuda` | `auto` |
| `HF_HUB_OFFLINE` | `1` para no conectarse a internet (modelos ya descargados) | – |
| `GPU_INDEX` | Qué GPU usar, si hay varias | `0` |
| `DEMO_OUTPUT_DIR` | Carpeta de resultados de la demo | `data/demo_runs` |
| `DEMO_EXAMPLES_DIR` | Carpeta de audios de ejemplo | `ejemplos` |
| `DEMO_PRELOAD` | `0` para no cargar los modelos al arrancar | `1` |

---

## 6. Ejecución con Docker

Hay dos versiones de la imagen, construidas con el mismo `Dockerfile`:

| Versión | Para qué | Construir |
|---|---|---|
| **CPU** (por defecto) | Cualquier ordenador; demo y audios sencillos | `docker build -t audio-ai-processor .` |
| **GPU** | Ordenador con GPU NVIDIA; muchos audios | `docker build --build-arg DEVICE=gpu -t audio-ai-processor-gpu .` |

### Demo

```bash
docker compose up --build demo       # CPU
docker compose up --build demo-gpu   # GPU NVIDIA
```

### Pipeline por lotes (`main.py`)

La imagen define `main.py` como *entrypoint*, por lo que **solo debes pasar los argumentos** del programa (sin escribir `python main.py`).

**1. Construir la imagen** (ver tabla de arriba)

**2. Ver la ayuda**

```bash
docker run --rm audio-ai-processor
```

**3. Ejecutar el pipeline completo**

Los volúmenes (`-v`) conectan tus carpetas locales con el contenedor. El volumen `hf_cache` guarda los modelos descargados para no volver a bajarlos en cada ejecución (lo comparte con la demo).

Con **GPU**:

```bash
docker run --rm -it --gpus all --shm-size=2g \
  --env-file .env \
  -v $(pwd)/data/audio:/app/audios \
  -v $(pwd)/data/trs:/app/gt \
  -v $(pwd)/data/procesado:/app/resultados \
  -v hf_cache:/app/.cache/huggingface \
  audio-ai-processor-gpu \
    --audio_dir /app/audios \
    --gt_dir /app/gt \
    --out_dir /app/resultados \
    --run_name evaluacion_docker \
    --num_gpus 1
```

Con **CPU**: el mismo comando sin `--gpus all`, con la imagen `audio-ai-processor` y `--num_gpus 1`.

Notas:
- `--shm-size=2g` evita errores de memoria compartida cuando se procesa en paralelo.
- Con GPU, ajusta `--num_gpus` a las GPUs expuestas al contenedor.
- En **Windows (PowerShell)**, cambia `$(pwd)` por `${PWD}` y la `\` del final de cada línea por `` ` `` (o escribe todo en una sola línea).
- Los archivos generados pertenecen a `root`; si necesitas otro propietario, ajústalo con `chown` después de la ejecución.
- `--env-file` requiere el formato `HF_TOKEN=valor`, sin comillas.

---

## 7. Pruebas

```bash
pytest tests/ -v
```

Los tests se ejecutan automáticamente con **GitHub Actions** (`.github/workflows/tests.yml`) en cada cambio subido a `main` y en cada *pull request*. El resultado se ve en la pestaña **Actions** del repositorio.

---

## 8. Licencias de los modelos

| Modelo | Uso en el proyecto | Licencia |
|---|---|---|
| OpenAI Whisper `large-v3-turbo` (vía Faster Whisper) | Transcripción | MIT |
| `pyannote/speaker-diarization-3.1` y `pyannote/segmentation-3.0` | Diarización | MIT (requiere aceptar las condiciones en Hugging Face) |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | Género | CC BY-NC-SA 4.0 (**solo uso no comercial**) |

Por la licencia del modelo de género, el sistema completo no puede usarse con fines comerciales.
