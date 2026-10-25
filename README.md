# THE HIVE (Equipo 11) - Detector de Interrupciones

Sistema en Python optimizado para procesar contenidos audiovisuales de voz, extraer transcripciones, diarizar hablantes, clasificar géneros mediante aprendizaje profundo y calcular métricas de equidad (tiempo de habla e interrupciones de género).

---

## Estructura del Proyecto
```text
gender_voice_metrics/
├── data/                   # Directorio de datos (raw, processed, ground_truth)
│   └── agora/c003/         # Audios de prueba (.wav)
├── src/                    # Código fuente modular
│   ├── pipeline/           # Procesamiento de audio y ejecución por lotes
│   ├── metrics/            # Lógica de negocio y cálculo de KPIs
│   ├── evaluation/         # Validación contra Ground Truth
│   └── utils/              # Logger, excepciones y parches de compatibilidad
├── tests/                  # Pruebas unitarias e integración con pytest
├── configs/                # Archivos de configuración (config.yaml)
├── requirements.txt        # Dependencias del sistema
├── conftest.py             # Configuración de ruta global para pytest
└── README.md
```
---

## Configuración del entorno

1. **Clonar el repositorio y crear un entorno virtual:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

```

2. **Instalar las dependencias del proyecto:**
```bash
pip install -r requirements.txt
```

3. **Configurar las credenciales de Hugging Face (Requerido para `pyannote.audio`):**
Crea un archivo llamado `.env` en la raíz del proyecto y añade `HF_TOKEN=tu_token` en el archivo `.env`.

---

## Ejecución de Pruebas Unitarias

Para validar que el pipeline de audio (`AudioProcessor`), los modelos neuronales y el esquema de datos funcionan correctamente en la GPU, ejecuta las pruebas mediante `pytest`:

```bash
python -m pytest tests/test_audio_processor.py -v -s
```
---

## Uso del Sistema

### A. Procesamiento por Lotes (*Batch Processing*)

Para procesar de forma concurrente todos los audios de un directorio distribuyéndolos de forma balanceada (*round-robin*) entre las 2 GPUs V100S:

```python
from src.utils.config_loader import load_config
from src.pipeline.batch_runner import BatchRunner

# Cargar configuración desde config.yaml
config = load_config("configs/config.yaml")

# Inicializar y ejecutar orquestador
runner = BatchRunner(
    input_dir="data/agora/c003",
    output_dir="data/processed",
    num_gpus=config["hardware"]["num_gpus"]
)

results = runner.run()
print(results)
```

### B. Cálculo de Métricas de Negocio

Una vez generado el archivo CSV con las transcripciones y hablantes etiquetados, puedes calcular los KPIs de equidad e interrupciones en la CPU:

```python
import pandas as pd
from src.metrics.calculator import MetricsCalculator

# Cargar resultado procesado
df = pd.read_csv("data/processed/agora_2007_01_08_a_transcript_speakers.csv")

# 1. Porcentaje de tiempo de habla por género
speaking_times = MetricsCalculator.calculate_speaking_time(df)
print("Distribución de tiempo de habla:", speaking_times)

# 2. Detección de interrupciones (con umbral de solapamiento de 0.5s)
interruptions_df = MetricsCalculator.detect_interruptions(df, overlap_threshold=0.5)
print(f"Total de interrupciones detectadas: {len(interruptions_df)}")
display(interruptions_df.head())
```
