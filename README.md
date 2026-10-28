# THE HIVE (Equipo 11) - Detector de Interrupciones y Evaluador

Sistema en Python optimizado para procesar contenidos audiovisuales de voz, extraer transcripciones, diarizar hablantes y clasificar géneros mediante aprendizaje profundo. Incluye un módulo completo de evaluación para contrastar el rendimiento del modelo (ASR, Diarización, Clasificación) contra transcripciones de referencia (*Ground Truth* en formato `.trs`) y calcular métricas de equidad.

---

## Estructura del Proyecto
```text
gender_voice_metrics/
├── data/
│   ├── raw/
│   │   ├── audio/          # Audios de prueba (.wav)
│   │   └── trs/            # Archivos Ground Truth (.trs / XML)
│   ├── interim/            # Transcripciones y predicciones generadas (.csv)
│   └── processed/          # Resultados de evaluación finales (.xlsx)
├── src/
│   ├── pipeline/           # Procesamiento de audio y ejecución por lotes
│   ├── metrics/            # Lógica matemática (rendimiento IA y equidad)
│   ├── evaluation/         # Validación contra Ground Truth (Batch Evaluator)
│   └── utils/              # Parsers (XML/TRS), logger y excepciones
├── tests/                  # Pruebas unitarias e integración con pytest
├── main.py                 # Orquestador principal 
├── requirements.txt        # Dependencias del sistema
└── README.md
```
---

## Configuración del Entorno

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

3. **Configurar las credenciales de Hugging Face:**
Crea un archivo llamado `.env` en la raíz del proyecto y añade el token necesario para el modelo de `pyannote.audio`:

```text
HF_TOKEN=tu_token_aqui
```

4. **Carga el token en la terminal con el comando correspondiente:**

   * **Linux y macOS (Bash / Zsh):**
     ```bash
     export $(grep -v '^#' .env | xargs)
     ```

   * **Windows (PowerShell):**
     ```powershell
     Get-Content .env | Where-Object { $_ -match '=' -and -not$_.StartsWith('#') } | ForEach-Object {
         $name, $value =$_.Split('=', 2)
         [Environment]::SetEnvironmentVariable($name.Trim(),$value.Trim(), 'Process')
     }
     ```
---

## Uso del Sistema 

El proyecto se orquesta ahora desde un punto de entrada centralizado (`main.py`) que permite ejecutar la inferencia mediante GPU y la evaluación posterior de forma consecutiva o aislada.

1. **Ejecución del Pipeline Completo (Inferencia + Evaluación)**

Ejecuta los modelos sobre los audios, genera los CSVs temporales y, a continuación, los cruza con los archivos `.trs` de referencia para exportar el Excel de métricas.

```bash
python main.py --audio_dir data/raw/audio --gt_dir data/raw/trs
```

2. **Evaluación Rápida (Saltar Inferencia)**

Si ya tienes las predicciones `.csv` generadas en ejecuciones anteriores y solo necesitas recalcular el Excel de métricas (por ejemplo, si has ajustado el cálculo de solapamientos), usa el flag `--skip_inference`:

```bash
python main.py --audio_dir data/raw/audio --gt_dir data/raw/trs --skip_inference
```

**3. Configuración Avanzada de Parámetros**

Puedes personalizar el número de GPUs concurrentes, los directorios de salida y el nombre del archivo final de Excel:

```bash
python main.py \
    --audio_dir data/raw/agora \
    --gt_dir data/raw/agora \
    --pred_dir data/predictions \
    --out_dir data/processed \
    --out_filename metricas_agora_v1.xlsx \
    --num_gpus 4
```

---

## Métricas Exportadas

El archivo Excel generado en `data/processed/` contiene las siguientes evaluaciones aplanadas por cada archivo de audio:

* **Rendimiento del Modelo (IA):**
* `ASR_WER` y `ASR_CER`: Word/Character Error Rate.
* `DER (Diarization)`: Tasa de error en la identificación de turnos de hablantes.
* `Gender_Accuracy / Sensitivity / Precision`: Rendimiento de la clasificación de género alineada temporalmente con el *Ground Truth*.

* **Dinámicas de Equidad (Sociolingüística):**
* `Time_Female_Pct` y `Time_Male_Pct`: Porcentaje de tiempo de uso de la palabra.
* `Interruptions_M2F`: Conteo de interrupciones directas de hablantes masculinos hacia hablantes femeninas.

* **Tiempo y Memoria (Inferencia):** se leen de `<audio>_metadata.json`, que la inferencia guarda junto a cada CSV.
* `Time_total_s` y `RTF_total`: segundos de procesamiento y *Real Time Factor* (segundos de cómputo por segundo de audio; <1 es más rápido que tiempo real).
* `Time_ASR_s`, `Time_Diarization_s`, `Time_Gender_s`: segundos de cada etapa del pipeline.
* `Model_load_s`: segundos de carga de los modelos (una vez por GPU).
* `RAM_peak_MB`: pico de RAM del proceso hasta terminar ese audio (incluye los modelos cargados).
* `GPU_peak_MB`: memoria máxima de la tarjeta gráfica que necesita el pipeline (incluye los tres modelos, también Whisper). El detalle por etapa está en el `_metadata.json`.

---

## Ejecución de Pruebas Unitarias

Para validar los módulos de forma aislada mediante `pytest`:
```bash
python -m pytest tests/ -v -s
```

Ejecutar un archivo de pruebas específico:
```bash
python -m pytest tests/test_performance.py -v
```
