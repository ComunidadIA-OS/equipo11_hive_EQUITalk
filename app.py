# SPDX-License-Identifier: Apache-2.0
"""
Demo interactiva de THE HIVE (Equipo 11).

Interfaz web (Gradio) para analizar un audio de forma visual:
subir o grabar un audio -> transcripción + diarización + género -> métricas de equidad
(tiempo de palabra, interrupciones...) y, si se adjunta un .trs, evaluación contra la referencia.

Reutiliza el mismo código que main.py (AudioProcessor + Evaluator), así que los
resultados de la demo son los mismos que los del pipeline por lotes.

Uso:
    export HF_TOKEN=tu_token
    python app.py                  # abre http://localhost:7860
"""
import json
import math
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # sin ventana: las figuras se dibujan en el navegador
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.evaluation.evaluator import Evaluator
from src.evaluation.report import ResultsWriter
from src.evaluation.results import FileEvaluation
from src.metrics.equity import EquityCalculator
from src.utils.paths import RunPaths

# Configuración
# ----------------------------------------------------------------------------
OUTPUT_ROOT = os.environ.get("DEMO_OUTPUT_DIR", "data/demo_runs")
EXAMPLES_DIR = Path(os.environ.get("DEMO_EXAMPLES_DIR", "ejemplos"))
GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))
# "auto": GPU si la hay, si no CPU. Se puede forzar con DEMO_DEVICE=cpu o DEMO_DEVICE=cuda
DEVICE = os.environ.get("DEMO_DEVICE", "auto")
# Modelo de Whisper. En un portátil sin GPU, "small" o "medium" van mucho más rápido (transcriben algo peor)
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")

LANGUAGES: Dict[str, Optional[str]] = {
    "Catalán": "ca",
    "Castellano": "es",
    "Gallego": "gl",
    "Detección automática": None,
}

GENDER_LABELS = {"female": "Mujer", "male": "Hombre", "child": "Infantil", "unknown": "Desconocido"}
GENDER_COLORS = {"female": "#7B3FA0", "male": "#E09F3E", "child": "#4C9F70", "unknown": "#A0A0A0"}

# Mismos parámetros que EquityCalculator.compute para que la tabla cuadre con las métricas
MIN_UNIT_S = 0.3
OVERLAP_THRESHOLD_S = 0.5


# Carga de modelos (una sola vez)
# ----------------------------------------------------------------------------
_processor = None
_processor_lock = threading.Lock()


def get_processor():
    """Carga los modelos la primera vez y los reutiliza en cada análisis."""
    global _processor
    with _processor_lock:
        if _processor is None:
            from src.pipeline.audio_processor import AudioProcessor  # import pesado: solo cuando hace falta

            _processor = AudioProcessor(gpu_index=GPU_INDEX, hf_token=os.environ.get("HF_TOKEN"),
                                        device=DEVICE, asr_model_name=ASR_MODEL)
        return _processor


# Lógica del análisis (sin dependencias de la interfaz)
# ----------------------------------------------------------------------------
@dataclass
class DemoResult:
    evaluation: FileEvaluation
    segments: pd.DataFrame
    turns: pd.DataFrame
    interruptions: pd.DataFrame
    files: List[str]


def save_outputs(output, paths: RunPaths, stem: str) -> Path:
    """Guarda las predicciones con la misma estructura que BatchRunner."""
    output.segments.to_csv(paths.segments_csv(stem), index=False)
    output.turns.to_csv(paths.diarization_csv(stem), index=False)
    with open(paths.metadata_json(stem), "w", encoding="utf-8") as f:
        json.dump(output.metadata, f, indent=2, ensure_ascii=False)
    return paths.segments_csv(stem)


def real_interruptions(segments: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Interrupciones con los mismos filtros que las métricas (sin ruido ni asentimientos)."""
    units = EquityCalculator._prepare(turns if turns is not None and not turns.empty else segments)
    units = units[units["duration"] >= MIN_UNIT_S].reset_index(drop=True)
    events = EquityCalculator.detect_interruptions(units, OVERLAP_THRESHOLD_S)
    return events[events["takes_floor"]].reset_index(drop=True)


def run_analysis(audio_path: str, trs_path: Optional[str], language: Optional[str], processor=None) -> DemoResult:
    """Procesa un audio, guarda los resultados y lo evalúa (contra el .trs si se adjunta)."""
    processor = processor or get_processor()
    processor.language = language

    stem = Path(audio_path).stem
    run_name = f"demo_{datetime.now():%Y%m%d_%H%M%S}_{stem}"
    paths = RunPaths.create(output_root=OUTPUT_ROOT, run_name=run_name)

    output = processor.process_file(audio_path)
    segments_csv = save_outputs(output, paths, stem)

    evaluation = Evaluator(predictions_csv_path=str(segments_csv), ground_truth_path=trs_path).analyze()

    writer = ResultsWriter(paths)
    writer.write_excel({"Summary": pd.DataFrame([evaluation.summary_row()])})
    writer.write_json({stem: evaluation.to_dict()})

    files = [str(paths.results_xlsx), str(paths.results_json),
             str(paths.segments_csv(stem)), str(paths.diarization_csv(stem))]
    return DemoResult(
        evaluation=evaluation,
        segments=output.segments,
        turns=output.turns,
        interruptions=real_interruptions(output.segments, output.turns),
        files=files,
    )


# Presentación de resultados
# ----------------------------------------------------------------------------
def _fmt(x, suffix: str = "", decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}{suffix}"


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def summary_markdown(ev: FileEvaluation) -> str:
    p = ev.equity.prediction
    st, sp, im = p.speaking_time, p.speakers, p.interruptions

    lines = [
        f"### Resultado: `{ev.file}`",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Tiempo de palabra de mujeres | **{_fmt(st.female_pct, ' %')}** |",
        f"| Palabras dichas por mujeres | {_fmt(p.words.female_pct, ' %')} |",
        f"| Hablantes (mujeres / hombres / desconocido) | {sp.n_female} / {sp.n_male} / {sp.n_unknown} |",
        f"| Interrupciones hombre → mujer | **{im.male_over_female}** |",
        f"| Interrupciones mujer → hombre | **{im.female_over_male}** |",
        f"| Índice de asimetría de interrupciones | {_fmt(im.asymmetry_index, decimals=2)} |",
        f"| Quién abre / quién cierra | {GENDER_LABELS.get(p.floor.first_speaker_gender, '—')}"
        f" / {GENDER_LABELS.get(p.floor.last_speaker_gender, '—')} |",
        f"| Primera voz femenina | {_mmss(p.floor.first_female_onset_s) if not math.isnan(p.floor.first_female_onset_s) else '—'} |",
        f"| Tiempo con habla solapada | {_fmt(st.overlap_pct, ' %')} |",
        "",
        "_Índice de asimetría: > 0 los hombres interrumpen más a las mujeres que al revés; < 0 al contrario._",
    ]

    if ev.has_reference:
        perf, err = ev.performance, ev.equity.error
        lines += [
            "",
            "### Comparación con la referencia (.trs)",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| WER (error de transcripción) | {_fmt(perf.asr.wer * 100, ' %')} |",
            f"| DER (error de diarización) | {_fmt(perf.diarization.der * 100, ' %')} |",
            f"| Acierto de género | {_fmt(perf.gender.accuracy * 100, ' %')} |",
            f"| Tiempo de mujeres en la referencia | {_fmt(ev.equity.reference.speaking_time.female_pct, ' %')} |",
            f"| Error en tiempo de mujeres | {_fmt(err.get('speaking_time.female_pct'), ' pp')} |",
        ]

    t = ev.performance.timing
    if t.available:
        lang = t.language_detected or "?"
        lines += ["", f"_Audio de {_fmt(t.audio_duration_s / 60, ' min')} procesado en "
                      f"{_fmt(t.total_s, ' s')} (RTF {_fmt(t.rtf_total, decimals=2)}). "
                      f"Idioma usado por Whisper: {lang}._"]
    return "\n".join(lines)


def timeline_figure(turns: pd.DataFrame, interruptions: pd.DataFrame):
    """Línea de tiempo: una fila por hablante, coloreada por género, con las interrupciones marcadas."""
    fig, ax = plt.subplots(figsize=(11, 0.6 * max(turns["speaker_label"].nunique(), 2) + 1.4))
    if turns.empty:
        ax.text(0.5, 0.5, "No se ha detectado habla", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    speakers = sorted(turns["speaker_label"].unique())
    gender_of = turns.groupby("speaker_label")["gender"].agg(lambda g: g.mode().iat[0]).to_dict()
    row = {spk: i for i, spk in enumerate(speakers)}

    for spk in speakers:
        t = turns[turns["speaker_label"] == spk]
        ax.broken_barh(list(zip(t["start_time"], t["end_time"] - t["start_time"])), (row[spk] - 0.35, 0.7),
                       facecolors=GENDER_COLORS.get(gender_of[spk], GENDER_COLORS["unknown"]))

    if not interruptions.empty:
        ys = [row.get(s, 0) + 0.5 for s in interruptions["interruptor_speaker"]]
        ax.scatter(interruptions["time_at"], ys, marker="v", s=55, color="#C0392B", zorder=3, label="Interrupción")
        ax.legend(loc="upper right", frameon=False)

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([f"{s} ({GENDER_LABELS.get(gender_of[s], '?')})" for s in speakers])
    ax.set_ylim(-0.6, len(speakers) - 0.1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _mmss(x)))
    ax.set_xlabel("Tiempo (mm:ss)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def distribution_figure(ev: FileEvaluation):
    """Reparto mujeres/hombres (tiempo, palabras, hablantes, turnos) y evolución por cuartos."""
    p = ev.equity.prediction
    shares = {
        "Tiempo": p.speaking_time.female_pct,
        "Palabras": p.words.female_pct,
        "Hablantes": p.speakers.female_speaker_pct,
        "Turnos": p.turns.female_pct,
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})

    labels = list(shares)[::-1]
    fem = [0 if math.isnan(shares[k]) else shares[k] for k in labels]
    ax1.barh(labels, fem, color=GENDER_COLORS["female"], label="Mujeres")
    ax1.barh(labels, [100 - f for f in fem], left=fem, color=GENDER_COLORS["male"], label="Hombres")
    for i, f in enumerate(fem):
        ax1.text(f / 2, i, f"{f:.0f} %", va="center", ha="center", color="white", fontsize=9)
    ax1.axvline(50, color="black", lw=0.8, ls="--")
    ax1.set_xlim(0, 100)
    ax1.set_title("Reparto (% de mujeres)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False)

    quarters = p.temporal.female_time_pct_by_quarter
    q_names = list(quarters) or ["q1", "q2", "q3", "q4"]
    q_vals = [0 if (quarters.get(q) is None or math.isnan(quarters.get(q))) else quarters[q] for q in q_names]
    ax2.bar([q.upper() for q in q_names], q_vals, color=GENDER_COLORS["female"])
    ax2.axhline(50, color="black", lw=0.8, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("% de tiempo de mujeres por cuarto del audio")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def transcript_table(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=["Inicio", "Fin", "Hablante", "Género", "Texto"])
    return pd.DataFrame({
        "Inicio": segments["start_time"].map(_mmss),
        "Fin": segments["end_time"].map(_mmss),
        "Hablante": segments["speaker_label"],
        "Género": segments["gender"].map(lambda g: GENDER_LABELS.get(g, g)),
        "Texto": segments["transcript"],
    })


def interruptions_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["Momento", "Interrumpe", "Interrumpido/a", "Solape (s)"])
    return pd.DataFrame({
        "Momento": events["time_at"].map(_mmss),
        "Interrumpe": [f"{s} ({GENDER_LABELS.get(g, g)})"
                       for s, g in zip(events["interruptor_speaker"], events["interruptor_gender"])],
        "Interrumpido/a": [f"{s} ({GENDER_LABELS.get(g, g)})"
                           for s, g in zip(events["interrupted_speaker"], events["interrupted_gender"])],
        "Solape (s)": events["overlap_s"].round(2),
    })


# Interfaz
# ----------------------------------------------------------------------------
def build_interface():
    import gradio as gr

    def on_analyze(audio_path, trs_file, language_name, progress=gr.Progress()):
        if not audio_path:
            raise gr.Error("Sube o graba un audio antes de analizar.")
        progress(0.05, desc="Cargando modelos (solo la primera vez)...")
        try:
            get_processor()
            progress(0.2, desc="Transcribiendo, separando hablantes y detectando género...")
            res = run_analysis(audio_path, trs_file, LANGUAGES[language_name])
        except Exception as e:
            raise gr.Error(f"No se ha podido analizar el audio: {e}")
        plt.close("all")
        return (
            summary_markdown(res.evaluation),
            distribution_figure(res.evaluation),
            timeline_figure(res.turns, res.interruptions),
            transcript_table(res.segments),
            interruptions_table(res.interruptions),
            res.files,
        )

    with gr.Blocks(title="THE HIVE · Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# THE HIVE · ¿Quién tiene la palabra?\n"
            "Sube o graba un audio con varias personas hablando. El sistema transcribe, separa a las "
            "personas que hablan, detecta si la voz es de mujer o de hombre y calcula cuánto protagonismo "
            "tiene cada una: tiempo de palabra, turnos e interrupciones."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                language = gr.Dropdown(list(LANGUAGES), value="Catalán", label="Idioma del audio")
                trs = gr.File(file_types=[".trs", ".xml"], type="filepath",
                              label="Referencia .trs (opcional, para evaluar el modelo)")
                button = gr.Button("Analizar", variant="primary")
                gr.Markdown(
                    "_El idioma solo afecta a la transcripción (y al % de palabras). El tiempo de palabra, "
                    "los turnos y las interrupciones salen de la voz, no del texto._"
                )
                examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.wav")) if EXAMPLES_DIR.exists() else []
                if examples:
                    gr.Examples(examples=[[e] for e in examples], inputs=[audio], label="Audios de ejemplo")
            with gr.Column(scale=2):
                with gr.Tab("Resumen"):
                    summary = gr.Markdown("Los resultados aparecerán aquí.")
                    distribution = gr.Plot(label="Reparto")
                with gr.Tab("Línea de tiempo"):
                    timeline = gr.Plot(label="Quién habla en cada momento")
                with gr.Tab("Transcripción"):
                    transcript = gr.Dataframe(wrap=True, interactive=False)
                with gr.Tab("Interrupciones"):
                    interrupts = gr.Dataframe(interactive=False)
                with gr.Tab("Descargas"):
                    downloads = gr.Files(label="Excel, JSON y CSV de esta ejecución")

        button.click(
            on_analyze,
            inputs=[audio, trs, language],
            outputs=[summary, distribution, timeline, transcript, interrupts, downloads],
            concurrency_limit=1,  # una sola GPU: los análisis se ponen en cola
        )
    return demo


if __name__ == "__main__":
    if os.environ.get("DEMO_PRELOAD", "1") == "1":
        print("Cargando modelos (la primera vez se descargan de Hugging Face)...")
        proc = get_processor()
        print(f"Modelos listos en {proc.device_str} (Whisper: {proc.asr_model_name}).")
    build_interface().queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
