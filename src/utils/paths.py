# SPDX-License-Identifier: Apache-2.0
"""Estructura de carpetas de una ejecución.

    <output_root>/<run_name>/
        predictions/
            <audio>/
                <audio>_transcript_speakers.csv   (segmentos ASR + hablante + género)
                <audio>_diarization.csv           (turnos de diarización + género)
                <audio>_metadata.json             (tiempos y memoria por etapa + GPU)
        evaluation_results.xlsx               (resumen y detalle de métricas)
        evaluation_results.json               (todo, anidado)
        run_config.json                       (argumentos y versiones)
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @classmethod
    def create(cls, output_root: str, run_name: str, make_dirs: bool = True) -> "RunPaths":
        if not run_name or Path(run_name).name != run_name:
            raise ValueError(f"run_name inválido: '{run_name}' (no debe contener rutas ni estar vacío)")
        paths = cls(run_dir=Path(output_root) / run_name)
        if make_dirs:
            paths.predictions_dir.mkdir(parents=True, exist_ok=True)
        return paths

    # Carpetas
    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    def audio_dir(self, stem: str) -> Path:
        """Crea y devuelve la subcarpeta específica para el audio."""
        folder = self.predictions_dir / stem
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # Predicciones por audio
    def segments_csv(self, stem: str) -> Path:
        return self.audio_dir(stem) / f"{stem}_transcript_speakers.csv"

    def diarization_csv(self, stem: str) -> Path:
        return self.audio_dir(stem) / f"{stem}_diarization.csv"

    def metadata_json(self, stem: str) -> Path:
        return self.audio_dir(stem) / f"{stem}_metadata.json"

    # Resultados
    @property
    def results_xlsx(self) -> Path:
        return self.run_dir / "evaluation_results.xlsx"

    @property
    def results_json(self) -> Path:
        return self.run_dir / "evaluation_results.json"

    @property
    def config_json(self) -> Path:
        return self.run_dir / "run_config.json"