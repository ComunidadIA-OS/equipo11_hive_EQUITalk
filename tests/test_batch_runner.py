import json

import pandas as pd
import pytest

from src.pipeline import batch_runner
from src.pipeline.audio_processor import ProcessingOutput
from src.evaluation.batch_evaluator import BatchEvaluator

METADATA = {
    "processing_seconds": 25.0,
    "audio_duration_seconds": 100.0,
    "model_load_seconds": 30.0,
    "stage_seconds": {"load_audio": 0.5, "asr": 12.0, "diarization": 9.0, "gender": 3.5},
    "ram_peak_mb": 4096.0,
    "gpu_peak_mb": 6200.0,
    "gpu_torch_peak_mb_by_stage": {"asr": 400.0, "diarization": 3000.0, "gender": 1500.0},
    "gpu_device_used_mb_by_stage": {"asr": 5100.0, "diarization": 6200.0, "gender": 6200.0},
}

class FakeProcessor:
    """Sustituye al AudioProcessor real: devuelve un resultado fijo sin usar la GPU."""
    def process_file(self, file_path, *args, **kwargs):
        segments = pd.DataFrame([{
            "start_time": 0.0, 
            "end_time": 4.0, 
            "speaker_label": "SPEAKER_00",
            "gender": "female", 
            "gender_confidence": 0.9, 
            "transcript": "bona nit"
        }])
        turns = segments.drop(columns=["transcript"])
        return ProcessingOutput(segments=segments, turns=turns, metadata=METADATA)


def test_worker_guarda_csv_turnos_y_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_runner, "_worker_processor", FakeProcessor())
    pred_dir = tmp_path / "pred"

    res = batch_runner._process_single_audio_worker((str(tmp_path / "debat.wav"), str(pred_dir)))
    
    # Muestra el error de res en el mensaje de aserción si status != "success"
    assert res.get("status") == "success", f"Error en worker: {res.get('error_message')}"
    
    out_dir = pred_dir / "predictions" / "debat"
    assert (out_dir / "debat_transcript_speakers.csv").exists()
    assert (out_dir / "debat_diarization.csv").exists()
    with open(out_dir / "debat_metadata.json", encoding="utf-8") as f:
        assert json.load(f)["stage_seconds"]["asr"] == 12.0

    df = BatchEvaluator(predictions_dir=str(pred_dir / "predictions"), output_root=str(tmp_path / "out")).run()
    fila = df.iloc[0]
    assert fila["Time_ASR_s"] == 12.0
    assert fila["RAM_peak_MB"] == 4096.0
    assert fila["GPU_peak_MB"] == 6200.0
    assert fila["RTF_total"] == pytest.approx(0.25)