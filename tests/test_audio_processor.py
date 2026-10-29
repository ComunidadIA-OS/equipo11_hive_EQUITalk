# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from unittest.mock import patch, MagicMock

from src.pipeline.audio_processor import AudioProcessor, ProcessingOutput

@pytest.fixture
@patch("src.pipeline.audio_processor.AudioProcessor._load_models")
def mock_processor(mock_load):
    # Inicializamos el procesador. 
    processor = AudioProcessor(gpu_index=0, hf_token="dummy", language="ca")
    processor.model_load_seconds = 0.5 # Simulamos el tiempo de carga
    
    # Forzamos el uso de CPU para que no falle en GitHub Actions al no detectar CUDA
    processor.device = torch.device("cpu")
    
    # Reemplazamos los modelos con Mocks para simular sus salidas
    processor.asr_model = MagicMock()
    processor.dia_pipe = MagicMock()
    processor.gender_model = MagicMock()
    processor.gender_proc = MagicMock()
    
    return processor

@pytest.fixture
def dummy_wav_file(tmp_path):
    """Genera un archivo .wav de silencio de 3 segundos al vuelo para el test."""
    file_path = tmp_path / "test_audio.wav"
    # Audio mono de 3 segundos a 16kHz
    dummy_audio = np.zeros(16000 * 3, dtype="float32")
    sf.write(str(file_path), dummy_audio, 16000)
    return str(file_path)

def test_assignment_and_classification_logic(mock_processor):
    """Prueba la lógica matemática separada de las nuevas funciones."""
    dummy_audio = np.zeros(16000 * 10, dtype="float32")
    
    # Simulamos la transcripción de Whisper
    transcripts = [
        {"start_time": 0.0, "end_time": 4.0, "transcript": "hola esto es una prueba"},
        {"start_time": 5.0, "end_time": 10.0, "transcript": "estoy de acuerdo"}
    ]
    
    # Simulamos la diarización de Pyannote
    turns = [
        (0.0, 4.5, "SPEAKER_00"),
        (4.8, 10.0, "SPEAKER_01")
    ]
    
    # 1. Probar asignación de hablantes al texto
    df_text = mock_processor._assign_speakers(transcripts, turns)
    assert len(df_text) == 2
    assert df_text.iloc[0]["speaker_label"] == "SPEAKER_00"
    assert df_text.iloc[1]["speaker_label"] == "SPEAKER_01"
    
    # 2. Probar clasificación de género
    def mock_get_gender_probs(audio, start, end):
        if start < 4.5:
            # SPEAKER_00: Mujer (índice 0 en CLASSES = ["female", "male", "child"])
            return np.array([0.9, 0.1, 0.0]), (end - start)
        else:
            # SPEAKER_01: Hombre (índice 1)
            return np.array([0.1, 0.8, 0.1]), (end - start)
            
    mock_processor._get_gender_probs = mock_get_gender_probs
    speaker_gender = mock_processor._classify_speakers(dummy_audio, turns, threshold=0.8)
    
    assert speaker_gender["SPEAKER_00"][0] == "female"
    assert speaker_gender["SPEAKER_00"][1] == pytest.approx(0.9)  # Confianza
    assert speaker_gender["SPEAKER_01"][0] == "male"
    assert speaker_gender["SPEAKER_01"][1] == pytest.approx(0.8)  # Confianza
    
    # 3. Probar construcción de outputs
    segments, turns_df = mock_processor._build_outputs(df_text, turns, speaker_gender)
    
    # Verificaciones sobre la tabla de segmentos
    assert list(segments.columns) == ["start_time", "end_time", "speaker_label", "gender", "gender_confidence", "transcript"]
    assert segments.iloc[0]["gender"] == "female"
    assert segments.iloc[0]["gender_confidence"] == pytest.approx(0.9)
    
    # Verificaciones sobre la tabla de turnos
    assert list(turns_df.columns) == ["start_time", "end_time", "speaker_label", "gender", "gender_confidence"]
    assert turns_df.iloc[1]["speaker_label"] == "SPEAKER_01"
    assert turns_df.iloc[1]["gender"] == "male"

def test_process_file_end_to_end_mocked(mock_processor, dummy_wav_file):
    """Prueba que el pipeline completo fluye correctamente con un archivo simulado y devuelve la dataclass esperada."""
    
    # 1. Simular la respuesta de Whisper (generador + info object)
    class MockSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
            
    mock_info = MagicMock()
    mock_info.language = "ca"
    mock_info.language_probability = 0.98
            
    mock_processor.asr_model.transcribe.return_value = (
        [MockSegment(0.0, 2.0, "texto simulado")], 
        mock_info
    )
    
    # 2. Simular la respuesta de Pyannote 
    class MockTrack:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    mock_diar_result = MagicMock()
    mock_diar_result.speaker_diarization = mock_diar_result
    mock_diar_result.itertracks.return_value = [
        (MockTrack(0.0, 2.0), None, "SPEAKER_00")
    ]
    mock_processor.dia_pipe.return_value = mock_diar_result
    
    # 3. Simular la respuesta de género
    mock_processor._get_gender_probs = lambda a, s, e: (np.array([0.9, 0.1, 0.0]), 2.0)
    
    # Ejecutar proceso
    result = mock_processor.process_file(dummy_wav_file)
    
    # Verificaciones de estructura
    assert isinstance(result, ProcessingOutput)
    
    # Validar tabla de segmentos (ASR)
    assert not result.segments.empty
    assert result.segments.iloc[0]["transcript"] == "texto simulado"
    assert result.segments.iloc[0]["speaker_label"] == "SPEAKER_00"
    assert result.segments.iloc[0]["gender"] == "female"
    assert result.segments.iloc[0]["gender_confidence"] == pytest.approx(0.9)
    
    # Validar tabla de turnos puros (Diarización)
    assert not result.turns.empty
    assert result.turns.iloc[0]["speaker_label"] == "SPEAKER_00"
    assert result.turns.iloc[0]["gender"] == "female"
    assert result.turns.iloc[0]["start_time"] == 0.0
    assert result.turns.iloc[0]["end_time"] == 2.0
    
    # Validar metadatos
    assert "language_detected" in result.metadata
    assert result.metadata["language_detected"] == "ca"
    assert result.metadata["language_probability"] == pytest.approx(0.98)
    assert result.metadata["gpu_index"] == 0