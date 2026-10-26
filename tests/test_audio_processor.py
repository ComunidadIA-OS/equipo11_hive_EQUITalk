import pytest
import numpy as np
import pandas as pd
import soundfile as sf
from unittest.mock import patch, MagicMock
from src.pipeline.audio_processor import AudioProcessor

@pytest.fixture
@patch("src.pipeline.audio_processor.AudioProcessor._load_models")
def mock_processor(mock_load):
    # Inicializamos el procesador. Al estar _load_models mockeado, no descargará nada.
    processor = AudioProcessor(gpu_index=0, hf_token="dummy")
    
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

def test_merge_and_classify_logic(mock_processor):
    """Prueba la lógica matemática."""
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
    
    # Simulamos la salida de _get_gender_probs (las probabilidades de la red neuronal)
    def mock_get_gender_probs(audio, start, end):
        if start < 4.5:
            # SPEAKER_00: Alta probabilidad de mujer (índice 0 en CLASSES = ["female", "male", "child"])
            return np.array([0.9, 0.1, 0.0]), (end - start)
        else:
            # SPEAKER_01: Alta probabilidad de hombre (índice 1)
            return np.array([0.1, 0.8, 0.1]), (end - start)
            
    mock_processor._get_gender_probs = mock_get_gender_probs
    
    # Ejecutamos la función de cruce
    df_result = mock_processor._merge_and_classify(dummy_audio, transcripts, turns, threshold=0.8)
    
    assert len(df_result) == 2
    
    # Validamos que se asignaron los hablantes y géneros correctos al texto
    assert df_result.iloc[0]["speaker_label"] == "SPEAKER_00"
    assert df_result.iloc[0]["gender"] == "female"
    
    assert df_result.iloc[1]["speaker_label"] == "SPEAKER_01"
    assert df_result.iloc[1]["gender"] == "male"

def test_process_file_end_to_end_mocked(mock_processor, dummy_wav_file):
    """Prueba que el pipeline completo fluye correctamente con un archivo simulado."""
    
    # 1. Simular la respuesta de Whisper (transcribe devuelve un generador)
    class MockSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
            
    mock_processor.asr_model.transcribe.return_value = (
        [MockSegment(0.0, 2.0, "texto simulado")], 
        None
    )
    
    # 2. Simular la respuesta de Pyannote (itertracks devuelve tuplas)
    class MockTrack:
        def __init__(self, start, end):
            self.start = start
            self.end = end
            
    mock_diar_result = MagicMock()
    mock_diar_result.itertracks.return_value = [
        (MockTrack(0.0, 2.0), None, "SPEAKER_00")
    ]
    mock_processor.dia_pipe.return_value = mock_diar_result
    
    # 3. Simular la respuesta de género para que no falle al intentar cruzar
    mock_processor._get_gender_probs = lambda a, s, e: (np.array([0.9, 0.1, 0.0]), 2.0)
    
    # Ejecutar proceso
    df_result = mock_processor.process_file(dummy_wav_file)
    
    # Verificaciones
    assert not df_result.empty
    assert list(df_result.columns) == ["start_time", "end_time", "speaker_label", "gender", "transcript"]
    assert df_result.iloc[0]["transcript"] == "texto simulado"
    assert df_result.iloc[0]["gender"] == "female"