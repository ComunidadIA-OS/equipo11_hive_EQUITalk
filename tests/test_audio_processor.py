import os
import pytest
import pandas as pd
from src.pipeline.audio_processor import AudioProcessor

@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="Requiere Token de HuggingFace")
def test_audio_processor_output_structure():
    """Valida que el pipeline genere un DataFrame con la estructura y datos correctos."""
    
    token = os.environ["HF_TOKEN"]
    processor = AudioProcessor(gpu_index=0, hf_token=token)
    test_audio = "data/agora/c003/agora_2007_01_08_a.wav" 
    
    df = processor.process_file(f"{test_audio}")
    
    # Validar tipos
    assert isinstance(df, pd.DataFrame), "El resultado debe ser un DataFrame de pandas"
    assert not df.empty, "El DataFrame resultante no debería estar vacío"
    
    # Validar columnas
    expected_columns = ["start_time", "end_time", "speaker_label", "gender", "transcript"]
    for col in expected_columns:
        assert col in df.columns, f"Falta la columna esperada: {col}"
        
    # Validar coherencia temporal
    assert (df["start_time"] < df["end_time"]).all(), "start_time debe ser estrictamente menor que end_time"
    
    # Validar que no hay nulos en campos críticos
    assert df["speaker_label"].notna().all(), "Hay etiquetas de hablante nulas"
    assert df["gender"].isin(["female", "male", "child", "unknown"]).all(), "Existen géneros fuera de las categorías esperadas"