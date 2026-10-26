import pytest
import pandas as pd
from src.metrics.performance import PerformanceCalculator

@pytest.fixture
def dummy_data():
    """Genera DataFrames de prueba: df_true (Ground Truth) y df_pred (Predicciones)"""
    df_true = pd.DataFrame([
        {"start_time": 0.0, "end_time": 5.0, "speaker_label": "spk1", "gender": "male", "transcript": "hola mundo cruento"},
        {"start_time": 5.0, "end_time": 10.0, "speaker_label": "spk2", "gender": "female", "transcript": "adiós"}
    ])
    
    df_pred = pd.DataFrame([
        # Error de transcripción (WER), pero acierto en género
        {"start_time": 0.0, "end_time": 5.0, "speaker_label": "SPEAKER_00", "gender": "male", "transcript": "hola mundo nuevo"},
        # Error de género (masculino en vez de femenino)
        {"start_time": 5.0, "end_time": 10.0, "speaker_label": "SPEAKER_01", "gender": "male", "transcript": "adiós"}
    ])
    
    return df_true, df_pred

def test_calculate_asr_metrics(dummy_data):
    df_true, df_pred = dummy_data
    metrics = PerformanceCalculator.calculate_asr_metrics(df_true, df_pred)
    
    # Ground truth: "hola mundo cruento adiós" (4 palabras)
    # Hipótesis: "hola mundo nuevo adiós" (4 palabras)
    # Errores: 1 sustitución ("cruento" por "nuevo")
    # WER = 1 / 4 = 0.25
    assert "WER" in metrics
    assert "CER" in metrics
    assert metrics["WER"] == 0.25

def test_calculate_gender_metrics(dummy_data):
    df_true, df_pred = dummy_data
    metrics = PerformanceCalculator.calculate_gender_metrics(df_true, df_pred)
    
    # 2 muestras: 1 acierto (male->male), 1 fallo (female->male)
    assert metrics["accuracy"] == 0.5
    assert "sensitivity_macro" in metrics
    assert "precision_macro" in metrics