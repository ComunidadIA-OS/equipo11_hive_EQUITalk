import pytest
import pandas as pd
from src.metrics.equity import EquityCalculator

@pytest.fixture
def df_speaking_time():
    """Genera datos para probar porcentajes de tiempo de habla."""
    return pd.DataFrame([
        {"start_time": 0.0, "end_time": 4.0, "gender": "female"},       # 4s
        {"start_time": 4.0, "end_time": 10.0, "gender": "male"},        # 6s
        {"start_time": 10.0, "end_time": 12.0, "gender": "unknown"},    # 2s (debe ser ignorado)
        {"start_time": 12.0, "end_time": 12.0, "gender": "female"}      # 0s (caso límite)
    ])

@pytest.fixture
def df_interruptions():
    """Genera casos de solapamiento para probar interrupciones."""
    return pd.DataFrame([
        # 1. Turno inicial (Mujer)
        {"start_time": 0.0, "end_time": 5.0, "speaker_label": "spk1", "gender": "female"},
        
        # 2. Hombre interrumpe a la Mujer (solapamiento de 1.0s) -> female_to_male
        {"start_time": 4.0, "end_time": 8.0, "speaker_label": "spk2", "gender": "male"},
        
        # 3. Hombre solapa con el anterior pero no supera el umbral (0.2s) -> Ignorado
        {"start_time": 7.8, "end_time": 10.0, "speaker_label": "spk3", "gender": "male"},
        
        # 4. Mujer interrumpe al Hombre 3 (solapamiento de 2.0s) -> male_to_female
        {"start_time": 8.0, "end_time": 12.0, "speaker_label": "spk1", "gender": "female"},
        
        # 5. La Mujer se solapa consigo misma (ej. error de diarización) -> Ignorado
        {"start_time": 11.0, "end_time": 13.0, "speaker_label": "spk1", "gender": "female"}
    ])


def test_calculate_speaking_time(df_speaking_time):
    result = EquityCalculator.calculate_speaking_time(df_speaking_time)
    
    # Tiempo válido total: 10s (4s female + 6s male). El 'unknown' se descarta.
    assert result["female_pct"] == 40.0
    assert result["male_pct"] == 60.0

def test_calculate_speaking_time_empty_and_zero():
    # Prueba con DataFrame vacío
    df_empty = pd.DataFrame(columns=["start_time", "end_time", "gender"])
    res_empty = EquityCalculator.calculate_speaking_time(df_empty)
    assert res_empty["female_pct"] == 0.0
    assert res_empty["male_pct"] == 0.0

    # Prueba con tiempo válido total igual a 0 para evitar ZeroDivisionError
    df_zero = pd.DataFrame([{"start_time": 0.0, "end_time": 0.0, "gender": "female"}])
    res_zero = EquityCalculator.calculate_speaking_time(df_zero)
    assert res_zero["female_pct"] == 0.0


def test_detect_interruptions(df_interruptions):
    # Umbral por defecto es 0.5s
    df_ints = EquityCalculator.detect_interruptions(df_interruptions)
    
    # Debe detectar 2 interrupciones válidas y filtrar las cortas (0.2s) o las del mismo hablante (spk1 -> spk1)
    assert len(df_ints) == 2
    
    # Validación de la interrupción 1: Hombre interrumpe a Mujer
    int_1 = df_ints.iloc[0]
    assert int_1["interrupted_speaker"] == "spk1"
    assert int_1["interruptor_speaker"] == "spk2"
    assert int_1["type"] == "female_to_male"
    assert int_1["overlap_duration"] == 1.0

    # Validación de la interrupción 2: Mujer interrumpe a Hombre
    int_2 = df_ints.iloc[1]
    assert int_2["interrupted_speaker"] == "spk3"
    assert int_2["interruptor_speaker"] == "spk1"
    assert int_2["type"] == "male_to_female"
    assert int_2["overlap_duration"] == 2.0


def test_detect_interruptions_custom_threshold(df_interruptions):
    # Si subimos el umbral a 1.5s, la primera interrupción (1.0s) se ignora y solo queda la de 2.0s
    df_ints = EquityCalculator.detect_interruptions(df_interruptions, overlap_threshold=1.5)
    assert len(df_ints) == 1
    assert df_ints.iloc[0]["type"] == "male_to_female"


def test_get_male_to_female_interruptions_count(df_interruptions):
    # El wrapper cuenta exclusivamente "female_to_male" en la columna "type"
    count = EquityCalculator.get_male_to_female_interruptions_count(df_interruptions)
    assert count == 1