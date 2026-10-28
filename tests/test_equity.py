import math
import pytest
import numpy as np
import pandas as pd
from src.metrics.equity import (
    EquityCalculator,
    _pct,
    _ratio,
    _gini,
    _stat
)

# Tests de Utils
# ----------------------------------------------------------------------------
def test_utils_pct_and_ratio():
    assert _pct(25.0, 100.0) == 25.0
    assert math.isnan(_pct(10.0, 0.0))
    
    assert _ratio(10.0, 2.0) == 5.0
    assert math.isnan(_ratio(10.0, 0.0))
    assert math.isnan(_ratio(float('nan'), 2.0))

def test_utils_gini():
    # Reparto equitativo
    assert math.isclose(_gini(np.array([10.0, 10.0, 10.0])), 0.0)
    # Reparto desigual
    assert _gini(np.array([0.0, 0.0, 10.0])) > 0.5
    # Casos límite
    assert math.isnan(_gini(np.array([])))
    assert math.isnan(_gini(np.array([0.0, 0.0])))

def test_utils_stat():
    s = pd.Series([1.0, 2.0, 3.0])
    assert _stat(s, np.mean) == 2.0
    assert math.isnan(_stat(pd.Series(dtype=float), np.mean))

# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def df_raw_segments():
    """Datos en bruto simulando salida de ASR/Diarización."""
    return pd.DataFrame([
        {"start_time": 0.0, "end_time": 4.0, "speaker_label": "spk1", "gender": "FEMALE", "transcript": "hola mundo"},
        {"start_time": 3.0, "end_time": 5.0, "speaker_label": "spk2", "gender": "Male", "transcript": "que tal"},
        {"start_time": 4.5, "end_time": 4.5, "speaker_label": "spk1", "gender": "female", "transcript": ""}, # Duración 0
        {"start_time": 6.0, "end_time": 7.0, "speaker_label": "spk3", "gender": "UNKNOWN", "transcript": "silencio"}
    ])

@pytest.fixture
def df_turns_gap():
    """Datos para probar la unión de turnos (_merge_turns) y detección de interrupciones."""
    return pd.DataFrame([
        {"start_time": 0.0, "end_time": 2.0, "speaker_label": "spk1", "gender": "female"},
        {"start_time": 2.5, "end_time": 4.0, "speaker_label": "spk1", "gender": "female"}, # Separado por 0.5s
        {"start_time": 6.0, "end_time": 8.0, "speaker_label": "spk1", "gender": "female"}, # Separado por 2.0s
        {"start_time": 3.0, "end_time": 5.0, "speaker_label": "spk2", "gender": "male"}
    ])

# Tests de Métodos Internos 
# ----------------------------------------------------------------------------
def test_prepare(df_raw_segments):
    df_prep = EquityCalculator._prepare(df_raw_segments)
    
    # Debe filtrar duraciones <= 0, normalizar géneros y rellenar transcripciones
    assert len(df_prep) == 3
    assert list(df_prep["gender"]) == ["female", "male", "unknown"]
    assert "duration" in df_prep.columns
    assert df_prep.iloc[0]["duration"] == 4.0

def test_merge_turns(df_turns_gap):
    # Con un gap de 1.0s, los dos primeros turnos de spk1 deben unirse
    df_merged = EquityCalculator._merge_turns(df_turns_gap, max_gap_s=1.0)
    
    spk1_turns = df_merged[df_merged["speaker_label"] == "spk1"]
    assert len(spk1_turns) == 2
    assert spk1_turns.iloc[0]["start_time"] == 0.0
    assert spk1_turns.iloc[0]["end_time"] == 4.0
    assert spk1_turns.iloc[1]["start_time"] == 6.0

def test_overlap_seconds(df_turns_gap):
    # spk1(0-2, 2.5-4, 6-8) y spk2(3-5). Solape entre 3.0 y 4.0 -> 1.0 segundo.
    overlap = EquityCalculator._overlap_seconds(df_turns_gap)
    assert overlap == 1.0

def test_detect_interruptions_backchannel():
    """Valida que un solape donde el interruptor NO toma la palabra se identifique adecuadamente."""
    df_units = pd.DataFrame([
        {"start_time": 0.0, "end_time": 5.0, "speaker_label": "spk1", "gender": "female"},
        {"start_time": 1.0, "end_time": 2.0, "speaker_label": "spk2", "gender": "male"} # Entra durante spk1 pero termina antes (back-channel)
    ])
    
    ev_all = EquityCalculator.detect_interruptions(df_units, overlap_threshold=0.5)
    assert len(ev_all) == 1
    assert not ev_all.iloc[0]["takes_floor"] # Es un asentimiento/backchannel

# Tests del cálculo principal
# ----------------------------------------------------------------------------
def test_compute_full_metrics(df_raw_segments, df_turns_gap):
    # Ejecutamos compute usando df_raw_segments como segmentos base
    metrics = EquityCalculator.compute(
        df_segments=df_raw_segments, 
        df_turns=df_turns_gap, # Usamos esto para forzar que base tiempos en turnos
        turn_gap_s=1.0, 
        overlap_threshold_s=0.5,
        n_temporal_bins=2
    )
    
    # 1. Configuración global
    assert metrics.time_source == "diarization_turns"
    assert metrics.span_s == 8.0 # de 0.0 a 8.0 en df_turns_gap
    
    # 2. Tiempos de Habla (SpeakingTimeMetrics)
    # spk1 total: (2.0-0.0) + (4.0-2.5) + (8.0-6.0) = 5.5s (female). spk2 total: 2.0s (male)
    assert metrics.speaking_time.female_s == 5.5
    assert metrics.speaking_time.male_s == 2.0
    assert metrics.speaking_time.female_pct == pytest.approx(73.33333333333333)
    assert metrics.speaking_time.overlap_s == 1.0
    
    # 3. Palabras (WordMetrics) - basado en df_segments
    assert metrics.words.female == 2 # "hola mundo"
    assert metrics.words.male == 2   # "que tal"
    assert metrics.words.unknown == 1 # "silencio"
    
    # 4. Hablantes (SpeakerMetrics)
    assert metrics.speakers.n_female == 1
    assert metrics.speakers.n_male == 1
    assert metrics.speakers.time_per_speaker_ratio_f_m == 2.75  # 5.5 / 2.0
    assert metrics.speakers.top_speaker_gender == "female"
    
    # 5. Turnos (TurnMetrics) y Transiciones
    assert metrics.turns.mean_s_female == 3.0 # Turnos de 4s y 2s
    assert metrics.turns.mean_ratio_f_m == 1.5 # 3.0 / 2.0
    assert metrics.turns.transitions["female_to_male"] == 1
    
    # 6. Control de la palabra (FloorMetrics)
    assert metrics.floor.first_speaker_gender == "female"
    assert metrics.floor.last_speaker_gender == "female"
    assert metrics.floor.first_female_onset_s == 0.0
    
    # 7. Interrupciones (InterruptionMetrics)
    # spk1 habla de 2.5 a 4.0. spk2 entra en 3.0 y habla hasta 5.0. 
    # Solape = 1.0s (>= 0.5s). spk2 finaliza en 5.0 > 4.0 (toma la palabra).
    assert metrics.interruptions.total == 1
    assert metrics.interruptions.male_over_female == 1
    assert metrics.interruptions.floor_taking_male_over_female == 1
    
    # 8. Análisis Temporal (TemporalMetrics)
    # 2 bins sobre 8.0s: q1(0-4s), q2(4-8s).
    assert "q1" in metrics.temporal.female_time_pct_by_quarter
    assert "q2" in metrics.temporal.female_time_pct_by_quarter

def test_compute_empty_data():
    df_empty = pd.DataFrame(columns=["start_time", "end_time", "speaker_label", "gender", "transcript"])
    metrics = EquityCalculator.compute(df_segments=df_empty)
    
    assert metrics.total_speech_s == 0.0
    assert math.isnan(metrics.speaking_time.female_pct)
    assert metrics.interruptions.total == 0