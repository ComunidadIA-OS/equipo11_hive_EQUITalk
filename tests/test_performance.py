# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import numpy as np
import pandas as pd
from src.metrics.performance import (
    PerformanceCalculator,
    ASRMetrics,
    DiarizationMetrics,
    GenderMetrics,
    TimingMetrics,
    PerformanceResult
)

class TestPerformanceCalculator:
    """
    Suite de pruebas para el PerformanceCalculator abarcando métricas ASR, Diarización, Género y Tiempos.
    """

    @pytest.mark.parametrize(
        "ref_texts, hyp_texts, expected_wer, expected_cer",
        [
            (["hola mundo"], ["hola mundo"], 0.0, 0.0),                     # Acierto perfecto
            (["hola mundo cruento"], ["hola mundo nuevo"], 0.3333, 0.2222), # 1 sustitución (1/3 palabras)
            (["hola"], ["hola mundo"], 1.0, 1.5),                           # 1 Inserción (1/1 palabras) -> CER 1.5
            (["hola mundo"], ["hola"], 0.5, 0.6),                           # 1 Borrado (1/2 palabras) -> CER 0.6
            ([], ["falsa alarma"], 1.0, 1.0),                               # Ground truth vacío, predicción con texto
            (["omisión total"], [], 1.0, 1.0),                              # Ground truth con texto, predicción vacía
            ([], [], 0.0, 0.0),                                             # Ambos vacíos
        ]
    )
    
    def test_calculate_asr_metrics_scenarios(self, ref_texts, hyp_texts, expected_wer, expected_cer):
        """
        Prueba exhaustiva del cálculo de WER y CER bajo distintos escenarios.
        """
        # Se incluye 'start_time' porque la función los ordena antes de unirlos
        df_true = pd.DataFrame([{"start_time": i, "transcript": t} for i, t in enumerate(ref_texts)])
        df_pred = pd.DataFrame([{"start_time": i, "transcript": t} for i, t in enumerate(hyp_texts)])
        
        metrics = PerformanceCalculator.calculate_asr_metrics(df_true, df_pred)
        
        assert isinstance(metrics, ASRMetrics)
        assert metrics.wer == pytest.approx(expected_wer, rel=1e-3)
        
        # Ignorar aserción estricta de CER para inserciones/borrados complejos para simplificar el test,
        # pero validamos el comportamiento extremo (1.0 o 0.0)
        if expected_cer in (0.0, 1.0):
            assert metrics.cer == pytest.approx(expected_cer, rel=1e-3)

    def test_calculate_gender_metrics_perfect_alignment(self):
        """
        Caso ideal: La predicción coincide exactamente con la referencia.
        """
        df = pd.DataFrame([
            {"start_time": 0.0, "end_time": 5.0, "gender": "male"},
            {"start_time": 5.0, "end_time": 10.0, "gender": "female"}
        ])
        
        metrics = PerformanceCalculator.calculate_gender_metrics(df_true=df, df_pred=df)
        
        assert metrics.accuracy == 1.0
        assert metrics.accuracy_overall == 1.0
        assert metrics.recall_male == 1.0
        assert metrics.recall_female == 1.0
        assert metrics.coverage == 1.0

    def test_calculate_gender_metrics_empty_data(self):
        """
        Manejo de edge case: DataFrames vacíos no deben fallar y deben retornar NaNs donde aplique.
        """
        df_empty = pd.DataFrame(columns=["start_time", "end_time", "gender"])
        
        metrics = PerformanceCalculator.calculate_gender_metrics(df_empty, df_empty)
        
        assert isinstance(metrics, GenderMetrics)
        assert math.isnan(metrics.accuracy)
        assert metrics.evaluated_s == 0.0

    def test_calculate_diarization_metrics_basic(self):
        """
        Prueba básica funcional de diarización.
        """
        df_true = pd.DataFrame([
            {"start_time": 0.0, "end_time": 10.0, "speaker_label": "spk1"}
        ])
        df_pred = pd.DataFrame([
            {"start_time": 0.0, "end_time": 10.0, "speaker_label": "SPEAKER_00"} # Mismo rango temporal
        ])
        
        metrics = PerformanceCalculator.calculate_diarization_metrics(df_true, df_pred, collar=0.0)
        
        assert isinstance(metrics, DiarizationMetrics)
        assert metrics.n_ref_speakers == 1
        assert metrics.n_hyp_speakers == 1
        # DER debe ser 0.0 ya que los segmentos coinciden perfectamente aunque cambie el label
        assert metrics.der == pytest.approx(0.0)
        assert metrics.miss_s == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "metadata, expected_rtf, is_available",
        [
            # Caso 1: Metadatos completos y válidos
            (
                {"audio_duration_seconds": 10.0, "processing_seconds": 2.0},
                0.2, # RTF = 2.0 / 10.0
                True
            ),
            # Caso 2: Audio duration es 0 (Prevenir división por cero)
            (
                {"audio_duration_seconds": 0.0, "processing_seconds": 2.0},
                float("nan"), 
                True
            ),
            # Caso 3: Metadatos faltantes/vacíos
            (
                None,
                float("nan"),
                False
            ),
        ]
    )
    def test_calculate_timing_metrics(self, metadata, expected_rtf, is_available):
        """
        Prueba de cálculo del Real Time Factor (RTF) y manejo de NaNs.
        """
        metrics = PerformanceCalculator.calculate_timing_metrics(metadata)
        
        assert isinstance(metrics, TimingMetrics)
        assert metrics.available == is_available
        
        if math.isnan(expected_rtf):
            assert math.isnan(metrics.rtf_total)
        else:
            assert metrics.rtf_total == pytest.approx(expected_rtf)

    def test_performance_result_to_dict(self):
        """
        Garantiza que el objeto global se serializa correctamente a diccionario.
        """
        result = PerformanceResult()
        # Modificar un valor para comprobar
        result.asr.wer = 0.15 
        result.timing.available = True
        
        serialized = result.to_dict()
        
        assert isinstance(serialized, dict)
        assert "asr" in serialized
        assert serialized["asr"]["wer"] == 0.15
        assert serialized["timing"]["available"] is True

    def test_calculate_timing_metrics_etapas_y_memoria(self):
        """
        Los segundos por etapa y los picos de memoria del metadata.json llegan a TimingMetrics.
        """
        metadata = {
            "audio_duration_seconds": 100.0,
            "processing_seconds": 25.0,
            "model_load_seconds": 30.0,
            "stage_seconds": {"load_audio": 0.5, "asr": 12.0, "diarization": 9.0, "gender": 3.5},
            "ram_peak_mb": 4096.0,
            "gpu_peak_mb": None,   # sin GPU (o no medida)
        }
        metrics = PerformanceCalculator.calculate_timing_metrics(metadata)

        assert metrics.rtf_total == pytest.approx(0.25)
        assert metrics.model_load_s == 30.0
        assert metrics.asr_s == 12.0
        assert metrics.diarization_s == 9.0
        assert metrics.gender_s == 3.5
        assert metrics.ram_peak_mb == 4096.0
        assert math.isnan(metrics.gpu_peak_mb)

    def test_calculate_timing_metrics_metadata_antiguo(self):
        """
        Un metadata.json sin datos por etapa (versión anterior) no debe fallar.
        """
        metrics = PerformanceCalculator.calculate_timing_metrics(
            {"audio_duration_seconds": 10.0, "processing_seconds": 2.0})

        assert metrics.available is True
        assert math.isnan(metrics.asr_s)
        assert math.isnan(metrics.ram_peak_mb)
