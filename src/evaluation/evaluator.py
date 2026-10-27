import time
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from src.utils.trs_parser import parse_transcriber_file
from src.metrics.performance import PerformanceCalculator, PerformanceResult
from src.metrics.equity import EquityCalculator
from src.evaluation.results import FileEvaluation, EquityComparison

class Evaluator:
    """
    Orquesta la evaluación cruzando las predicciones del modelo con el Ground Truth (si existe).
    """

    def __init__(self, predictions_csv_path: str, ground_truth_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.file_name = Path(predictions_csv_path).stem.replace("_transcript_speakers", "")
        self.df_pred = pd.read_csv(predictions_csv_path)
        
        self.df_true = None
        if ground_truth_path and Path(ground_truth_path).exists():
            self.df_true = parse_transcriber_file(ground_truth_path)
            
        self.metadata = metadata

    def analyze(self) -> FileEvaluation:
        """
        Ejecuta el análisis unificado (inferencia y opcionalmente evaluación).
        """
        t0 = time.perf_counter()
        
        # 1. Analíticas base de la predicción
        pred_equity = EquityCalculator.compute(df_segments=self.df_pred)
        
        perf_result = PerformanceResult()
        if self.metadata:
            perf_result.timing = PerformanceCalculator.calculate_timing_metrics(self.metadata)
            
        has_ref = False
        
        # 2. Evaluación contra referencia (solo si hay GT)
        if self.df_true is not None and not self.df_true.empty:
            has_ref = True
            
            perf_result.asr = PerformanceCalculator.calculate_asr_metrics(self.df_true, self.df_pred)
            perf_result.diarization = PerformanceCalculator.calculate_diarization_metrics(self.df_true, self.df_pred)
            perf_result.gender = PerformanceCalculator.calculate_gender_metrics(self.df_true, self.df_pred)
            
            ref_equity = EquityCalculator.compute(df_segments=self.df_true)
            eq_comp = EquityComparison.build(pred_equity, ref_equity)
        else:
            eq_comp = EquityComparison(prediction=pred_equity)
            
        perf_result.evaluation_s = {"total": time.perf_counter() - t0}
        
        return FileEvaluation(
            file=self.file_name, 
            performance=perf_result, 
            equity=eq_comp, 
            has_reference=has_ref
        )