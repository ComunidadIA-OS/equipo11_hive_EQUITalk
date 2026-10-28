import time
import json
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

    def __init__(self, predictions_csv_path: str, ground_truth_path: Optional[str] = None):
        pred_path = Path(predictions_csv_path)
        self.file_name = pred_path.stem.replace("_transcript_speakers", "")
        self.df_pred = pd.read_csv(predictions_csv_path)
        
        base_dir = pred_path.parent
        turns_csv_path = base_dir / f"{self.file_name}_diarization.csv"
        metadata_json_path = base_dir / f"{self.file_name}_metadata.json"
        
        # Cargar turnos 
        self.df_turns = None
        if turns_csv_path.exists():
            self.df_turns = pd.read_csv(turns_csv_path)
            
        # Cargar metadatos
        self.metadata = None
        if metadata_json_path.exists():
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        
        self.df_true = None
        if ground_truth_path and Path(ground_truth_path).exists():
            self.df_true = parse_transcriber_file(ground_truth_path)

    def analyze(self) -> FileEvaluation:
        """
        Ejecuta el análisis unificado (inferencia y opcionalmente evaluación).
        """
        t0 = time.perf_counter()
        
        # 1. Analíticas base de la predicción usando turnos para el tiempo real
        pred_equity = EquityCalculator.compute(df_segments=self.df_pred, df_turns=self.df_turns)
        
        perf_result = PerformanceResult()
        if self.metadata:
            perf_result.timing = PerformanceCalculator.calculate_timing_metrics(self.metadata)
            
        has_ref = False
        
        # 2. Evaluación contra referencia (solo si hay GT)
        if self.df_true is not None and not self.df_true.empty:
            has_ref = True
            
            perf_result.asr = PerformanceCalculator.calculate_asr_metrics(self.df_true, self.df_pred)
            
            diarization_df = self.df_turns if self.df_turns is not None else self.df_pred
            perf_result.diarization = PerformanceCalculator.calculate_diarization_metrics(self.df_true, diarization_df)
            
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