import os
import pandas as pd
from typing import Dict, Any
from src.utils.trs_parser import parse_transcriber_file
from src.metrics.performance import PerformanceCalculator
from src.metrics.equity import EquityCalculator

class Evaluator:
    """Orquesta la evaluación cruzando las predicciones del modelo con el Ground Truth."""
    
    def __init__(self, ground_truth_path: str, predictions_csv_path: str):
        self.df_true = parse_transcriber_file(ground_truth_path)
        self.df_pred = pd.read_csv(predictions_csv_path)

    def evaluate_performance(self) -> Dict[str, Any]:
        """Ejecuta las métricas de rendimiento."""
        asr_metrics = PerformanceCalculator.calculate_asr_metrics(self.df_true, self.df_pred)
        der_metrics = PerformanceCalculator.calculate_diarization_metrics(self.df_true, self.df_pred)
        gender_metrics = PerformanceCalculator.calculate_gender_metrics(self.df_true, self.df_pred)

        return {
            "Performance Metrics": {
                "ASR": asr_metrics,
                "Diarization": der_metrics,
                "Gender_Classification": gender_metrics
            }
        }

    def evaluate_equity(self) -> Dict[str, Any]:
        """Ejecuta todas las métricas de equidad."""
        speaking_time = EquityCalculator.calculate_speaking_time(self.df_pred)
        interruptions = EquityCalculator.get_male_to_female_interruptions_count(self.df_pred)

        return {
            "Equity_Metrics": {
                "Speaking_Time": speaking_time,
                "Male_to_Female_Interruptions": interruptions
            }
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """Ejecuta todas las métricas: rendimiento y  de equidad."""
        
        return {
            **self.evaluate_performance(),
            **self.evaluate_equity()
        }