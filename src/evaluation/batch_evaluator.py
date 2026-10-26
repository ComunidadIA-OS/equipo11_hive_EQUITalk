import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from src.evaluation.evaluator import Evaluator

class BatchEvaluator:
    """
    Orquesta la evaluación masiva de un directorio de archivos Ground Truth (.trs)
    contra sus respectivas predicciones (.csv), exportando el resumen a Excel.
    """
    
    def __init__(self, ground_truth_dir: str, predictions_dir: str, output_dir: str = "data/processed"):
        self.gt_dir = Path(ground_truth_dir)
        self.pred_dir = Path(predictions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, output_filename: str = "agora_evaluation_results.xlsx") -> pd.DataFrame:
        results = []
        
        # Escanear recursivamente los archivos de referencia TRS
        trs_files = list(self.gt_dir.glob("**/*.trs"))
        
        for trs_path in trs_files:
            base_name = trs_path.stem  # ej: 'agora_2007_11_05_a'
            
            # Reconstruir el nombre del CSV generado por batch_runner
            pred_filename = f"{base_name}_transcript_speakers.csv"
            pred_path = self.pred_dir / pred_filename
            
            if not pred_path.exists():
                print(f"Omitiendo {base_name}: No se encontró el CSV de predicción en {pred_path}")
                continue
                
            try:
                evaluator = Evaluator(str(trs_path), str(pred_path))
                file_metrics = evaluator.evaluate_all()
                flat_metrics = self._flatten_metrics(base_name, file_metrics)
                results.append(flat_metrics)
            except Exception as e:
                print(f"Error evaluando {base_name}: {str(e)}")

        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Calcular una fila de promedios para todas las métricas numéricas
            mean_row = df_results.mean(numeric_only=True).to_dict()
            mean_row["File"] = "AVERAGE"
            df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
            
            # Exportar a Excel
            out_path = self.output_dir / output_filename
            df_results.to_excel(out_path, index=False, float_format="%.4f")
            print(f"Evaluación completada. Resultados guardados en: {out_path}")
            
        return df_results

    def _flatten_metrics(self, filename: str, metrics: Dict[str, Any]) -> dict:
        """Aplana el diccionario anidado para convertirlo en columnas de Pandas."""
        flat = {"File": filename}
        
        # Rendimiento
        perf = metrics.get("Performance Metrics", {})
        flat["ASR_WER"] = perf.get("ASR", {}).get("WER", None)
        flat["ASR_CER"] = perf.get("ASR", {}).get("CER", None)
        flat["DER (Diarization)"] = perf.get("Diarization", {}).get("DER", None)
        
        gender = perf.get("Gender_Classification", {})
        flat["Gender_Accuracy"] = gender.get("accuracy", None)
        flat["Gender_Sensitivity"] = gender.get("sensitivity_macro", None)
        flat["Gender_Precision"] = gender.get("precision_macro", None)
        
        # Equidad
        equity = metrics.get("Equity_Metrics", {})
        spk_time = equity.get("Speaking_Time", {})
        flat["Time_Female_Pct"] = spk_time.get("female_pct", None)
        flat["Time_Male_Pct"] = spk_time.get("male_pct", None)
        flat["Interruptions_M2F"] = equity.get("Male_to_Female_Interruptions", None)
        
        return flat