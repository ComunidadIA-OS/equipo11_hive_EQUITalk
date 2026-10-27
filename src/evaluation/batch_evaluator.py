import json
from pathlib import Path
import pandas as pd
from typing import Optional

from src.evaluation.evaluator import Evaluator
from src.evaluation.report import ResultsWriter
from src.utils.paths import RunPaths

class BatchEvaluator:
    """
    Orquesta la evaluación masiva de un directorio de predicciones (.csv),
    cruzándolas con su Ground Truth (.trs) si está disponible.
    """
    
    def __init__(self, predictions_dir: str, ground_truth_dir: Optional[str] = None, output_root: str = "data/processed", run_name: str = "evaluation"):
        self.pred_dir = Path(predictions_dir)
        self.gt_dir = Path(ground_truth_dir) if ground_truth_dir else None
        
        # Inicializamos RunPaths usando su método de clase oficial
        self.paths = RunPaths.create(output_root=output_root, run_name=run_name)

    def run(self) -> pd.DataFrame:
        results = []
        evaluations_json = {}
        
        csv_files = list(self.pred_dir.glob("**/*_transcript_speakers.csv"))
        
        for csv_path in csv_files:
            base_name = csv_path.name.replace("_transcript_speakers.csv", "")
            
            meta_path = csv_path.with_name(f"{base_name}_metadata.json")
            metadata = None
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            
            trs_path = None
            if self.gt_dir:
                candidates = list(self.gt_dir.glob(f"**/{base_name}.trs"))
                if candidates:
                    trs_path = candidates[0]
            
            try:
                evaluator = Evaluator(
                    predictions_csv_path=str(csv_path),
                    ground_truth_path=str(trs_path) if trs_path else None,
                    metadata=metadata
                )
                file_eval = evaluator.analyze()
                
                results.append(file_eval.summary_row())
                evaluations_json[base_name] = file_eval.to_dict()
                
            except Exception as e:
                print(f"Error procesando {base_name}: {str(e)}")

        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            mean_row = df_results.mean(numeric_only=True).to_dict()
            mean_row["File"] = "AVERAGE"
            mean_row["Has_Reference"] = df_results["Has_Reference"].any()
            df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
            
            writer = ResultsWriter(self.paths)
            writer.write_excel({"Summary": df_results})
            writer.write_json(evaluations_json)
            writer.print_summary(df_results)
            
        return df_results