import argparse
import logging
from src.pipeline.batch_runner import BatchRunner
from src.evaluation.batch_evaluator import BatchEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de inferencia y evaluación de audio")
    
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directorio con los archivos de audio (.wav) para procesar.")
    
    parser.add_argument("--gt_dir", type=str, default=None, 
                        help="Directorio del Ground Truth (.trs). Opcional. Si no se provee, solo se calcularán métricas de equidad.")
    
    parser.add_argument("--pred_dir", type=str, default="data/processed/predictions", 
                        help="Directorio donde se guardarán (y desde donde se leerán) las predicciones (.csv).")
    
    parser.add_argument("--out_dir", type=str, default="data/processed", 
                        help="Directorio raíz para los reportes finales (Excel, JSON).")
    
    parser.add_argument("--num_gpus", type=int, default=2, 
                        help="Número de GPUs a utilizar en la inferencia.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Fase de Inferencia
    print(f"Iniciando inferencia en {args.audio_dir} con {args.num_gpus} GPUs...")
    runner = BatchRunner(
        input_dir=args.audio_dir,
        output_dir=args.pred_dir,
        num_gpus=args.num_gpus
    )
    runner.run()

    # 2. Fase de Evaluación
    print("Iniciando cruce de métricas y evaluación...")
    
    # CORRECCIÓN APLICADA: output_root en lugar de output_dir
    evaluator = BatchEvaluator(
        predictions_dir=args.pred_dir,
        ground_truth_dir=args.gt_dir,
        output_root=args.out_dir
    )
    evaluator.run()
    
    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()