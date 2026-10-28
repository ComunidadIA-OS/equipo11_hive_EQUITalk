import argparse
import logging
from pathlib import Path
from src.pipeline.batch_runner import BatchRunner
from src.evaluation.batch_evaluator import BatchEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de inferencia y evaluación de audios (The Hive)")
    
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directorio con los archivos de audio (.wav) para procesar.")
    
    parser.add_argument("--gt_dir", type=str, default=None, 
                        help="Directorio del Ground Truth (.trs o .xml). Opcional. Si se omite, solo evalúa equidad.")
    
    parser.add_argument("--pred_dir", type=str, default="data/interim", 
                        help="Directorio donde se guardarán/leerán las predicciones temporales (.csv).")
    
    parser.add_argument("--out_dir", type=str, default="data/processed", 
                        help="Directorio raíz para los reportes de resultados finales.")
    
    parser.add_argument("--out_filename", type=str, default="evaluation_run", 
                        help="Nombre de la ejecución o archivo final (se usará como nombre de la carpeta de resultados).")
    
    parser.add_argument("--num_gpus", type=int, default=2, 
                        help="Número de GPUs a utilizar en la inferencia.")
    
    parser.add_argument("--skip_inference", action="store_true", 
                        help="Salta la generación de predicciones y corre directamente la evaluación usando los CSVs en pred_dir.")

    parser.add_argument("--skip_inference", action="store_true", help="Salta la inferencia y solo ejecuta la evaluación")


    return parser.parse_args()

def main():
    args = parse_args()
    
    run_name = args.out_filename.replace(".xlsx", "")

    # 1. Inferencia (opcional si se usa --skip_inference)
    if not args.skip_inference:
        logger.info(f"Iniciando inferencia en {args.audio_dir} con {args.num_gpus} GPUs...")
        runner = BatchRunner(
            input_dir=args.audio_dir,
            output_dir=args.pred_dir,
            num_gpus=args.num_gpus
        )
        runner.run()
    else:
        logger.info(f"Omitiendo inferencia (--skip_inference). Se usarán los CSVs de {args.pred_dir}")

    # 2. Evaluación
    logger.info("Iniciando cruce de métricas y evaluación...")
    evaluator = BatchEvaluator(
        predictions_dir=args.pred_dir,
        ground_truth_dir=args.gt_dir,
        output_root=args.out_dir,
        run_name=run_name
    )
    
    eval_results = evaluator.run()
    
    if eval_results is not None and not eval_results.empty:
        logger.info("Pipeline completado exitosamente. Resultados guardados en la ruta de salida.")
    else:
        logger.warning("El pipeline terminó, pero no se generaron resultados válidos.")

if __name__ == "__main__":
    main()