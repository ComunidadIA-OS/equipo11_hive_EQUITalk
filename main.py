import argparse
from src.pipeline.batch_runner import BatchRunner
from src.evaluation.batch_evaluator import BatchEvaluator

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia y Evaluación de Audio Ágora")
    
    # Rutas de entrada requeridas
    parser.add_argument("--audio_dir", type=str, required=True, help="Ruta al directorio de audios (.wav)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ruta al directorio Ground Truth (.trs)")
    
    # Rutas y nombres de salida con valores por defecto
    parser.add_argument("--pred_dir", type=str, default="data/processed", help="Ruta para guardar/leer los CSVs")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Ruta para exportar el Excel de métricas")
    parser.add_argument("--out_filename", type=str, default="evaluation_results.xlsx", help="Nombre del archivo Excel generado")
    
    # Configuración de ejecución
    parser.add_argument("--num_gpus", type=int, default=2, help="Número de GPUs a utilizar")
    parser.add_argument("--skip_inference", action="store_true", help="Salta la inferencia y solo ejecuta la evaluación")
    
    args = parser.parse_args()

    # 1. Inferencia (opcional si se usa --skip_inference)
    if not args.skip_inference:
        print(f"Iniciando inferencia en {args.audio_dir} con {args.num_gpus} GPUs...")
        runner = BatchRunner(
            input_dir=args.audio_dir,
            output_dir=args.pred_dir,
            num_gpus=args.num_gpus
        )
        runner.run()
    else:
        print("Saltando inferencia. Procediendo a evaluación con predicciones existentes...")
    
    # 2. Evaluación
    print("Iniciando cruce de métricas y evaluación...")
    evaluator = BatchEvaluator(
        ground_truth_dir=args.gt_dir,
        predictions_dir=args.pred_dir,
        output_dir=args.out_dir
    )
    evaluator.run(args.out_filename)

if __name__ == "__main__":
    main()