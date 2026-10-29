# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from pathlib import Path

from src.pipeline.batch_runner import BatchRunner
from src.evaluation.batch_evaluator import BatchEvaluator
from src.utils.paths import RunPaths

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de inferencia y evaluación de audios"
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directorio con los archivos de audio (.wav) para procesar. "
             "Obligatorio salvo que se use --skip_inference.",
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directorio del Ground Truth (.trs o .xml). Opcional. Si se omite, solo evalúa equidad.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Directorio raíz donde se creará la carpeta de la ejecución.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="evaluation_run",
        help="Nombre de la ejecución (se creará la carpeta <out_dir>/<run_name>/).",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Número de GPUs a utilizar en la inferencia.",
    )

    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Salta la generación de predicciones y corre directamente la evaluación usando los datos existentes.",
    )

    args = parser.parse_args()

    if not args.skip_inference and not args.audio_dir:
        parser.error("--audio_dir es obligatorio a menos que se use --skip_inference.")

    return args


def main():
    args = parse_args()

    # Normalizar el nombre del run (por si se pasa una extensión por error)
    run_name = args.run_name.replace(".xlsx", "").replace(".json", "")

    # Crear la estructura de directorios del run mediante RunPaths
    paths = RunPaths.create(output_root=args.out_dir, run_name=run_name)
    logger.info(f"Directorio de la ejecución: {paths.run_dir}")

    # 1. Inferencia (opcional si se usa --skip_inference)
    if not args.skip_inference:
        logger.info(f"Iniciando inferencia en {args.audio_dir} con {args.num_gpus} GPUs...")
        runner = BatchRunner(
            input_dir=args.audio_dir,
            output_dir=str(paths.run_dir),  
            num_gpus=args.num_gpus,
        )
        runner.run()
    else:
        logger.info(f"Omitiendo inferencia (--skip_inference). Se usará {paths.predictions_dir}")

    # 2. Evaluación
    logger.info("Iniciando cruce de métricas y evaluación...")
    evaluator = BatchEvaluator(
        predictions_dir=str(paths.predictions_dir),
        ground_truth_dir=args.gt_dir,
        output_root=args.out_dir,
        run_name=run_name,
    )

    eval_results = evaluator.run()

    if eval_results is not None and not eval_results.empty:
        logger.info(f"Pipeline completado exitosamente. Resultados guardados en: {paths.run_dir}")
    else:
        logger.warning("El pipeline terminó, pero no se generaron resultados válidos.")


if __name__ == "__main__":
    main()