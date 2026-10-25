import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import concurrent.futures
import pandas as pd
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

def _process_single_audio_worker(args: tuple) -> Dict[str, Any]:
    """
    Función de trabajador ejecutada en un proceso independiente.
    Garantiza el aislamiento del contexto CUDA en la GPU asignada.
    """
    file_path, gpu_id, hf_token, output_dir = args
    file_name = Path(file_path).stem
    
    logger.info(f"Iniciando procesamiento de '{file_name}' en GPU física {gpu_id}")
    
    try:
        # Inicializa el procesador de audio, cargando los modelos exclusivamente en la GPU asignada
        processor = AudioProcessor(gpu_index=gpu_id, hf_token=hf_token)
        
        # Ejecuta el pipeline completo (ASR -> Diarización -> Clasificación)
        df_result = processor.process_file(file_path)
        
        # Guarda el resultado intermedio en CSV
        out_path = Path(output_dir) / f"{file_name}_transcript_speakers.csv"
        os.makedirs(output_dir, exist_ok=True)
        df_result.to_csv(out_path, index=False)
        
        logger.info(f"Procesamiento exitoso: {file_name} -> {out_path}")
        return {
            "file": file_path, 
            "status": "success", 
            "output_path": str(out_path)
        }
        
    except Exception as e:
        logger.error(f"Error crítico procesando {file_path} en GPU {gpu_id}: {e}", exc_info=True)
        return {
            "file": file_path, 
            "status": "error", 
            "error_message": str(e)
        }


class BatchRunner:
    """
    Orquestador para el procesamiento por lotes (batch processing) de múltiples archivos de audio,
    distribuyendo las tareas concurrentemente entre los nodos de hardware disponibles.
    
    """
    
    def __init__(
        self, 
        input_dir: str, 
        output_dir: str, 
        num_gpus: int = 2, 
        hf_token: Optional[str] = None
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("Se requiere el Token de Hugging Face (HF_TOKEN) para autenticar pyannote.audio.")

    def discover_audio_files(self) -> List[str]:
        """
        Escanea recursivamente el directorio de entrada en busca de archivos de audio WAV.
        
        Returns:
            List[str]: Lista con las rutas absolutas o relativas de los audios encontrados.
        """
        wav_files = sorted([str(p) for p in self.input_dir.glob("**/*.wav")])
        logger.info(f"Se encontraron {len(wav_files)} archivos WAV en el directorio: {self.input_dir}")
        return wav_files

    def run(self) -> List[Dict[str, Any]]:
        """
        Ejecuta el pipeline por lotes en paralelo utilizando un pool de procesos.
        Cada proceso maneja una GPU de forma aislada para evitar colisiones de VRAM.
        
        Returns:
            List[Dict[str, Any]]: Resumen con el estado de ejecución por cada archivo.
        """
        audio_files = self.discover_audio_files()
        if not audio_files:
            logger.warning("No se encontraron archivos de audio para procesar en la ruta especificada.")
            return []

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Asignación balanceada (Round-Robin) de GPUs: Archivo 0 -> GPU 0, Archivo 1 -> GPU 1, Archivo 2 -> GPU 0...
        tasks = []
        for idx, file_path in enumerate(audio_files):
            gpu_id = idx % self.num_gpus
            tasks.append((file_path, gpu_id, self.hf_token, str(self.output_dir)))

        results: List[Dict[str, Any]] = []
        logger.info(f"Iniciando batch runner para {len(audio_files)} archivos usando {self.num_gpus} GPUs concurrentes...")
        
        # ProcessPoolExecutor aísla el espacio de memoria de cada GPU por proceso operativo
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_file = {
                executor.submit(_process_single_audio_worker, task): task[0] 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"El hilo principal capturó un fallo inesperado para el archivo {file_path}: {exc}")
                    results.append({
                        "file": file_path, 
                        "status": "critical_error", 
                        "error_message": str(exc)
                    })
                    
        logger.info("Procesamiento por lotes completado en su totalidad.")
        return results