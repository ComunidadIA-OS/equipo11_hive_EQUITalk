import os
import json
import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Any
import concurrent.futures
import pandas as pd
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Variable global a nivel de subproceso (worker) para mantener los modelos en memoria
_worker_processor = None

def _init_worker(gpu_queue, hf_token):
    """
    Inicializa los modelos una sola vez por cada proceso (worker) asignado a una GPU.
    """
    global _worker_processor
    # Extraer una GPU de la cola asignada para este proceso específico
    gpu_id = gpu_queue.get()
    logger.info(f"Inicializando Worker en GPU {gpu_id} y cargando modelos...")
    
    # La variable CUDA_VISIBLE_DEVICES quedará anclada de forma segura para este worker
    _worker_processor = AudioProcessor(gpu_index=gpu_id, hf_token=hf_token)

def _process_single_audio_worker(args: tuple) -> Dict[str, Any]:
    """
    Función de trabajador. Utiliza los modelos ya precargados en RAM/VRAM.
    """
    global _worker_processor
    file_path, output_dir = args
    file_name = Path(file_path).stem
    
    logger.info(f"Iniciando procesamiento de '{file_name}'")
    
    try:
        # Usa la instancia estática global que ya cargó los modelos en _init_worker
        output = _worker_processor.process_file(file_path)
        
        # process_file devuelve tres cosas: segmentos (ASR), turnos (diarización) y metadatos
        out_dir = Path(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"{file_name}_transcript_speakers.csv"
        output.segments.to_csv(out_path, index=False)
        output.turns.to_csv(out_dir / f"{file_name}_diarization.csv", index=False)
        with open(out_dir / f"{file_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(output.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Procesamiento exitoso: {file_name} -> {out_path}")
        return {"file": file_path, "status": "success", "output_path": str(out_path)}
        
    except Exception as e:
        logger.error(f"Error crítico procesando {file_path}: {e}", exc_info=True)
        return {"file": file_path, "status": "error", "error_message": str(e)}

class BatchRunner:
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
        wav_files = sorted([str(p) for p in self.input_dir.glob("**/*.wav")])
        logger.info(f"Se encontraron {len(wav_files)} archivos WAV en el directorio: {self.input_dir}")
        return wav_files

    def run(self) -> List[Dict[str, Any]]:
        audio_files = self.discover_audio_files()
        if not audio_files:
            return []

        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Preparar la cola de GPUs para que cada worker tome su ID al arrancar
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        for i in range(self.num_gpus):
            gpu_queue.put(i)
            
        # 2. Las tareas ahora solo necesitan el archivo y la carpeta destino
        tasks = [(file_path, str(self.output_dir)) for file_path in audio_files]
        results: List[Dict[str, Any]] = []
        
        logger.info(f"Iniciando batch runner para {len(audio_files)} archivos usando {self.num_gpus} GPUs concurrentes...")
        
        # 3. Utilizar initializer para ejecutar _init_worker al levantar cada subproceso
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_gpus,
            initializer=_init_worker,
            initargs=(gpu_queue, self.hf_token)
        ) as executor:
            
            future_to_file = {executor.submit(_process_single_audio_worker, task): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"El hilo principal capturó un fallo inesperado para el archivo {file_path}: {exc}")
                    results.append({"file": file_path, "status": "critical_error", "error_message": str(exc)})
                    
        logger.info("Procesamiento por lotes completado en su totalidad.")
        return results