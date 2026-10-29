# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path

def setup_logger(name: str = "gender_voice_metrics", log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configura y retorna un logger estandarizado para consola y archivo.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(processName)s:%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para consola (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handler para archivo 
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger