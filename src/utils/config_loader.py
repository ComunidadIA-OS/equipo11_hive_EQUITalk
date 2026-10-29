# SPDX-License-Identifier: Apache-2.0

import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.exceptions import ConfigurationError

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Carga y valida el archivo de configuración YAML del proyecto.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"No se encontró el archivo de configuración en: {config_path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error al parsear el archivo YAML de configuración: {e}")