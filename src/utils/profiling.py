# SPDX-License-Identifier: Apache-2.0
"""Medición de tiempo y memoria (RAM y GPU) por etapa del pipeline.

Uso:
    prof = ResourceProfiler()
    with prof.stage("asr"):
        ...  # código a medir
    prof.summary()  # -> dict apto para guardar en JSON

Qué se mide en cada etapa:
- Segundos de reloj.
- GPU (si hay CUDA), con dos medidas complementarias:
  * pico de PyTorch dentro de la etapa (exacto, pero solo ve pyannote y wav2vec2);
  * memoria ocupada en toda la tarjeta al terminar la etapa (ve todas las librerías,
    también faster-whisper, que no usa PyTorch sino CTranslate2).
- RAM: pico del proceso desde que arrancó (incluye los modelos cargados). El sistema
  operativo no permite reiniciar ese contador, así que no se puede dar por etapa.
"""
import time
from contextlib import contextmanager
from typing import Dict, Optional

import psutil

MB = 1024 * 1024


def _ram_peak_mb() -> float:
    """Pico de RAM del proceso desde que arrancó, en MB."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows guarda el pico directamente
    if peak is None:
        try:  # Linux: ru_maxrss viene en KB
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:  # sin forma de conocer el pico: usamos el consumo actual
            peak = info.rss
    return max(peak, info.rss) / MB


def _cuda_torch():
    """Devuelve el módulo torch si hay una GPU CUDA disponible; si no, None."""
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


class ResourceProfiler:
    """Mide segundos y memoria (GPU y RAM) de cada etapa con `with prof.stage(nombre):`."""

    def __init__(self, use_gpu: Optional[bool] = None):
        # use_gpu=None -> detectar automáticamente; False -> no medir GPU
        self._torch = None if use_gpu is False else _cuda_torch()
        self.stage_seconds: Dict[str, float] = {}
        self.gpu_torch_peak_mb_by_stage: Dict[str, float] = {}
        self.gpu_device_used_mb_by_stage: Dict[str, float] = {}
        self.ram_peak_mb: Optional[float] = None

    @property
    def gpu_enabled(self) -> bool:
        return self._torch is not None

    @contextmanager
    def stage(self, name: str):
        cuda = self._torch.cuda if self.gpu_enabled else None
        if cuda:
            cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                cuda.synchronize()  # esperar a que la GPU termine antes de parar el reloj
            self.stage_seconds[name] = round(time.perf_counter() - t0, 3)
            if cuda:
                self.gpu_torch_peak_mb_by_stage[name] = round(cuda.max_memory_reserved() / MB, 1)
                free, total = cuda.mem_get_info()
                self.gpu_device_used_mb_by_stage[name] = round((total - free) / MB, 1)
            self.ram_peak_mb = round(_ram_peak_mb(), 1)

    def summary(self) -> dict:
        """Resultados en un dict apto para JSON (None cuando no hay dato).

        gpu_peak_mb es la mayor de las dos medidas de GPU en cualquier etapa: la mejor
        estimación de cuánta memoria de tarjeta gráfica necesita el pipeline.
        """
        gpu_values = list(self.gpu_torch_peak_mb_by_stage.values()) + \
            list(self.gpu_device_used_mb_by_stage.values())
        return {
            "stage_seconds": dict(self.stage_seconds),
            "ram_peak_mb": self.ram_peak_mb,
            "gpu_peak_mb": max(gpu_values) if gpu_values else None,
            "gpu_torch_peak_mb_by_stage": dict(self.gpu_torch_peak_mb_by_stage),
            "gpu_device_used_mb_by_stage": dict(self.gpu_device_used_mb_by_stage),
        }
