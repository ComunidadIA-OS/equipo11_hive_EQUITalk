import json
import time
from types import SimpleNamespace

import pytest

from src.utils.profiling import ResourceProfiler


def test_stage_mide_segundos_por_etapa():
    prof = ResourceProfiler(use_gpu=False)
    with prof.stage("asr"):
        time.sleep(0.05)
    with prof.stage("gender"):
        pass

    assert list(prof.stage_seconds) == ["asr", "gender"]
    assert prof.stage_seconds["asr"] >= 0.04
    assert prof.stage_seconds["gender"] < prof.stage_seconds["asr"]


def test_stage_mide_ram():
    prof = ResourceProfiler(use_gpu=False)
    with prof.stage("asr"):
        datos = bytearray(20 * 1024 * 1024)  # reservamos 20 MB
    del datos

    assert prof.ram_peak_mb is not None
    assert prof.ram_peak_mb >= 20


def test_stage_registra_tiempo_aunque_haya_error():
    prof = ResourceProfiler(use_gpu=False)
    with pytest.raises(ValueError):
        with prof.stage("diarization"):
            raise ValueError("fallo simulado")

    assert "diarization" in prof.stage_seconds


def test_summary_sin_gpu_es_json_valido():
    prof = ResourceProfiler(use_gpu=False)
    with prof.stage("asr"):
        pass
    summary = prof.summary()

    assert prof.gpu_enabled is False
    assert summary["gpu_peak_mb"] is None
    assert summary["gpu_torch_peak_mb_by_stage"] == {}
    assert summary["gpu_device_used_mb_by_stage"] == {}
    assert set(summary["stage_seconds"]) == {"asr"}
    json.dumps(summary, allow_nan=False)  # sin NaN: se puede guardar como JSON estándar


class FakeCuda:
    """Imita torch.cuda para probar la medición de GPU sin tener tarjeta gráfica."""
    MB = 1024 * 1024

    def __init__(self, torch_peaks_mb, device_used_mb, total_mb=16000):
        self._peaks = iter(torch_peaks_mb)   # un valor por etapa
        self._used = iter(device_used_mb)
        self._total = total_mb
        self.resets = 0

    def reset_peak_memory_stats(self):
        self.resets += 1

    def synchronize(self):
        pass

    def max_memory_reserved(self):
        return next(self._peaks) * self.MB

    def mem_get_info(self):
        used = next(self._used)
        return (self._total - used) * self.MB, self._total * self.MB  # (libre, total)


def test_stage_mide_gpu_por_etapa():
    cuda = FakeCuda(torch_peaks_mb=[500, 3000], device_used_mb=[5200, 6100])
    prof = ResourceProfiler(use_gpu=False)
    prof._torch = SimpleNamespace(cuda=cuda)   # inyectamos la GPU simulada

    with prof.stage("asr"):          # Whisper: PyTorch apenas lo ve (500), la tarjeta sí (5200)
        pass
    with prof.stage("diarization"):
        pass
    summary = prof.summary()

    assert cuda.resets == 2                     # el pico de PyTorch se reinicia en cada etapa
    assert summary["gpu_torch_peak_mb_by_stage"] == {"asr": 500.0, "diarization": 3000.0}
    assert summary["gpu_device_used_mb_by_stage"] == {"asr": 5200.0, "diarization": 6100.0}
    assert summary["gpu_peak_mb"] == 6100.0     # la mayor medida de todas las etapas
