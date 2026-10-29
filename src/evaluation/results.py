# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.metrics.equity import EquityMetrics
from src.metrics.performance import PerformanceResult
from src.utils.serialization import flatten_dict, to_serializable

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)

@dataclass
class EquityComparison:
    """
    Equidad medida en la predicción, en la referencia (TRS) opcional y su diferencia.
    """
    prediction: EquityMetrics = field(default_factory=EquityMetrics)
    reference: EquityMetrics = field(default_factory=EquityMetrics)
    error: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, prediction: EquityMetrics, reference: EquityMetrics) -> "EquityComparison":
        p = flatten_dict(prediction.to_dict())
        r = flatten_dict(reference.to_dict())
        err = {k: p[k] - r[k] for k in p if k in r and _is_num(p[k]) and _is_num(r[k])}
        return cls(prediction, reference, err)


@dataclass
class FileEvaluation:
    file: str
    performance: PerformanceResult = field(default_factory=PerformanceResult)
    equity: EquityComparison = field(default_factory=EquityComparison)
    has_reference: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return to_serializable(self)

    def summary_row(self) -> Dict[str, Any]:
        """Fila plana con las métricas clave para la hoja 'Summary' del Excel."""
        p, eq = self.performance, self.equity
        pe, re_ = eq.prediction, eq.reference
        t = p.timing

        return {
            "File": self.file,
            "Audio_min": (t.audio_duration_s / 60.0) if t.available else None,
            "Has_Reference": self.has_reference,
            
            # Rendimiento (Solo válido si hay Ground Truth)
            "ASR_WER": p.asr.wer if self.has_reference else None,
            "ASR_CER": p.asr.cer if self.has_reference else None,
            "DER": p.diarization.der if self.has_reference else None,
            "DER_no_collar": p.diarization.der_no_collar if self.has_reference else None,
            "Gender_Accuracy": p.gender.accuracy if self.has_reference else None,
            "Gender_Sensitivity": p.gender.sensitivity_macro if self.has_reference else None,
            
            # Equidad (Predicción del modelo)
            "Time_Female_Pct": pe.speaking_time.female_pct,
            "Words_Female_Pct": pe.words.female_pct,
            "Speakers_Female_Pct": pe.speakers.female_speaker_pct,
            "Interruptions_M2F": pe.interruptions.male_over_female,
            "Interruptions_F2M": pe.interruptions.female_over_male,
            "Interruption_Asymmetry": pe.interruptions.asymmetry_index,
            "Speaking_Gini": pe.speakers.speaking_time_gini,
            "Opened_By": pe.floor.first_speaker_gender if hasattr(pe, "floor") else None,
            "Closed_By": pe.floor.last_speaker_gender if hasattr(pe, "floor") else None,
            "Longest_Monologue_Duration_s": max(
                (d for d in [pe.turns.max_s_female, pe.turns.max_s_male] if not math.isnan(d)), 
                default=None
            ) if hasattr(pe, "turns") else None,
            
            # Equidad (Diferencia vs Referencia, si existe)
            "Time_Female_Pct_Ref": re_.speaking_time.female_pct if self.has_reference else None,
            "Time_Female_Err_pp": eq.error.get("speaking_time.female_pct") if self.has_reference else None,
            "Interruptions_M2F_Ref": re_.interruptions.male_over_female if self.has_reference else None,
            
            # Tiempos e inferencia
            "Time_total_s": t.total_s if t.available else None,
            "RTF_total": t.rtf_total if t.available else None,
            "Model_load_s": t.model_load_s if t.available else None,
            "Time_ASR_s": t.asr_s if t.available else None,
            "Time_Diarization_s": t.diarization_s if t.available else None,
            "Time_Gender_s": t.gender_s if t.available else None,
            "RAM_peak_MB": t.ram_peak_mb if t.available else None,
            "GPU_peak_MB": t.gpu_peak_mb if t.available else None,
            "Language_detected": t.language_detected if t.available else None,
            "GPU_Index": t.gpu_index if t.available else None,
            "Time_eval_s": sum(p.evaluation_s.values()) if p.evaluation_s else None,
        }