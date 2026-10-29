# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import jiwer

from src.utils.serialization import to_serializable
from src.utils.text_normalizer import basic_normalize, normalize_text

NAN = float("nan")

# Clases de resultados
# ----------------------------------------------------------------------------
@dataclass
class ASRMetrics:
    wer: float = NAN                 # WER con texto normalizado
    cer: float = NAN
    ref_words: int = 0
    hyp_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    ref_chars: int = 0
    char_errors: int = 0


@dataclass
class DiarizationMetrics:
    der: float = NAN                 # con collar
    der_no_collar: float = NAN       # criterio estricto (collar=0)
    miss_s: float = NAN              # habla de la referencia no detectada
    false_alarm_s: float = NAN
    confusion_s: float = NAN
    ref_speech_s: float = NAN
    miss_pct: float = NAN            # componentes como % del habla de referencia
    false_alarm_pct: float = NAN
    confusion_pct: float = NAN
    n_ref_speakers: int = 0
    n_hyp_speakers: int = 0
    collar_s: float = 0.25
    hypothesis_source: str = ""      # 'diarization_turns' | 'asr_segments'


@dataclass
class GenderMetrics:
    """
    Evaluación por tiempo (rejilla de ``resolution_s``) contra el género del TRS.
    """
    accuracy: float = NAN
    accuracy_overall: float = NAN    # cuenta unknown/sin detección como error
    sensitivity_macro: float = NAN
    precision_macro: float = NAN
    recall_female: float = NAN
    recall_male: float = NAN
    precision_female: float = NAN
    precision_male: float = NAN
    coverage: float = NAN
    evaluated_s: float = 0.0
    resolution_s: float = 0.05
    confusion_s: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TimingMetrics:
    """
    Tiempos y configuración de la inferencia extraídos de los metadatos del AudioProcessor.
    """
    available: bool = False
    audio_duration_s: float = NAN
    total_s: float = NAN                         
    rtf_total: float = NAN                       
    model_load_s: float = NAN
    asr_s: float = NAN                           # segundos por etapa
    diarization_s: float = NAN
    gender_s: float = NAN
    ram_peak_mb: float = NAN                     # pico de RAM del proceso
    gpu_peak_mb: float = NAN                     # memoria de GPU necesaria (máximo entre etapas)
    language_requested: Optional[str] = None
    language_detected: Optional[str] = None
    language_probability: Optional[float] = None
    gpu_index: Optional[int] = None


@dataclass
class PerformanceResult:
    asr: ASRMetrics = field(default_factory=ASRMetrics)
    diarization: DiarizationMetrics = field(default_factory=DiarizationMetrics)
    gender: GenderMetrics = field(default_factory=GenderMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    evaluation_s: Dict[str, float] = field(default_factory=dict)  # tiempo de la propia evaluación

    def to_dict(self) -> dict:
        return to_serializable(self)


# Auxiliares
# ----------------------------------------------------------------------------
def _error_components(ref: str, hyp: str, chars: bool = False) -> Dict[str, int]:
    if hasattr(jiwer, "process_words"):
        out = jiwer.process_characters(ref, hyp) if chars else jiwer.process_words(ref, hyp)
        return {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "H": out.hits}
    if chars:
        n = len(ref)
        errs = round(jiwer.cer(ref, hyp) * n)
        return {"S": errs, "D": 0, "I": 0, "H": n - errs}
    m = jiwer.compute_measures(ref, hyp)
    return {"S": m["substitutions"], "D": m["deletions"], "I": m["insertions"], "H": m["hits"]}


def gender_metrics_from_confusion(conf: Dict[str, Dict[str, float]], resolution_s: float) -> GenderMetrics:
    f = lambda t, p: float(conf.get(t, {}).get(p, 0.0))
    classes = ("female", "male")
    decided_total = sum(f(t, p) for t in classes for p in classes)
    gt_total = sum(sum(conf.get(t, {}).values()) for t in classes)
    gm = GenderMetrics(resolution_s=resolution_s, confusion_s=conf, evaluated_s=gt_total)
    if gt_total <= 0:
        return gm
    correct = sum(f(c, c) for c in classes)
    gm.accuracy = correct / decided_total if decided_total > 0 else NAN
    gm.accuracy_overall = correct / gt_total
    gm.coverage = decided_total / gt_total
    rec, prec = {}, {}
    for c in classes:
        gt_c = sum(f(c, p) for p in classes)
        pr_c = sum(f(t, c) for t in classes)
        rec[c] = f(c, c) / gt_c if gt_c > 0 else NAN
        prec[c] = f(c, c) / pr_c if pr_c > 0 else NAN
    gm.recall_female, gm.recall_male = rec["female"], rec["male"]
    gm.precision_female, gm.precision_male = prec["female"], prec["male"]
    gm.sensitivity_macro = float(np.nanmean(list(rec.values()))) if not all(np.isnan(list(rec.values()))) else NAN
    gm.precision_macro = float(np.nanmean(list(prec.values()))) if not all(np.isnan(list(prec.values()))) else NAN
    return gm

def _num(x) -> float:
    """Convierte a float; None o valores ausentes pasan a NaN."""
    return float(x) if x is not None else NAN


_BITS = {"female": 1, "male": 2}
_PRED_LABEL = {0: "no_speech", 1: "female", 2: "male", 4: "unknown"}

def _paint(df: pd.DataFrame, n: int, res: float) -> np.ndarray:
    arr = np.zeros(n, dtype=np.uint8)
    for s, e, g in zip(df["start_time"], df["end_time"], df["gender"].astype(str).str.lower()):
        i0, i1 = int(round(s / res)), int(round(e / res))
        if i1 > i0:
            arr[i0:i1] |= _BITS.get(g, 4)
    return arr

# Calculadora
# ----------------------------------------------------------------------------
class PerformanceCalculator:

    # ASR 
    @staticmethod
    def calculate_asr_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame,
                              strip_accents: bool = False) -> ASRMetrics:
        def join(df: pd.DataFrame) -> str:
            if df.empty:
                return ""
            d = df.sort_values("start_time")
            return " ".join(d["transcript"].fillna("").astype(str).tolist())

        ref_raw, hyp_raw = join(df_true), join(df_pred)
        ref = normalize_text(ref_raw, strip_accents)
        hyp = normalize_text(hyp_raw, strip_accents)
        m = ASRMetrics(ref_words=len(ref.split()), hyp_words=len(hyp.split()), ref_chars=len(ref))
        if not ref:
            m.wer = m.cer = 0.0 if not hyp else 1.0
            return m

        w = _error_components(ref, hyp) if hyp else {"S": 0, "D": m.ref_words, "I": 0, "H": 0}
        c = _error_components(ref, hyp, chars=True) if hyp else {"S": 0, "D": len(ref), "I": 0, "H": 0}
        m.substitutions, m.deletions, m.insertions, m.hits = w["S"], w["D"], w["I"], w["H"]
        m.wer = (w["S"] + w["D"] + w["I"]) / m.ref_words
        m.char_errors = c["S"] + c["D"] + c["I"]
        m.cer = m.char_errors / max(len(ref), 1)

        return m

    # Diarización 
    @staticmethod
    def calculate_diarization_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame,
                                      collar: float = 0.25, skip_overlap: bool = False,
                                      hypothesis_source: str = "diarization_turns",
                                      dur: Optional[float] = None) -> DiarizationMetrics:
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate

        def to_annotation(df: pd.DataFrame) -> Annotation:
            ann = Annotation()
            for i, (s, e, spk) in enumerate(zip(df["start_time"], df["end_time"], df["speaker_label"].astype(str))):
                if e > s:
                    ann[Segment(s, e), i] = spk
            return ann

        ref, hyp = to_annotation(df_true), to_annotation(df_pred)
        dm = DiarizationMetrics(collar_s=collar, hypothesis_source=hypothesis_source,
                                n_ref_speakers=len(ref.labels()), n_hyp_speakers=len(hyp.labels()))
        if len(ref) == 0:
            dm.der = dm.der_no_collar = 1.0 if len(hyp) > 0 else 0.0
            return dm

        if dur is not None:
            uem = Timeline([Segment(0, dur)])
        else:
            lo = min(ref.get_timeline().extent().start, hyp.get_timeline().extent().start if len(hyp) else np.inf)
            hi = max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end if len(hyp) else 0.0)
            uem = Timeline([Segment(lo, hi)])

        def run(col: float) -> dict:
            metric = DiarizationErrorRate(collar=col, skip_overlap=skip_overlap)
            return metric(ref, hyp, uem=uem, detailed=True)

        comp = run(collar)
        total = comp["total"]
        errors = comp["missed detection"] + comp["false alarm"] + comp["confusion"]
        dm.der = errors / total if total > 0 else NAN
        dm.miss_s, dm.false_alarm_s, dm.confusion_s = comp["missed detection"], comp["false alarm"], comp["confusion"]
        dm.ref_speech_s = total
        if total > 0:
            dm.miss_pct, dm.false_alarm_pct, dm.confusion_pct = (100 * x / total for x in (dm.miss_s, dm.false_alarm_s, dm.confusion_s))
        c0 = run(0.0) if collar > 0 else comp
        dm.der_no_collar = (c0["missed detection"] + c0["false alarm"] + c0["confusion"]) / c0["total"] if c0["total"] > 0 else NAN
        return dm

    # Género 
    @staticmethod
    def calculate_gender_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame,
                                 resolution_s: float = 0.05) -> GenderMetrics:
        if df_true.empty:
            return GenderMetrics(resolution_s=resolution_s)
        end = max(df_true["end_time"].max(), df_pred["end_time"].max() if not df_pred.empty else 0.0)
        n = int(np.ceil(end / resolution_s)) + 2
        gt, pr = _paint(df_true, n, resolution_s), _paint(df_pred, n, resolution_s)
        mask = np.isin(gt, (1, 2))
        conf: Dict[str, Dict[str, float]] = {}
        for tv, tname in ((1, "female"), (2, "male")):
            row = {}
            sel = pr[mask & (gt == tv)]
            for pv, pname in _PRED_LABEL.items():
                row[pname] = float((sel == pv).sum() * resolution_s)
            row["overlap_or_mixed"] = float((~np.isin(sel, list(_PRED_LABEL))).sum() * resolution_s)
            conf[tname] = row
        return gender_metrics_from_confusion(conf, resolution_s)

    # Tiempos 
    @staticmethod
    def calculate_timing_metrics(metadata: Optional[dict]) -> TimingMetrics:
        """
        Convierte los metadatos del AudioProcessor en TimingMetrics.
        """
        if not metadata:
            return TimingMetrics()
            
        audio_dur = metadata.get("audio_duration_seconds", NAN)
        total_sec = metadata.get("processing_seconds", NAN)
        
        rtf = NAN
        if audio_dur > 0 and not np.isnan(total_sec):
            rtf = total_sec / audio_dur
            
        stages = metadata.get("stage_seconds") or {}

        return TimingMetrics(
            available=True,
            audio_duration_s=audio_dur,
            total_s=total_sec,
            rtf_total=rtf,
            model_load_s=_num(metadata.get("model_load_seconds")),
            asr_s=_num(stages.get("asr")),
            diarization_s=_num(stages.get("diarization")),
            gender_s=_num(stages.get("gender")),
            ram_peak_mb=_num(metadata.get("ram_peak_mb")),
            gpu_peak_mb=_num(metadata.get("gpu_peak_mb")),
            language_requested=metadata.get("language_requested"),
            language_detected=metadata.get("language_detected"),
            language_probability=metadata.get("language_probability"),
            gpu_index=metadata.get("gpu_index")
        )