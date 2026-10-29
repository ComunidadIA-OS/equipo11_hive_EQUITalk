# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.serialization import to_serializable

KNOWN = ("female", "male")
NAN = float("nan")


# Clases de resultados
# ----------------------------------------------------------------------------
@dataclass
class SpeakingTimeMetrics:
    female_s: float = 0.0
    male_s: float = 0.0
    unknown_s: float = 0.0          # género no determinado (unknown/child)
    female_pct: float = NAN         # sobre female+male (métrica clásica)
    male_pct: float = NAN
    unknown_pct: float = NAN        # sobre todo el habla
    known_coverage_pct: float = NAN # % del habla con género asignado
    overlap_s: float = 0.0          # tiempo con >=2 hablantes a la vez
    overlap_pct: float = NAN        # sobre el tiempo con habla


@dataclass
class WordMetrics:
    female: int = 0
    male: int = 0
    unknown: int = 0
    female_pct: float = NAN         # sobre female+male
    male_pct: float = NAN


@dataclass
class SpeakerMetrics:
    n_female: int = 0
    n_male: int = 0
    n_unknown: int = 0
    female_speaker_pct: float = NAN         # presencia: % de hablantes mujeres
    mean_time_female_s: float = NAN         # tiempo medio por hablante
    mean_time_male_s: float = NAN
    time_per_speaker_ratio_f_m: float = NAN # >1: cada mujer habla más que cada hombre
    participation_gap_pp: float = NAN       # %tiempo mujeres - %hablantes mujeres (pp)
    speaking_time_gini: float = NAN         # 0 reparto igual, ->1 un hablante domina
    top_speaker_gender: str = ""
    top_speaker_share_pct: float = NAN


@dataclass
class TurnMetrics:
    n_female: int = 0
    n_male: int = 0
    n_unknown: int = 0
    female_pct: float = NAN
    mean_s_female: float = NAN
    mean_s_male: float = NAN
    median_s_female: float = NAN
    median_s_male: float = NAN
    max_s_female: float = NAN               # monólogo más largo
    max_s_male: float = NAN
    mean_ratio_f_m: float = NAN
    transitions: Dict[str, int] = field(default_factory=dict)  # quién toma la palabra tras quién


@dataclass
class FloorMetrics:
    first_speaker_gender: str = ""          # quién abre
    last_speaker_gender: str = ""           # quién cierra
    first_female_onset_s: float = NAN       # segundos hasta la primera voz femenina
    first_female_onset_pct: float = NAN     # ídem, % de la duración del programa


@dataclass
class InterruptionMetrics:
    threshold_s: float = 0.5
    total: int = 0
    male_over_female: int = 0
    female_over_male: int = 0
    male_over_male: int = 0
    female_over_female: int = 0
    floor_taking_male_over_female: int = 0   # el interruptor acaba tomando la palabra
    floor_taking_female_over_male: int = 0
    asymmetry_index: float = NAN             # (M/F - F/M)/(M/F + F/M): >0 hombres interrumpen más a mujeres
    made_per_hour_female: float = NAN
    made_per_hour_male: float = NAN
    received_per_hour_female: float = NAN
    received_per_hour_male: float = NAN
    made_per_turn_female: float = NAN        # normalizado por nº de turnos propios
    made_per_turn_male: float = NAN
    received_per_turn_female: float = NAN
    received_per_turn_male: float = NAN


@dataclass
class TemporalMetrics:
    female_time_pct_by_quarter: Dict[str, float] = field(default_factory=dict)  # q1..q4


@dataclass
class EquityMetrics:
    span_s: float = 0.0
    total_speech_s: float = 0.0
    time_source: str = ""                    # 'diarization_turns' | 'asr_segments'
    speaking_time: SpeakingTimeMetrics = field(default_factory=SpeakingTimeMetrics)
    words: WordMetrics = field(default_factory=WordMetrics)
    speakers: SpeakerMetrics = field(default_factory=SpeakerMetrics)
    turns: TurnMetrics = field(default_factory=TurnMetrics)
    floor: FloorMetrics = field(default_factory=FloorMetrics)
    interruptions: InterruptionMetrics = field(default_factory=InterruptionMetrics)
    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)

    def to_dict(self) -> dict:
        return to_serializable(self)


# Utilidades
# ----------------------------------------------------------------------------
def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b > 0 else NAN


def _ratio(a: float, b: float) -> float:
    return a / b if b and not np.isnan(a) and not np.isnan(b) else NAN


def _gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    n, s = len(v), v.sum()
    if n == 0 or s <= 0:
        return NAN
    return float((2 * np.sum(np.arange(1, n + 1) * v)) / (n * s) - (n + 1) / n)


def _stat(series: pd.Series, fn) -> float:
    return float(fn(series)) if len(series) else NAN


# Calculadora
# ----------------------------------------------------------------------------
class EquityCalculator:
    """
    Calcula métricas de equidad e interacciones a partir de segmentos/turnos.
    """

    @staticmethod
    def _prepare(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        cols = ["start_time", "end_time", "speaker_label", "gender", "transcript"]
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols + ["duration"])
        out = df.copy()
        if "transcript" not in out.columns:
            out["transcript"] = ""
        out["transcript"] = out["transcript"].fillna("").astype(str)
        out["speaker_label"] = out["speaker_label"].astype(str)
        g = out["gender"].astype(str).str.strip().str.lower()
        out["gender"] = g.where(g.isin(KNOWN), "unknown")
        out["duration"] = out["end_time"] - out["start_time"]
        out = out[out["duration"] > 0]
        return out[cols + ["duration"]].sort_values(["start_time", "end_time"]).reset_index(drop=True)

    @staticmethod
    def _merge_turns(df: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
        """
        Une, por hablante, intervalos separados por <= max_gap_s en un único turno.
        """
        cols = ["start_time", "end_time", "speaker_label", "gender"]
        if df.empty:
            return pd.DataFrame(columns=cols + ["duration"])
        rows = []
        for spk, g in df.groupby("speaker_label", sort=False):
            gender = g["gender"].mode().iat[0]
            s_arr, e_arr = g["start_time"].to_numpy(), g["end_time"].to_numpy()
            order = np.argsort(s_arr)
            s_arr, e_arr = s_arr[order], e_arr[order]
            cs, ce = s_arr[0], e_arr[0]
            for s, e in zip(s_arr[1:], e_arr[1:]):
                if s - ce <= max_gap_s:
                    ce = max(ce, e)
                else:
                    rows.append((cs, ce, spk, gender))
                    cs, ce = s, e
            rows.append((cs, ce, spk, gender))
        turns = pd.DataFrame(rows, columns=cols)
        turns["duration"] = turns["end_time"] - turns["start_time"]
        return turns.sort_values(["start_time", "end_time"]).reset_index(drop=True)

    @staticmethod
    def _overlap_seconds(df: pd.DataFrame) -> float:
        """
        Segundos con >=2 hablantes distintos activos.
        """
        merged = EquityCalculator._merge_turns(df, 0.0)  # evita auto-solapes del mismo hablante
        if len(merged) < 2:
            return 0.0
        starts, ends = merged["start_time"].to_numpy(), merged["end_time"].to_numpy()
        times = np.concatenate([starts, ends])
        deltas = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
        order = np.lexsort((deltas, times))      # a igual tiempo, primero los fines
        times, counts = times[order], np.cumsum(deltas[order])
        return float(np.diff(times)[counts[:-1] >= 2].sum())

    # Interrupciones
    @staticmethod
    def detect_interruptions(df_turns: pd.DataFrame, overlap_threshold: float = 0.5) -> pd.DataFrame:
        """Detecta interrupciones entre turnos.
        B interrumpe a A si B empieza mientras A aún habla (A empezó antes),
        son hablantes distintos y el solape supera ``overlap_threshold``.
        ``takes_floor`` indica si B sigue hablando cuando A termina.
        """
        cols = ["time_at", "interruptor_speaker", "interruptor_gender",
                "interrupted_speaker", "interrupted_gender", "overlap_s", "takes_floor", "type"]
        if df_turns is None or len(df_turns) < 2:
            return pd.DataFrame(columns=cols)
        turns = df_turns.sort_values(["start_time", "end_time"]).reset_index(drop=True)
        active, events = [], []
        for row in turns.itertuples(index=False):
            active = [a for a in active if a.end_time > row.start_time]
            for a in active:
                if a.speaker_label == row.speaker_label or a.start_time >= row.start_time:
                    continue
                overlap = min(a.end_time, row.end_time) - row.start_time
                if overlap >= overlap_threshold:
                    events.append({
                        "time_at": row.start_time,
                        "interruptor_speaker": row.speaker_label,
                        "interruptor_gender": row.gender,
                        "interrupted_speaker": a.speaker_label,
                        "interrupted_gender": a.gender,
                        "overlap_s": overlap,
                        "takes_floor": row.end_time > a.end_time,
                        "type": f"{row.gender}_over_{a.gender}",
                    })
            active.append(row)
        return pd.DataFrame(events, columns=cols)

    # Cálculo principal
    @classmethod
    def compute(
        cls,
        df_segments: pd.DataFrame,
        df_turns: Optional[pd.DataFrame] = None,
        turn_gap_s: float = 1.0,
        overlap_threshold_s: float = 0.5,
        n_temporal_bins: int = 4,
    ) -> EquityMetrics:
        seg = cls._prepare(df_segments)
        turns_in = cls._prepare(df_turns)
        units = turns_in if not turns_in.empty else seg   # base para tiempo/turnos/interrupciones
        res = EquityMetrics(time_source="diarization_turns" if not turns_in.empty else "asr_segments")
        res.interruptions.threshold_s = overlap_threshold_s
        if units.empty:
            return res

        t0, t1 = units["start_time"].min(), units["end_time"].max()
        res.span_s = float(t1 - t0)
        res.total_speech_s = float(units["duration"].sum())

        # Tiempo de habla
        by_g = units.groupby("gender")["duration"].sum()
        f_s, m_s, u_s = (float(by_g.get(k, 0.0)) for k in ("female", "male", "unknown"))
        st = res.speaking_time
        st.female_s, st.male_s, st.unknown_s = f_s, m_s, u_s
        st.female_pct, st.male_pct = _pct(f_s, f_s + m_s), _pct(m_s, f_s + m_s)
        st.unknown_pct = _pct(u_s, f_s + m_s + u_s)
        st.known_coverage_pct = _pct(f_s + m_s, f_s + m_s + u_s)
        st.overlap_s = cls._overlap_seconds(units)
        merged_speech = units["duration"].sum()
        st.overlap_pct = _pct(st.overlap_s, merged_speech)

        # Palabras
        if not seg.empty:
            n_words = seg["transcript"].str.split().str.len().fillna(0)
            w = n_words.groupby(seg["gender"]).sum()
            wm = res.words
            wm.female, wm.male, wm.unknown = (int(w.get(k, 0)) for k in ("female", "male", "unknown"))
            wm.female_pct, wm.male_pct = _pct(wm.female, wm.female + wm.male), _pct(wm.male, wm.female + wm.male)

        # Hablantes
        spk = units.groupby("speaker_label").agg(
            time=("duration", "sum"), gender=("gender", lambda x: x.mode().iat[0]))
        sm = res.speakers
        sm.n_female, sm.n_male, sm.n_unknown = (int((spk["gender"] == k).sum()) for k in ("female", "male", "unknown"))
        sm.female_speaker_pct = _pct(sm.n_female, sm.n_female + sm.n_male)
        sm.mean_time_female_s = _stat(spk.loc[spk.gender == "female", "time"], np.mean)
        sm.mean_time_male_s = _stat(spk.loc[spk.gender == "male", "time"], np.mean)
        sm.time_per_speaker_ratio_f_m = _ratio(sm.mean_time_female_s, sm.mean_time_male_s)
        sm.participation_gap_pp = (st.female_pct - sm.female_speaker_pct
                                   if not (np.isnan(st.female_pct) or np.isnan(sm.female_speaker_pct)) else NAN)
        sm.speaking_time_gini = _gini(spk["time"].to_numpy())
        top = spk["time"].idxmax()
        sm.top_speaker_gender = str(spk.loc[top, "gender"])
        sm.top_speaker_share_pct = _pct(float(spk.loc[top, "time"]), float(spk["time"].sum()))

        # Turnos
        turns = cls._merge_turns(units, turn_gap_s)
        tm = res.turns
        tm.n_female, tm.n_male, tm.n_unknown = (int((turns["gender"] == k).sum()) for k in ("female", "male", "unknown"))
        tm.female_pct = _pct(tm.n_female, tm.n_female + tm.n_male)
        for g, tag in (("female", "female"), ("male", "male")):
            d = turns.loc[turns.gender == g, "duration"]
            setattr(tm, f"mean_s_{tag}", _stat(d, np.mean))
            setattr(tm, f"median_s_{tag}", _stat(d, np.median))
            setattr(tm, f"max_s_{tag}", _stat(d, np.max))
        tm.mean_ratio_f_m = _ratio(tm.mean_s_female, tm.mean_s_male)
        trans = {f"{a}_to_{b}": 0 for a in KNOWN for b in KNOWN}
        seq = turns[turns.gender.isin(KNOWN)].reset_index(drop=True)
        for i in range(len(seq) - 1):
            if seq.at[i, "speaker_label"] != seq.at[i + 1, "speaker_label"]:
                trans[f"{seq.at[i, 'gender']}_to_{seq.at[i + 1, 'gender']}"] += 1
        tm.transitions = trans

        # Control de la palabra
        fm = res.floor
        if not seq.empty:
            fm.first_speaker_gender = str(seq.iloc[0]["gender"])
            fm.last_speaker_gender = str(seq.loc[seq["end_time"].idxmax(), "gender"])
        fem = turns[turns.gender == "female"]
        if not fem.empty:
            fm.first_female_onset_s = float(fem["start_time"].min() - t0)
            fm.first_female_onset_pct = _pct(fm.first_female_onset_s, res.span_s)

        # Interrupciones
        # 1. Filtramos ruido (segmentos < 0.3s) y usamos las unidades sin fundir
        units_clean = units[units["duration"] >= 0.3].reset_index(drop=True)
        ev_all = cls.detect_interruptions(units_clean, overlap_threshold_s)
        
        # 2. Separamos backchannels (asentimientos) de las interrupciones reales
        if not ev_all.empty:
            ev_all["is_backchannel"] = ~ev_all["takes_floor"]
            ev = ev_all[~ev_all["is_backchannel"]].copy()
        else:
            ev = ev_all.copy()

        im = res.interruptions
        im.total = len(ev)
        
        if im.total:
            counts = ev["type"].value_counts()
            im.male_over_female = int(counts.get("male_over_female", 0))
            im.female_over_male = int(counts.get("female_over_male", 0))
            im.male_over_male = int(counts.get("male_over_male", 0))
            im.female_over_female = int(counts.get("female_over_female", 0))
            
            # Las interrupciones reales ('ev') ya son aquellas donde takes_floor es True, 
            ft = ev[ev["takes_floor"]]["type"].value_counts()
            im.floor_taking_male_over_female = int(ft.get("male_over_female", 0))
            im.floor_taking_female_over_male = int(ft.get("female_over_male", 0))
            
        mf, fmv = im.male_over_female, im.female_over_male
        im.asymmetry_index = (mf - fmv) / (mf + fmv) if (mf + fmv) > 0 else NAN
        hours = res.span_s / 3600.0
        made = {g: int((ev["interruptor_gender"] == g).sum()) for g in KNOWN} if im.total else {g: 0 for g in KNOWN}
        recv = {g: int((ev["interrupted_gender"] == g).sum()) for g in KNOWN} if im.total else {g: 0 for g in KNOWN}
        n_turns = {"female": tm.n_female, "male": tm.n_male}
        for g in KNOWN:
            setattr(im, f"made_per_hour_{g}", made[g] / hours if hours > 0 else NAN)
            setattr(im, f"received_per_hour_{g}", recv[g] / hours if hours > 0 else NAN)
            setattr(im, f"made_per_turn_{g}", made[g] / n_turns[g] if n_turns[g] else NAN)
            setattr(im, f"received_per_turn_{g}", recv[g] / n_turns[g] if n_turns[g] else NAN)

        # Evolución temporal: % de tiempo femenino por tramo 
        edges = np.linspace(t0, t1, n_temporal_bins + 1)
        s_arr, e_arr = units["start_time"].to_numpy(), units["end_time"].to_numpy()
        is_f, is_m = (units["gender"] == "female").to_numpy(), (units["gender"] == "male").to_numpy()
        for b in range(n_temporal_bins):
            ov = np.clip(np.minimum(e_arr, edges[b + 1]) - np.maximum(s_arr, edges[b]), 0, None)
            res.temporal.female_time_pct_by_quarter[f"q{b + 1}"] = _pct(ov[is_f].sum(), ov[is_f | is_m].sum())
        return res