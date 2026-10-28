import os
import gc
import time
import logging
from dataclasses import dataclass
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import src.utils.compatibility 

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from src.utils.exceptions import ModelLoadingError, AudioProcessingError
from src.utils.profiling import ResourceProfiler

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SEGMENT_COLUMNS = ["start_time", "end_time", "speaker_label", "gender", "gender_confidence", "transcript"]
TURN_COLUMNS = ["start_time", "end_time", "speaker_label", "gender", "gender_confidence"]


class ModelHead(nn.Module):
    """Cabezal de clasificación para el modelo base Wav2Vec2."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Modelo preentrenado de HuggingFace adaptado para predicción de edad y género."""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        h = self.wav2vec2(input_values)[0]
        h = torch.mean(h, dim=1)  # Resumen por chunk de audio
        return self.age(h), torch.softmax(self.gender(h), dim=1)


@dataclass
class ProcessingOutput:
    """Resultado de procesar un audio."""
    segments: pd.DataFrame   # segmentos ASR + hablante + género
    turns: pd.DataFrame      # turnos de diarización + género (admite solapes)
    metadata: dict            # tiempos por etapa 


class AudioProcessor:
    """Orquesta el pipeline de procesamiento de voz (ASR, Diarización, Género)."""

    CLASSES = ["female", "male", "child"]

    def __init__(self, gpu_index: int = 0, hf_token: str = None,
                 language: Optional[str] = "ca", monitor_gpu: bool = True):
        """
        Inicializa el entorno y carga los modelos en la GPU asignada.
        """
        self.gpu_index = gpu_index
        self.language = language
        self.monitor_gpu = monitor_gpu

        # 1. Aislar la GPU a nivel de entorno operativo 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # 2. Al haber aislado la GPU, PyTorch siempre la mapeará al índice lógico 0
        self.device_str = "cuda:0"
        self.device = torch.device(self.device_str)

        t0 = time.perf_counter()
        try:
            self._load_models(hf_token)
            self.model_load_seconds = round(time.perf_counter() - t0, 3)
        except Exception as e:
            logger.error(f"Fallo al inicializar modelos en {self.device_str}: {e}")
            raise ModelLoadingError(f"Error cargando modelos en GPU {gpu_index}: {str(e)}") from e

    def _load_models(self, hf_token: str) -> None:
        """
        Carga los modelos en memoria de video (VRAM) optimizados para float16.
        """
        logger.info(f"Cargando modelos en {self.device_str} (float16)...")

        # 1. ASR (faster-whisper)
        self.asr_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

        # 2. Diarización (pyannote.audio 3.1)
        if not hf_token:
            raise ValueError("Se requiere hf_token para pyannote.")
        try:      # pyannote >= 4 
            self.dia_pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        except TypeError:   # pyannote 3.x
            self.dia_pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        self.dia_pipe.to(self.device)

        # 3. Clasificación de género
        name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        self.gender_proc = Wav2Vec2Processor.from_pretrained(name)
        self.gender_model = AgeGenderModel.from_pretrained(name).half().to(self.device).eval()

    @staticmethod
    def _load_audio(file_path: str) -> np.ndarray:
        """
        Lee el audio y lo deja en mono / 16 kHz.
        """
        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly
            g = gcd(SAMPLE_RATE, sr)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype("float32")
        return np.ascontiguousarray(audio)

    def process_file(self, file_path: str, gender_threshold: float = 0.8) -> ProcessingOutput:
            """Ejecuta el pipeline completo para un audio."""
            logger.info(f"Procesando archivo: {file_path}")
            waveform = None
            
            # Medir tiempo de ejecución total del archivo y, por etapa, tiempo y memoria
            t_start = time.perf_counter()
            prof = ResourceProfiler()
            
            try:
                with prof.stage("load_audio"):
                    audio = self._load_audio(file_path)
                duration = len(audio) / SAMPLE_RATE

                # 1. Transcripción 
                with prof.stage("asr"):
                    segments, info = self.asr_model.transcribe(
                        audio,
                        language=self.language,
                        multilingual=True,
                        beam_size=5,

                        vad_filter=False,
                        vad_parameters=dict(
                            threshold=0.2,
                            min_silence_duration_ms=700,
                            speech_pad_ms=500,
                        ),

                        condition_on_previous_text=False,
                        temperature=[0.0, 0.2, 0.4],
                        word_timestamps=True,
                        hallucination_silence_threshold=2.0,
                    )
                    transcripts = [{"start_time": s.start, "end_time": s.end, "transcript": s.text.strip()}
                                for s in segments]

                # 2. Diarización
                with prof.stage("diarization"):
                    waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                    diar = self.dia_pipe({"waveform": waveform, "sample_rate": SAMPLE_RATE})
                    annotation = getattr(diar, "speaker_diarization", diar)
                    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]

                # 3. Asignar hablante y 4. Género por hablante
                with prof.stage("gender"):
                    df_text = self._assign_speakers(transcripts, turns)
                    speaker_gender = self._classify_speakers(audio, turns, gender_threshold)
                    segments_df, turns_df = self._build_outputs(df_text, turns, speaker_gender)
                
                # Calcular tiempo total de procesamiento
                processing_seconds = round(time.perf_counter() - t_start, 3)

                metadata = {
                    "processing_seconds": processing_seconds,  # <--- TIEMPO TOTAL AQUÍ
                    "audio_duration_seconds": round(duration, 3),
                    "model_load_seconds": self.model_load_seconds,
                    "language_requested": self.language,
                    "language_detected": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                    "gpu_index": self.gpu_index,
                    **prof.summary(),  # stage_seconds, ram_peak_mb, gpu_peak_mb y detalle de GPU por etapa
                }
                return ProcessingOutput(segments_df, turns_df, metadata)

            except Exception as e:
                logger.error(f"Fallo durante el procesamiento de {file_path}: {e}")
                raise AudioProcessingError(f"Error procesando el archivo {file_path}: {str(e)}") from e

            finally:
                del waveform
                torch.cuda.empty_cache()
                gc.collect()

    @staticmethod
    def _assign_speakers(transcripts: List[Dict], turns: List[Tuple]) -> pd.DataFrame:
        """
        Asigna a cada segmento de texto el hablante con mayor solape temporal.
        """
        df = pd.DataFrame(transcripts, columns=["start_time", "end_time", "transcript"])
        if df.empty:
            df["speaker_label"] = []
            return df
        t_start = np.array([t[0] for t in turns], dtype=float)
        t_end = np.array([t[1] for t in turns], dtype=float)
        t_spk = [t[2] for t in turns]
        labels = []
        for s, e in zip(df["start_time"], df["end_time"]):
            if len(turns) == 0:
                labels.append("UNKNOWN")
                continue
            ov = np.minimum(t_end, e) - np.maximum(t_start, s)
            acc: Dict[str, float] = {}
            for i in np.nonzero(ov > 0)[0]:
                acc[t_spk[i]] = acc.get(t_spk[i], 0.0) + float(ov[i])
            labels.append(max(acc, key=acc.get) if acc else "UNKNOWN")
        df["speaker_label"] = labels
        return df

    def _classify_speakers(self, audio: np.ndarray, turns: List[Tuple], threshold: float) -> Dict[str, Tuple[str, float]]:
        """
        Género por hablante: {speaker: (género, confianza)}. Máx. 60 s de audio por hablante.
        """
        turns_df = pd.DataFrame(turns, columns=["start", "end", "speaker"])
        turns_df["dur"] = turns_df["end"] - turns_df["start"]
        result: Dict[str, Tuple[str, float]] = {}
        for spk, group in turns_df.groupby("speaker"):
            probs, weights, used = [], [], 0.0
            valid = group[group["dur"] >= 1.0]
            if valid.empty:                       # hablante solo con intervenciones cortas
                valid = group[group["dur"] >= 0.5]
            for _, r in valid.sort_values("dur", ascending=False).iterrows():
                p, w = self._get_gender_probs(audio, r["start"], r["end"])
                if p is not None:
                    probs.append(p)
                    weights.append(w)
                    used += w
                if used >= 60.0:
                    break
            if probs:
                avg_p = np.average(probs, axis=0, weights=weights)
                conf = float(max(avg_p))
                result[spk] = (self.CLASSES[int(np.argmax(avg_p))] if conf >= threshold else "unknown", conf)
            else:
                result[spk] = ("unknown", 0.0)
        return result

    @staticmethod
    def _build_outputs(df_text: pd.DataFrame, turns: List[Tuple],
                       speaker_gender: Dict[str, Tuple[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        gmap = {k: v[0] for k, v in speaker_gender.items()}
        cmap = {k: v[1] for k, v in speaker_gender.items()}

        df_text = df_text.copy()
        df_text["gender"] = df_text["speaker_label"].map(gmap).fillna("unknown")
        df_text["gender_confidence"] = df_text["speaker_label"].map(cmap).fillna(0.0)
        segments = df_text.reindex(columns=SEGMENT_COLUMNS)

        turns_df = pd.DataFrame(turns, columns=["start_time", "end_time", "speaker_label"])
        turns_df["gender"] = turns_df["speaker_label"].map(gmap).fillna("unknown")
        turns_df["gender_confidence"] = turns_df["speaker_label"].map(cmap).fillna(0.0)
        turns_df = turns_df.sort_values(["start_time", "end_time"]).reset_index(drop=True)
        return segments, turns_df.reindex(columns=TURN_COLUMNS)

    @torch.no_grad()
    def _get_gender_probs(self, audio: np.ndarray, start: float, end: float, max_chunk: float = 15.0):
        """
        Infiere probabilidades de género dividiendo el audio en chunks para proteger la VRAM.
        """
        results, weights = [], []
        t = start
        while t < end:
            t2 = min(t + max_chunk, end)
            if t2 - t >= 0.5:
                chunk = audio[int(t * SAMPLE_RATE): int(t2 * SAMPLE_RATE)]
                x = self.gender_proc(chunk, sampling_rate=SAMPLE_RATE)["input_values"][0]
                x = torch.tensor(x, dtype=torch.float16, device=self.device)[None]
                _, p = self.gender_model(x)
                results.append(p[0].float().cpu().numpy())
                weights.append(t2 - t)
            t = t2
        if not results:
            return None, 0.0
        return np.average(results, axis=0, weights=weights), sum(weights)