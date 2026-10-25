import os
import gc
import logging
from typing import List, Dict, Tuple
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

# Importar excepciones personalizadas
from src.utils.exceptions import ModelLoadingError, AudioProcessingError

logger = logging.getLogger(__name__)

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
        h = torch.mean(h, dim=1) # Resumen por chunk de audio
        return self.age(h), torch.softmax(self.gender(h), dim=1)


class AudioProcessor:
    """
    Orquesta el pipeline de procesamiento de voz (ASR, Diarización, Género).
    """
    
    CLASSES = ["female", "male", "child"]
    
    def __init__(self, gpu_index: int = 0, hf_token: str = None):
        """
        Inicializa el entorno y carga los modelos en la GPU asignada.
        """
        self.device_str = f"cuda:{gpu_index}"
        self.device = torch.device(self.device_str)
        
        # Aislar GPU a nivel de entorno operativo
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        
        try:
            self._load_models(hf_token)
        except Exception as e:
            logger.error(f"Fallo al inicializar modelos en {self.device_str}: {e}")
            raise ModelLoadingError(f"Error cargando modelos en GPU {gpu_index}: {str(e)}") from e

    def _load_models(self, hf_token: str) -> None:
        """Carga los modelos en memoria de video (VRAM) optimizados para float16."""
        logger.info(f"Cargando modelos en {self.device_str} (float16)...")
        
        # 1. ASR (faster-whisper)
        self.asr_model = WhisperModel(
            "large-v3-turbo", 
            device="cuda", 
            compute_type="float16"
        )
        
        # 2. Diarización (pyannote.audio 3.1)
        if not hf_token:
            raise ValueError("Se requiere hf_token para pyannote.")
        self.dia_pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        self.dia_pipe.to(self.device)
        
        # 3. Clasificación de género
        name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        self.gender_proc = Wav2Vec2Processor.from_pretrained(name)
        self.gender_model = AgeGenderModel.from_pretrained(name).half().to(self.device).eval()

    def process_file(self, file_path: str, gender_threshold: float = 0.8) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo para un archivo de audio.
        """
        logger.info(f"Procesando archivo: {file_path}")
        
        try:
            # Leer audio (mono; 16kHz)
            audio, sr = sf.read(file_path, dtype="float32")
            if sr != 16000:
                raise ValueError(f"Frecuencia de muestreo incorrecta: {sr}Hz. Se esperaba 16000Hz.")
            if audio.ndim != 1:
                raise ValueError("El audio debe tener un solo canal (mono).")

            # 1. Transcripción
            segments, _ = self.asr_model.transcribe(
                audio, language="ca", vad_filter=True, condition_on_previous_text=False
            )
            transcripts = [{"start_time": s.start, "end_time": s.end, "transcript": s.text.strip()} for s in segments]
            
            # 2. Diarización
            waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            diar = self.dia_pipe({"waveform": waveform, "sample_rate": 16000})
            turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
            
            # 3. Cruce de datos y Clasificación de Género
            df_result = self._merge_and_classify(audio, transcripts, turns, gender_threshold)
            
            return df_result
            
        except Exception as e:
            logger.error(f"Fallo durante el procesamiento de {file_path}: {e}")
            raise AudioProcessingError(f"Error procesando el archivo {file_path}: {str(e)}") from e
            
        finally:
            # Limpieza exhaustiva de VRAM en cada iteración
            if 'waveform' in locals():
                del waveform
            torch.cuda.empty_cache()
            gc.collect()

    def _merge_and_classify(
        self, 
        audio: np.ndarray, 
        transcripts: List[Dict], 
        turns: List[Tuple],
        threshold: float
    ) -> pd.DataFrame:
        """Asigna hablantes al texto y calcula el género por cada turno de voz."""
        
        # A. Asignar Hablante al Texto
        df_text = pd.DataFrame(transcripts)
        if df_text.empty:
            return pd.DataFrame(columns=["start_time", "end_time", "speaker_label", "transcript", "gender"])
            
        def get_best_speaker(seg):
            overlap = {}
            for start, end, spk in turns:
                o = min(end, seg["end_time"]) - max(start, seg["start_time"])
                if o > 0:
                    overlap[spk] = overlap.get(spk, 0) + o
            return max(overlap, key=overlap.get) if overlap else "UNKNOWN"
            
        df_text["speaker_label"] = df_text.apply(get_best_speaker, axis=1)

        # B. Extraer género por cada hablante detectado
        turns_df = pd.DataFrame(turns, columns=["start", "end", "speaker"])
        turns_df["dur"] = turns_df["end"] - turns_df["start"]
        
        speaker_genders = {}
        for spk, group in turns_df.groupby("speaker"):
            probs, weights, used = [], [], 0.0
            
            # Filtrar turnos cortos y ordenar por duración
            valid_turns = group[group["dur"] >= 1.0].sort_values("dur", ascending=False)
            
            for _, r in valid_turns.iterrows():
                p, w = self._get_gender_probs(audio, r["start"], r["end"])
                if p is not None:
                    probs.append(p)
                    weights.append(w)
                    used += w
                if used >= 60.0:  # Analizar un máximo de 60 segundos por hablante para eficiencia
                    break
                    
            if probs:
                avg_p = np.average(probs, axis=0, weights=weights)
                # Aplicar umbral para decisión final
                if max(avg_p) >= threshold:
                    speaker_genders[spk] = self.CLASSES[int(np.argmax(avg_p))]
                else:
                    speaker_genders[spk] = "unknown"
            else:
                speaker_genders[spk] = "unknown"

        # C. Unir género a la tabla final
        df_text["gender"] = df_text["speaker_label"].map(speaker_genders).fillna("unknown")
        
        # Retornar el dataframe con el orden deseado
        return df_text[["start_time", "end_time", "speaker_label", "gender", "transcript"]]

    @torch.no_grad()
    def _get_gender_probs(self, audio: np.ndarray, start: float, end: float, max_chunk: float = 15.0):
        """
        Infiere probabilidades de género dividiendo el audio en chunks para proteger la VRAM.
        """
        results, weights = [], []
        t = start
        
        while t < end:
            t2 = min(t + max_chunk, end)
            if t2 - t >= 0.5:  # Omitir fragmentos minúsculos
                chunk = audio[int(t * 16000) : int(t2 * 16000)]
                x = self.gender_proc(chunk, sampling_rate=16000)["input_values"][0]
                # Convertir a tensor float16
                x = torch.tensor(x, dtype=torch.float16, device=self.device)[None]
                
                _, p = self.gender_model(x)
                results.append(p[0].float().cpu().numpy())
                weights.append(t2 - t)
            t = t2
            
        if not results:
            return None, 0.0
            
        return np.average(results, axis=0, weights=weights), sum(weights)