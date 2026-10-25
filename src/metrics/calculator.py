import pandas as pd
from typing import Dict, Any

class MetricsCalculator:
    """
    Calcula métricas de equidad e interacciones a partir del DataFrame de transcripciones.
    """

    @staticmethod
    def calculate_speaking_time(df: pd.DataFrame) -> Dict[str, float]:
        """Calcula el porcentaje de tiempo de habla por género."""
        if df.empty:
            return {"female_pct": 0.0, "male_pct": 0.0}
            
        df_valid = df[df["gender"].isin(["female", "male"])].copy()
        df_valid["duration"] = df_valid["end_time"] - df_valid["start_time"]
        
        total_time = df_valid["duration"].sum()
        if total_time == 0:
            return {"female_pct": 0.0, "male_pct": 0.0}
            
        gender_times = df_valid.groupby("gender")["duration"].sum()
        
        return {
            "female_pct": (gender_times.get("female", 0) / total_time) * 100,
            "male_pct": (gender_times.get("male", 0) / total_time) * 100
        }

    @staticmethod
    def detect_interruptions(df: pd.DataFrame, overlap_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detecta interrupciones utilizando lógica de ventanas cruzadas.
        Se considera interrupción si el hablante B comienza a hablar mientras A sigue hablando,
        y el solapamiento supera un umbral.
        """
        # Ordenar cronológicamente por inicio
        df_sorted = df.sort_values(by="start_time").reset_index(drop=True)
        interruptions = []

        for i in range(1, len(df_sorted)):
            current = df_sorted.iloc[i]
            # Buscar segmentos anteriores que aún no han terminado
            previous_segments = df_sorted.iloc[:i]
            active_previous = previous_segments[previous_segments["end_time"] > current["start_time"]]
            
            for _, prev in active_previous.iterrows():
                # Ignorar si es el mismo hablante
                if prev["speaker_label"] == current["speaker_label"]:
                    continue
                    
                overlap = min(prev["end_time"], current["end_time"]) - current["start_time"]
                
                if overlap >= overlap_threshold:
                    interruption_type = f"{prev['gender']}_to_{current['gender']}"
                    interruptions.append({
                        "interrupted_speaker": prev["speaker_label"],
                        "interrupted_gender": prev["gender"],
                        "interruptor_speaker": current["speaker_label"],
                        "interruptor_gender": current["gender"],
                        "overlap_duration": overlap,
                        "type": interruption_type,
                        "time_at": current["start_time"]
                    })
                    
        return pd.DataFrame(interruptions)
        
    @classmethod
    def get_male_to_female_interruptions_count(cls, df: pd.DataFrame) -> int:
        """Wrapper para la métrica de interrupciones."""
        df_interruptions = cls.detect_interruptions(df)
        if df_interruptions.empty:
            return 0
        return len(df_interruptions[df_interruptions["type"] == "female_to_male"])