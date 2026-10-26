import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
import jiwer
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

class PerformanceCalculator:
    
    @staticmethod
    def calculate_asr_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
        """Calcula Word Error Rate (WER) y Character Error Rate (CER)."""
        ref_text = " ".join(df_true['transcript'].fillna("").astype(str).tolist())
        hyp_text = " ".join(df_pred['transcript'].fillna("").astype(str).tolist())
        
        # Normalización básica 
        ref_text = " ".join(ref_text.lower().split())
        hyp_text = " ".join(hyp_text.lower().split())
        
        # Si ambos textos están vacíos, jiwer puede fallar o retornar 1.0/0.0
        if not ref_text and not hyp_text:
            return {"WER": 0.0, "CER": 0.0}
        elif not ref_text:
            return {"WER": 1.0, "CER": 1.0}
            
        return {
            "WER": jiwer.wer(ref_text, hyp_text),
            "CER": jiwer.cer(ref_text, hyp_text)
        }

    @staticmethod
    def calculate_diarization_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
        """Calcula el Diarization Error Rate (DER)."""
        ref_annotation = Annotation()
        hyp_annotation = Annotation()
        
        for _, row in df_true.iterrows():
            ref_annotation[Segment(row['start_time'], row['end_time'])] = str(row['speaker_label'])
            
        for _, row in df_pred.iterrows():
            hyp_annotation[Segment(row['start_time'], row['end_time'])] = str(row['speaker_label'])
            
        if len(ref_annotation) == 0:
            return {"DER": 1.0 if len(hyp_annotation) > 0 else 0.0}
            
        uem = ref_annotation.get_timeline() | hyp_annotation.get_timeline()
        
        der_metric = DiarizationErrorRate()
        return {"DER": der_metric(ref_annotation, hyp_annotation, uem=uem)}

    @staticmethod
    def calculate_gender_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
        """
        Calcula métricas de clasificación de género emparejando los segmentos
        predichos con los reales mediante su punto central (midpoint).
        """
        y_true, y_pred = [], []
        
        for _, p_row in df_pred.iterrows():
            midpoint = p_row['start_time'] + (p_row['end_time'] - p_row['start_time']) / 2
            
            matching_true = df_true[
                (df_true['start_time'] <= midpoint) & 
                (df_true['end_time'] >= midpoint)
            ]
            
            if not matching_true.empty:
                # Normalizar a minúsculas
                true_gender = str(matching_true.iloc[0]['gender']).strip().lower()
                pred_gender = str(p_row['gender']).strip().lower()
                
                y_true.append(true_gender)
                y_pred.append(pred_gender)

        # Prevención de error si no hay coincidencias
        labels = ['male', 'female']
        if not y_true:
            return {
                "accuracy": 0.0,
                "sensitivity_macro": 0.0,
                "precision_macro": 0.0
            }

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "sensitivity_macro": recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        }