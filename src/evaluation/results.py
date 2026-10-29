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
            
            # Nuevas métricas de análisis de turnos y estructura
            "Opened_By": pe.turns.opened_by if hasattr(pe, "turns") else None,
            "Closed_By": pe.turns.closed_by if hasattr(pe, "turns") else None,
            "Longest_Monologue_Speaker": pe.turns.longest_monologue_speaker if hasattr(pe, "turns") else None,
            "Longest_Monologue_Duration_s": pe.turns.longest_monologue_duration_s if hasattr(pe, "turns") else None,
            
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