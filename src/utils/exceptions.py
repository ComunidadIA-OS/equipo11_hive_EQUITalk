class AudioPipelineError(Exception):
    """Excepción base para errores en el pipeline de procesamiento de audio."""
    pass

class ModelLoadingError(AudioPipelineError):
    """Se lanza cuando falla la carga de modelos de Machine Learning (Whisper, Pyannote, Wav2Vec2)."""
    pass

class AudioProcessingError(AudioPipelineError):
    """Se lanza cuando ocurre un error durante la inferencia o manipulación del archivo de audio."""
    pass

class ConfigurationError(AudioPipelineError):
    """Se lanza cuando hay parámetros inválidos o faltantes en el archivo de configuración."""
    pass