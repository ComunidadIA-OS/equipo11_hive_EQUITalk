import torch
import torchaudio
import huggingface_hub
from dataclasses import dataclass

def apply_compatibility_patches():
    """
    Aplica parches en tiempo de ejecución para resolver incompatibilidades
    entre PyTorch 2.6+, torchaudio y pyannote.audio 3.1.
    """
    # 1. Parche para torchaudio.AudioMetaData
    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData

    # 2. Parches para los backends de torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None

    # 3. Parche de seguridad para torch.load en PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except AttributeError:
        pass

    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    
    torch.load = _patched_torch_load

    # 4. Parche para huggingface_hub (use_auth_token deprecado)
    _orig_hf_hub_download = huggingface_hub.hf_hub_download
    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)
        
    huggingface_hub.hf_hub_download = _patched_hf_hub_download

# Ejecutar el parche automáticamente al importar este módulo
apply_compatibility_patches()