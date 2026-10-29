# SPDX-License-Identifier: Apache-2.0
"""Arquitectura para cargar los pesos de audeering/wav2vec2-large-robust-24-ft-age-gender.

El checkpoint guarda un codificador Wav2Vec2 (``wav2vec2.*``) y dos cabezales de
clasificación (``age.*`` y ``gender.*``), cada uno con las capas ``dense`` y ``out_proj``.
Los nombres de los atributos tienen que coincidir con los del checkpoint para que
``from_pretrained`` cargue los pesos.

El cabezal sigue el diseño de ``RobertaClassificationHead`` de Hugging Face Transformers
(Apache-2.0): dropout -> lineal -> tanh -> dropout -> lineal. A diferencia de RoBERTa,
aquí la entrada es la media temporal de los estados ocultos, no el primer token.

Los pesos del modelo tienen licencia CC BY-NC-SA 4.0 y no forman parte de este
repositorio: se descargan de Hugging Face al ejecutar el sistema (ver README).
"""
import torch
from torch import nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class ClassificationHead(nn.Module):
    """Capa oculta con tanh seguida de la capa de salida."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.dense(self.dropout(features)))
        return self.out_proj(self.dropout(hidden))


class AgeGenderWav2Vec2(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 con un cabezal de edad (1 valor) y otro de género (female, male, child)."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ClassificationHead(config.hidden_size, 1, config.final_dropout)
        self.gender = ClassificationHead(config.hidden_size, 3, config.final_dropout)
        self.post_init()

    def forward(self, input_values: torch.Tensor):
        """Devuelve (edad, probabilidades de género) para cada fragmento del lote."""
        pooled = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        return self.age(pooled), self.gender(pooled).softmax(dim=-1)
