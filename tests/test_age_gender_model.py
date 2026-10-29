# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import Wav2Vec2Config

from src.pipeline.age_gender_model import AgeGenderWav2Vec2

# Claves de los cabezales en el checkpoint de audeering; si cambian, from_pretrained
# no cargaría esos pesos y los dejaría con valores aleatorios.
EXPECTED_HEAD_KEYS = {
    f"{head}.{layer}.{param}"
    for head in ("age", "gender")
    for layer in ("dense", "out_proj")
    for param in ("weight", "bias")
}


def tiny_model() -> AgeGenderWav2Vec2:
    """Modelo diminuto con la misma estructura que el real, para probar sin descargar nada."""
    config = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    return AgeGenderWav2Vec2(config).eval()


def test_nombres_de_capas_coinciden_con_el_checkpoint():
    keys = set(tiny_model().state_dict())
    assert EXPECTED_HEAD_KEYS <= keys
    assert any(k.startswith("wav2vec2.") for k in keys)


def test_salidas_edad_y_probabilidades_de_genero():
    with torch.no_grad():
        age, gender = tiny_model()(torch.randn(2, 16000))

    assert age.shape == (2, 1)
    assert gender.shape == (2, 3)
    assert torch.allclose(gender.sum(dim=-1), torch.ones(2))
