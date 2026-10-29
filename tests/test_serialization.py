# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import numpy as np
from dataclasses import dataclass

from src.utils.serialization import to_serializable, flatten_dict

@dataclass
class DummyConfig:
    """
    Dataclass auxiliar para validar la serialización.
    """
    threshold: float
    is_active: bool
    labels: tuple


class TestSerializationUtils:
    """
    Suite de pruebas para utilidades de transformación y limpieza de datos.
    """

    # Tests de to_serializable
    # ----------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_val, expected_val",
        [
            (42, 42),
            ("texto", "texto"),
            (True, True),
            (None, None),
            # Las tuplas se convierten en listas por list comprehension
            ((1, 2, 3), [1, 2, 3]),
            (["a", "b"], ["a", "b"]),
        ]
    )
    def test_to_serializable_python_builtins(self, input_val, expected_val):
        """
        Valida que los tipos estándar nativos de Python pasen o se adapten correctamente.
        """
        assert to_serializable(input_val) == expected_val

    @pytest.mark.parametrize(
        "input_val, expected_val",
        [
            (np.int32(100), 100),
            (np.float64(3.1415), 3.1415),
            (np.bool_(False), False),
            (np.array([1])[0], 1)
        ]
    )
    def test_to_serializable_numpy_types(self, input_val, expected_val):
        """
        Garantiza que los tipos primitivos de NumPy se desempaqueten mediante .item().
        """
        result = to_serializable(input_val)
        assert result == expected_val
        assert not isinstance(result, np.generic)
        assert isinstance(result, (int, float, bool))

    @pytest.mark.parametrize(
        "input_val",
        [math.nan, float("nan"), math.inf, float("inf"), -math.inf]
    )
    def test_to_serializable_floats_nan_inf(self, input_val):
        """
        Verifica que los flotantes no admitidos por JSON (NaN, inf) se transformen en None.
        """
        assert to_serializable(input_val) is None

    def test_to_serializable_dataclass(self):
        """
        Valida la conversión automática de dataclasses usando asdict().
        """
        obj = DummyConfig(threshold=math.inf, is_active=True, labels=("A", "B"))
        result = to_serializable(obj)
        
        assert isinstance(result, dict)
        assert result["threshold"] is None       # inf muta a None
        assert result["is_active"] is True
        assert result["labels"] == ["A", "B"]    # la tupla muta a lista

    def test_to_serializable_nested_structures(self):
        """
        Prueba recursión profunda en diccionarios asegurando que las keys sean siempre str.
        """
        input_data = {
            1: np.float32(1.5),                 # La key int muta a str "1"
            "b": [math.nan, {"c": (1, 2)}]      # Lista con NaN y diccionario anidado
        }
        expected = {
            "1": 1.5,
            "b": [None, {"c": [1, 2]}]
        }
        assert to_serializable(input_data) == expected

    # Test de flatten_dict
    # ----------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_dict, prefix, sep, expected",
        [
            # Caso 1: Diccionario plano (no surte efecto)
            ({"a": 1, "b": 2}, "", ".", {"a": 1, "b": 2}),
            
            # Caso 2: Anidado simple
            ({"a": {"b": 1}}, "", ".", {"a.b": 1}),
            
            # Caso 3: Múltiples ramas y niveles de profundidad
            (
                {"a": {"b": {"c": 1}, "d": 2}, "e": 3},
                "", ".", 
                {"a.b.c": 1, "a.d": 2, "e": 3}
            ),
            
            # Caso 4: Uso de separador personalizado
            ({"config": {"theme": "dark"}}, "", "_", {"config_theme": "dark"}),
            
            # Caso 5: Listas anidadas (no se expanden por índice, se mantienen)
            ({"a": {"b": [1, 2, 3]}}, "", ".", {"a.b": [1, 2, 3]}),
            
            # Caso 6: Fuerza de un prefijo inicial exterior
            ({"val": 10}, "root", ".", {"root.val": 10}),
            
            # Caso 7: Diccionario vacío y diccionarios con diccionarios vacíos
            ({}, "", ".", {}),
            ({"a": {}}, "", ".", {})
        ]
    )
    def test_flatten_dict_scenarios(self, input_dict, prefix, sep, expected):
        """
        Prueba distintos niveles de anidación y configuración del aplanador de diccionarios.
        """
        assert flatten_dict(input_dict, prefix=prefix, sep=sep) == expected