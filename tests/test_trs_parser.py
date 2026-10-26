import pytest
import pandas as pd
from src.utils.trs_parser import parse_transcriber_file

def test_parse_transcriber_file(tmp_path):
    # Simular un archivo TRS mínimo con casos límite (textos con saltos de línea, mayúsculas)
    xml_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <Trans>
        <Speakers>
            <Speaker id="spk1" name="Locutor" type="male"/>
            <Speaker id="spk2" name="Invitada" type="female"/>
        </Speakers>
        <Episode>
            <Section type="report" startTime="0" endTime="10">
                <Turn speaker="spk1" startTime="0.0" endTime="5.0">
                    Hola a todos.
                </Turn>
                <Turn speaker="spk2" startTime="5.0" endTime="10.0">
                    <Event desc="b" type="noise" extent="instantaneous"/>
                    Muchas gracias.
                </Turn>
            </Section>
        </Episode>
    </Trans>"""
    
    # Crear archivo temporal
    test_file = tmp_path / "test_audio.trs"
    test_file.write_text(xml_content, encoding="ISO-8859-1")
    
    # Ejecutar función
    df = parse_transcriber_file(str(test_file))
    
    # Validaciones
    assert len(df) == 2, "Debe extraer exactamente 2 turnos de habla"
    assert list(df.columns) == ["start_time", "end_time", "speaker_label", "gender", "transcript"]
    
    # Validar mapeo de géneros y limpieza de texto
    assert df.iloc[0]["speaker_label"] == "spk1"
    assert df.iloc[0]["gender"] == "male"
    assert df.iloc[0]["transcript"] == "Hola a todos."
    
    assert df.iloc[1]["speaker_label"] == "spk2"
    assert df.iloc[1]["gender"] == "female"
    assert df.iloc[1]["transcript"] == "Muchas gracias.", "Debe ignorar la etiqueta de ruido <Event>"