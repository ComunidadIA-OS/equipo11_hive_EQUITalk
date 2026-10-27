import pytest
import pandas as pd
# Asegúrate de ajustar esta importación a la ruta real de tu proyecto
from src.utils.trs_parser import parse_transcriber_file

def test_parse_transcriber_file(tmp_path):
    # Simular un archivo TRS con casos límite y la nueva estructura compleja
    xml_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <Trans>
        <Speakers>
            <Speaker id="spk1" name="Locutor" type="male"/>
            <Speaker id="spk2" name="Invitada" type="female"/>
            <Speaker id="spk3" name="Extraño" type="alien"/> <!-- Género inválido para forzar 'unknown' -->
        </Speakers>
        <Episode>
            <Section type="report" startTime="0" endTime="30">
                
                <!-- Caso 1: Turno simple (sin hijos estructurales) -->
                <Turn speaker="spk1" startTime="0.0" endTime="5.0">
                    Hola a todos.
                </Turn>
                
                <!-- Caso 2: Turno con Sync (divide en varios segmentos) y ruido intercalado -->
                <Turn speaker="spk2" startTime="5.0" endTime="10.0">
                    <Sync time="5.0"/>
                    <Event desc="b" type="noise" extent="instantaneous"/>
                    Muchas gracias.
                    <Sync time="7.5"/>
                    Continuamos.
                </Turn>
                
                <!-- Caso 3: Turno con solape (múltiples speakers) y etiquetas Who -->
                <Turn speaker="spk1 spk2" startTime="10.0" endTime="15.0">
                    <Sync time="10.0"/>
                    <Who nb="1"/>Buenos días.
                    <Sync time="12.5"/>
                    <Who nb="2"/>Hola.
                </Turn>

                <!-- Caso 4: Turno sin hablante definido (UNKNOWN) -->
                <Turn startTime="15.0" endTime="20.0">
                    <Sync time="15.0"/>Texto sin hablante.
                </Turn>

                <!-- Caso 5: Turno de hablante con género no normalizado -->
                <Turn speaker="spk3" startTime="20.0" endTime="25.0">
                    Probando género.
                </Turn>
                
                <!-- Caso 6: Turno vacío (solo ruido, debe ignorarse) -->
                <Turn speaker="spk1" startTime="25.0" endTime="30.0">
                    <Event desc="noise" type="noise" extent="instantaneous"/>
                </Turn>
                
            </Section>
        </Episode>
    </Trans>"""
    
    # Crear archivo temporal
    test_file = tmp_path / "test_audio.trs"
    test_file.write_text(xml_content, encoding="ISO-8859-1")
    
    # Ejecutar función
    df = parse_transcriber_file(str(test_file))
    
    # Validaciones de estructura general
    assert len(df) == 7, "Debe extraer 7 segmentos de texto válidos (el turno con solape y los sync generan subdivisiones, el vacío se ignora)"
    assert list(df.columns) == ["start_time", "end_time", "speaker_label", "gender", "transcript"]
    
    # Validar Caso 1: Turno simple
    assert df.iloc[0]["speaker_label"] == "spk1"
    assert df.iloc[0]["gender"] == "male"
    assert df.iloc[0]["start_time"] == 0.0
    assert df.iloc[0]["transcript"] == "Hola a todos."
    
    # Validar Caso 2: División temporal por <Sync> y filtrado de <Event>
    assert df.iloc[1]["speaker_label"] == "spk2"
    assert df.iloc[1]["start_time"] == 5.0
    assert df.iloc[1]["end_time"] == 7.5
    assert df.iloc[1]["transcript"] == "Muchas gracias."
    
    assert df.iloc[2]["speaker_label"] == "spk2"
    assert df.iloc[2]["start_time"] == 7.5
    assert df.iloc[2]["end_time"] == 10.0
    assert df.iloc[2]["transcript"] == "Continuamos."
    
    # Validar Caso 3: Solapes con <Who>
    assert df.iloc[3]["speaker_label"] == "spk1"
    assert df.iloc[3]["gender"] == "male"
    assert df.iloc[3]["start_time"] == 10.0
    assert df.iloc[3]["end_time"] == 12.5
    assert df.iloc[3]["transcript"] == "Buenos días."
    
    assert df.iloc[4]["speaker_label"] == "spk2"
    assert df.iloc[4]["gender"] == "female"
    assert df.iloc[4]["start_time"] == 12.5
    assert df.iloc[4]["end_time"] == 15.0
    assert df.iloc[4]["transcript"] == "Hola."
    
    # Validar Caso 4: Manejo de UNKNOWN
    assert df.iloc[5]["speaker_label"] == "UNKNOWN"
    assert df.iloc[5]["gender"] == "unknown"
    assert df.iloc[5]["start_time"] == 15.0
    assert df.iloc[5]["transcript"] == "Texto sin hablante."
    
    # Validar Caso 5: Normalización de géneros extraños
    assert df.iloc[6]["speaker_label"] == "spk3"
    assert df.iloc[6]["gender"] == "unknown"  # "alien" debió normalizarse a "unknown"
    assert df.iloc[6]["transcript"] == "Probando género."