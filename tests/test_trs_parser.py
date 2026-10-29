# SPDX-License-Identifier: Apache-2.0

from src.utils.trs_parser import parse_transcriber_file

def test_parse_transcriber_file(tmp_path):
    xml_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <Trans>
        <Speakers>
            <Speaker id="spk1" name="Locutor" type="male"/>
            <Speaker id="spk2" name="Invitada" type="female"/>
            <Speaker id="spk3" name="Extraño" type="alien"/>
        </Speakers>
        <Episode>
            <Section type="report" startTime="0" endTime="30">
                
                <!-- Caso 1: Turno simple -->
                <Turn speaker="spk1" startTime="0.0" endTime="5.0">
                    Hola a todos.
                </Turn>
                
                <!-- Caso 2: Turno con Sync y ruido intercalado -->
                <Turn speaker="spk2" startTime="5.0" endTime="10.0">
                    <Sync time="5.0"/>
                    <Event desc="b" type="noise" extent="instantaneous"/>
                    Muchas gracias.
                    <Sync time="7.5"/>
                    Continuamos.
                </Turn>
                
                <!-- Caso 3: Solape simultáneo real en el mismo intervalo de Sync -->
                <Turn speaker="spk1 spk2" startTime="10.0" endTime="15.0">
                    <Sync time="10.0"/>
                    <Who nb="1"/>Buenos días.
                    <Who nb="2"/>Hola.
                    <Sync time="12.5"/>
                    <Who nb="1"/>Adiós.
                </Turn>

                <!-- Caso 4: Turno sin hablante definido (UNKNOWN) -->
                <Turn startTime="15.0" endTime="20.0">
                    <Sync time="15.0"/>Texto sin hablante.
                </Turn>

                <!-- Caso 5: Turno de hablante con género no normalizado -->
                <Turn speaker="spk3" startTime="20.0" endTime="25.0">
                    Probando género.
                </Turn>
                
                <!-- Caso 6: Turno vacío -->
                <Turn speaker="spk1" startTime="25.0" endTime="30.0">
                    <Event desc="noise" type="noise" extent="instantaneous"/>
                </Turn>
                
            </Section>
        </Episode>
    </Trans>"""
    
    test_file = tmp_path / "test_audio.trs"
    test_file.write_text(xml_content, encoding="ISO-8859-1")
    
    df = parse_transcriber_file(str(test_file))
    
    # Se esperan 8 segmentos (el solape emite 2 segmentos simultáneos en la ventana 10.0 - 12.5)
    assert len(df) == 8
    assert list(df.columns) == ["start_time", "end_time", "speaker_label", "gender", "transcript"]
    
    # Validar Caso 1
    assert df.iloc[0]["speaker_label"] == "spk1"
    assert df.iloc[0]["transcript"] == "Hola a todos."
    
    # Validar Caso 2
    assert df.iloc[1]["start_time"] == 5.0
    assert df.iloc[1]["transcript"] == "Muchas gracias."
    assert df.iloc[2]["start_time"] == 7.5
    assert df.iloc[2]["transcript"] == "Continuamos."
    
    # Validar Caso 3: Emisión paralela (habla solapada en los mismos tiempos)
    overlap_spk1 = df[(df["start_time"] == 10.0) & (df["speaker_label"] == "spk1")].iloc[0]
    overlap_spk2 = df[(df["start_time"] == 10.0) & (df["speaker_label"] == "spk2")].iloc[0]
    
    assert overlap_spk1["transcript"] == "Buenos días."
    assert overlap_spk1["end_time"] == 12.5
    assert overlap_spk2["transcript"] == "Hola."
    assert overlap_spk2["end_time"] == 12.5
    
    # Continuación post-solape
    after_overlap = df[df["start_time"] == 12.5].iloc[0]
    assert after_overlap["speaker_label"] == "spk1"
    assert after_overlap["transcript"] == "Adiós."
    
    # Validar UNKNOWN y Género
    assert df.iloc[6]["speaker_label"] == "UNKNOWN"
    assert df.iloc[6]["gender"] == "unknown"
    assert df.iloc[7]["speaker_label"] == "spk3"
    assert df.iloc[7]["gender"] == "unknown"