import xml.etree.ElementTree as ET
import pandas as pd

def parse_transcriber_file(filepath: str) -> pd.DataFrame:
    """Extrae los segmentos de voz y el mapeo de hablantes del XML (.trs)."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # 1. Extraer mapeo de hablantes y su género
    speakers = {}
    for spk in root.findall('.//Speaker'):
        spk_id = spk.get('id')
        spk_gender = spk.get('type', 'unknown').lower()
        speakers[spk_id] = spk_gender

    # 2. Extraer turnos de habla
    segments = []
    for turn in root.findall('.//Turn'):
        speaker_id = turn.get('speaker', 'UNKNOWN')
        start_time = float(turn.get('startTime'))
        end_time = float(turn.get('endTime'))
        gender = speakers.get(speaker_id, 'unknown')
        
        # Extraer texto ignorando etiquetas de ruido (<Event>, <Sync>, etc.)
        text_parts = []
        for text in turn.itertext():
            clean_text = text.strip()
            if clean_text:
                text_parts.append(clean_text)
                
        transcript = " ".join(text_parts)
        
        # Filtrar segmentos vacíos o sin hablante identificado
        if speaker_id != 'UNKNOWN' and transcript:
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "speaker_label": speaker_id,
                "gender": gender,
                "transcript": transcript
            })
            
    return pd.DataFrame(segments)