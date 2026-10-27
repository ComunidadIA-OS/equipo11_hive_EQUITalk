import xml.etree.ElementTree as ET
import pandas as pd

def parse_transcriber_file(filepath: str) -> pd.DataFrame:
    """
    Extrae los segmentos de voz del XML (.trs) soportando solapes (<Who>), 
    sincronismos (<Sync>), turnos UNKNOWN y normalización de género.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # 1. Extraer mapeo de hablantes y normalizar su género
    speakers = {}
    for spk in root.findall('.//Speaker'):
        spk_id = spk.get('id')
        spk_gender = spk.get('type', 'unknown').lower()
        if spk_gender not in ['female', 'male', 'child']:
            spk_gender = 'unknown'
        speakers[spk_id] = spk_gender

    segments = []

    # 2. Recorrer cada turno de habla
    for turn in root.findall('.//Turn'):
        turn_speakers_str = turn.get('speaker', 'UNKNOWN')
        turn_start = float(turn.get('startTime'))
        turn_end = float(turn.get('endTime'))
        
        # Identificar hablantes en el turno 
        turn_speakers = turn_speakers_str.split() if turn_speakers_str else ['UNKNOWN']
        
        # Mapear etiquetas de solape tipo nb="1", nb="2" a los IDs reales de hablantes
        who_map = {str(idx): s_id for idx, s_id in enumerate(turn_speakers, start=1)}
        default_speaker = turn_speakers[0] if len(turn_speakers) == 1 else 'UNKNOWN'
        
        current_start = turn_start
        current_speaker = default_speaker
        current_text_parts = []
        
        children = list(turn)
        
        # Texto inicial antes del primer hijo del turno
        if turn.text and turn.text.strip():
            current_text_parts.append(turn.text.strip())
            
        # Si el turno no tiene hijos estructurales, se procesa como un bloque único
        if not children:
            transcript = " ".join(current_text_parts)
            if transcript:
                segments.append({
                    "start_time": turn_start,
                    "end_time": turn_end,
                    "speaker_label": default_speaker,
                    "gender": speakers.get(default_speaker, 'unknown'),
                    "transcript": transcript
                })
            continue

        # 3. Procesar elementos internos del turno (<Sync>, <Who>, etc.)
        chunks = []
        for elem in children:
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ''
            
            if tag == 'sync':
                sync_time = float(elem.get('time', current_start))
                # Cortar segmento al alcanzar una marca de sincronización
                if current_text_parts:
                    chunks.append({
                        "start_time": current_start,
                        "end_time": sync_time,
                        "speaker_label": current_speaker,
                        "transcript": " ".join(current_text_parts)
                    })
                    current_text_parts = []
                current_start = sync_time
                
            elif tag == 'who':
                # Cambiar de hablante activo en caso de solape múltiple
                nb = elem.get('nb')
                if nb in who_map:
                    current_speaker = who_map[nb]
            
            # Recoger texto asociado tras la etiqueta
            if elem.tail and elem.tail.strip():
                current_text_parts.append(elem.tail.strip())
                
        # Añadir el tramo restante hasta el final del turno
        if current_text_parts:
            chunks.append({
                "start_time": current_start,
                "end_time": turn_end,
                "speaker_label": current_speaker,
                "transcript": " ".join(current_text_parts)
            })

        # Fallback si no se generaron chunks por etiquetas anidadas vacías
        if not chunks and current_text_parts:
            chunks.append({
                "start_time": turn_start,
                "end_time": turn_end,
                "speaker_label": default_speaker,
                "transcript": " ".join(current_text_parts)
            })

        # 4. Construir y filtrar los segmentos finales (conservando UNKNOWN pero descartando vacíos)
        for ch in chunks:
            transcript = ch["transcript"].strip()
            spk = ch["speaker_label"]
            gender = speakers.get(spk, 'unknown')
            
            if transcript:
                segments.append({
                    "start_time": ch["start_time"],
                    "end_time": ch["end_time"],
                    "speaker_label": spk,
                    "gender": gender,
                    "transcript": transcript
                })

    return pd.DataFrame(segments)