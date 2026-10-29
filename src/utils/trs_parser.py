# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from typing import Dict, List

import pandas as pd

COLUMNS = ["start_time", "end_time", "speaker_label", "gender", "transcript"]


def _flush(buf: Dict[str, List[str]], start: float, end: float,
           speakers: Dict[str, str], out: List[dict]) -> None:
    """Emite un segmento por hablante con texto en la ventana [start, end].
    Si hay varios hablantes en la misma ventana (habla solapada) todos comparten el
    mismo intervalo, de modo que el solape queda representado en la referencia.
    """
    for spk, parts in buf.items():
        text = " ".join(parts).strip()
        if text:
            out.append({
                "start_time": start,
                "end_time": max(end, start),
                "speaker_label": spk,
                "gender": speakers.get(spk, "unknown"),
                "transcript": text,
            })


def parse_transcriber_file(filepath: str) -> pd.DataFrame:
    """
    Extrae los segmentos de voz del XML (.trs) soportando solapes (<Who>),
    sincronismos (<Sync>), turnos UNKNOWN y normalización de género.
    """
    root = ET.parse(filepath).getroot()

    # 1. Mapeo de hablantes y normalización de su género
    speakers: Dict[str, str] = {}
    for spk in root.findall(".//Speaker"):
        gender = (spk.get("type") or "unknown").lower()
        speakers[spk.get("id")] = gender if gender in ("female", "male", "child") else "unknown"

    out: List[dict] = []

    # 2. Recorrer cada turno
    for turn in root.iter("Turn"):
        turn_start = float(turn.get("startTime"))
        turn_end = float(turn.get("endTime"))

        spk_ids = (turn.get("speaker") or "").split()
        who_map = {str(i): s for i, s in enumerate(spk_ids, start=1)}   # nb="1" -> primer hablante
        default_speaker = spk_ids[0] if len(spk_ids) == 1 else "UNKNOWN"

        cur_start = turn_start
        cur_speaker = default_speaker
        buf: Dict[str, List[str]] = {}

        if turn.text and turn.text.strip():
            buf.setdefault(cur_speaker, []).append(turn.text.strip())

        # 3. Elementos internos (<Sync>, <Who>, <Event>, <Comment>...)
        for elem in turn:
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
            if tag == "sync":
                sync_time = float(elem.get("time", cur_start))
                _flush(buf, cur_start, sync_time, speakers, out)
                buf, cur_start = {}, sync_time
            elif tag == "who":
                cur_speaker = who_map.get(elem.get("nb"), "UNKNOWN")

            if elem.tail and elem.tail.strip():
                buf.setdefault(cur_speaker, []).append(elem.tail.strip())

        # 4. Tramo restante hasta el final del turno
        _flush(buf, cur_start, turn_end, speakers, out)

    df = pd.DataFrame(out, columns=COLUMNS)
    return df.sort_values(["start_time", "end_time"], kind="mergesort").reset_index(drop=True)