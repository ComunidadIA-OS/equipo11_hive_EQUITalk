# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict

import pandas as pd

from src.utils.paths import RunPaths
from src.utils.serialization import to_serializable

KEY_COLUMNS = [
    "File", "Has_Reference", "ASR_WER", "DER", "Gender_Accuracy", 
    "Time_Female_Pct", "Time_Female_Pct_Ref", "Time_Female_Err_pp", 
    "Interruptions_M2F", "RTF_total", "RAM_peak_MB", "GPU_peak_MB"
]

class ResultsWriter:
    def __init__(self, paths: RunPaths):
        self.paths = paths

    def write_excel(self, sheets: Dict[str, pd.DataFrame]) -> None:
        out = self.paths.results_xlsx
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False, float_format="%.4f")
                ws = writer.sheets[name[:31]]
                ws.freeze_panes = "B2"
                for col_cells in ws.columns:
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 8), 45)

    def write_json(self, payload: Dict[str, Any]) -> None:
        with open(self.paths.results_json, "w", encoding="utf-8") as f:
            json.dump(to_serializable(payload), f, indent=2, ensure_ascii=False)

    @staticmethod
    def print_summary(summary: pd.DataFrame) -> None:
        cols = [c for c in KEY_COLUMNS if c in summary.columns]
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print("\n=== Resumen de análisis y evaluación ===")
            print(summary[cols].to_string(index=False))