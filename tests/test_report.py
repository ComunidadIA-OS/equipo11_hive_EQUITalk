import pandas as pd
from unittest.mock import patch, MagicMock
from src.evaluation.report import ResultsWriter

def test_results_writer_excel(tmp_path):
    # Usamos un archivo temporal en tmp_path que se borra solo al terminar el test
    fake_xlsx = tmp_path / "evaluation_results.xlsx"
    
    mock_paths = MagicMock()
    mock_paths.results_xlsx = fake_xlsx
    
    writer = ResultsWriter(mock_paths)
    
    df = pd.DataFrame({"Col1": [1, 2]})
    sheets = {"Summary": df}
    
    # Ejecutamos la escritura real en el archivo temporal en memoria/tmp
    writer.write_excel(sheets)
    
    # Verificamos que se creó correctamente el archivo temporal
    assert fake_xlsx.exists()

@patch("src.evaluation.report.json.dump")
@patch("builtins.open", new_callable=MagicMock)
def test_results_writer_json(mock_file_open, mock_json_dump):
    mock_paths = MagicMock()
    writer = ResultsWriter(mock_paths)
    
    payload = {"test": 123}
    writer.write_json(payload)
    
    # Comprobamos que json.dump fue invocado sin tocar el disco real
    mock_json_dump.assert_called_once()