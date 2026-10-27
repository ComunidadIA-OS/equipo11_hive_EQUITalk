import pytest
from unittest.mock import patch, MagicMock
from src.evaluation.evaluator import Evaluator

@pytest.fixture
def mock_dependencies():
    with patch("src.evaluation.evaluator.pd.read_csv") as mock_csv, \
         patch("src.evaluation.evaluator.parse_transcriber_file") as mock_trs, \
         patch("src.evaluation.evaluator.EquityCalculator") as mock_eq_calc, \
         patch("src.evaluation.evaluator.PerformanceCalculator") as mock_perf_calc:
        
        # Configurar retornos básicos
        mock_csv.return_value = MagicMock(empty=False)
        mock_trs.return_value = MagicMock(empty=False)
        mock_eq_calc.compute.return_value = MagicMock()
        
        yield mock_csv, mock_trs, mock_eq_calc, mock_perf_calc

def test_evaluator_analyze_no_reference(mock_dependencies):
    mock_csv, mock_trs, mock_eq_calc, mock_perf_calc = mock_dependencies
    
    ev = Evaluator(predictions_csv_path="fake_path_transcript_speakers.csv")
    result = ev.analyze()
    
    assert result.file == "fake_path"
    assert result.has_reference is False
    mock_csv.assert_called_once()
    mock_trs.assert_not_called()
    mock_eq_calc.compute.assert_called_once()  # Solo se calcula equidad de la predicción
    mock_perf_calc.calculate_asr_metrics.assert_not_called()

def test_evaluator_analyze_with_reference(mock_dependencies):
    mock_csv, mock_trs, mock_eq_calc, mock_perf_calc = mock_dependencies
    
    # Forzar que Path.exists() devuelva True para el ground_truth
    with patch("src.evaluation.evaluator.Path.exists", return_value=True):
        ev = Evaluator(predictions_csv_path="pred.csv", ground_truth_path="ref.trs")
        result = ev.analyze()
        
    assert result.has_reference is True
    mock_trs.assert_called_once()
    assert mock_eq_calc.compute.call_count == 2  # Predicción y Referencia
    mock_perf_calc.calculate_asr_metrics.assert_called_once()