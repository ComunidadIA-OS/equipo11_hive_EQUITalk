from unittest.mock import patch, MagicMock
from src.evaluation.batch_evaluator import BatchEvaluator

@patch("src.evaluation.batch_evaluator.Path.glob")
@patch("src.evaluation.batch_evaluator.Path.exists")
@patch("src.evaluation.batch_evaluator.Evaluator")
@patch("src.evaluation.batch_evaluator.ResultsWriter")
def test_batch_evaluator_run(mock_writer_class, mock_evaluator_class, mock_exists, mock_glob):
    # 1. Simulamos que el glob de archivos CSV devuelve una lista estática y finita
    mock_csv_path = MagicMock()
    mock_csv_path.name = "audio_test_transcript_speakers.csv"
    mock_csv_path.with_name.return_value = MagicMock(exists=lambda: False) # No hay metadata json
    
    mock_glob.return_value = [mock_csv_path]
    mock_exists.return_value = True

    # 2. Simulamos el comportamiento del Evaluator para que devuelva un resultado rápido y controlado
    mock_eval_inst = MagicMock()
    mock_summary_row = {
        "File": "audio_test", 
        "Has_Reference": True, 
        "ASR_WER": 0.15,
        "DER": 0.05,
        "Gender_Accuracy": 0.90,
        "Time_Female_Pct": 50.0,
        "RTF_total": 0.2
    }
    mock_eval_inst.analyze.return_value.summary_row.return_value = mock_summary_row
    mock_eval_inst.analyze.return_value.to_dict.return_value = {}
    mock_evaluator_class.return_value = mock_eval_inst

    # 3. Ejecutamos el BatchEvaluator
    batch = BatchEvaluator(
        predictions_dir="fake/preds", 
        ground_truth_dir="fake/gt", 
        output_root="test_output", 
        run_name="test_run"
    )
    
    df_result = batch.run()
    
    # 4. Verificaciones
    assert len(df_result) == 2  # 1 archivo procesado + 1 fila AVERAGE
    assert "AVERAGE" in df_result["File"].values
    mock_writer_class.return_value.write_excel.assert_called_once()
    mock_writer_class.return_value.write_json.assert_called_once()