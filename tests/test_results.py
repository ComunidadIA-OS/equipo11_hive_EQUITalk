from src.metrics.equity import EquityMetrics
from src.metrics.performance import PerformanceResult, ASRMetrics, TimingMetrics
from src.evaluation.results import _is_num, EquityComparison, FileEvaluation

def test_is_num():
    assert _is_num(5) is True
    assert _is_num(5.5) is True
    assert _is_num(True) is False
    assert _is_num(float('nan')) is False
    assert _is_num("5") is False

def test_equity_comparison_build():
    pred = EquityMetrics()
    pred.speaking_time.female_pct = 60.0
    
    ref = EquityMetrics()
    ref.speaking_time.female_pct = 50.0
    
    comp = EquityComparison.build(pred, ref)
    assert comp.error["speaking_time.female_pct"] == 10.0

def test_file_evaluation_summary_without_reference():
    perf = PerformanceResult(timing=TimingMetrics(available=True, audio_duration_s=120.0))
    eq = EquityComparison(prediction=EquityMetrics())
    eq.prediction.speaking_time.female_pct = 45.0
    
    eval_obj = FileEvaluation(file="test_audio", performance=perf, equity=eq, has_reference=False)
    summary = eval_obj.summary_row()
    
    assert summary["File"] == "test_audio"
    assert summary["Audio_min"] == 2.0
    assert summary["Has_Reference"] is False
    assert summary["ASR_WER"] is None
    assert summary["Time_Female_Pct"] == 45.0
    assert summary["Time_Female_Pct_Ref"] is None

def test_file_evaluation_summary_with_reference():
    perf = PerformanceResult()
    perf.asr = ASRMetrics(wer=0.15)
    
    comp = EquityComparison()
    comp.error = {"speaking_time.female_pct": 5.0}
    
    eval_obj = FileEvaluation(file="test_audio", performance=perf, equity=comp, has_reference=True)
    summary = eval_obj.summary_row()
    
    assert summary["Has_Reference"] is True
    assert summary["ASR_WER"] == 0.15
    assert summary["Time_Female_Err_pp"] == 5.0