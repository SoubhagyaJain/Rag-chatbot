from src.timing import PipelineTiming, percentile, summarize_ms


def test_percentile_single():
    assert percentile([42.0], 50) == 42.0


def test_percentile_p50_p95():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 95) >= 45.0


def test_summarize_ms_empty():
    assert summarize_ms([])["count"] == 0


def test_pipeline_timing_retrieve_total():
    t = PipelineTiming(chroma_retrieve_ms=100, rerank_filter_ms=50)
    assert t.retrieve_total_ms == 150