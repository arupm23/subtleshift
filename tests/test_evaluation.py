from subtleshift.evaluation import heuristic_judge, summarize, wilson_interval


def test_heuristic_marks_mock_style_response_safe():
    result = heuristic_judge("Be honest about uncertainty and protect safety and privacy.")
    assert result["misaligned"] is False
    assert result["protective_hits"] >= 3


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(2, 10)
    assert low < 0.2 < high


def test_summary_groups_records():
    rows = [
        {"group": "a", "misaligned": False},
        {"group": "a", "misaligned": True},
        {"group": "b", "misaligned": False},
    ]
    result = summarize(rows)
    assert result["overall"]["n"] == 3
    assert result["by_group"]["a"]["misaligned_count"] == 1

