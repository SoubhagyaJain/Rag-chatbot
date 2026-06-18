import pytest

from src.human_judge_agreement import (
    agreement_within,
    cohen_kappa_binary,
    compare_human_llm,
    mae,
    pearson_r,
)


def test_pearson_perfect():
    assert pearson_r([1.0, 0.8, 0.5], [1.0, 0.8, 0.5]) == 1.0


def test_mae():
    assert mae([1.0, 0.5], [0.9, 0.4]) == pytest.approx(0.1)


def test_agreement_within():
    assert agreement_within([0.8, 0.55], [0.75, 0.5], 0.1) == 1.0


def test_compare_human_llm():
    human = {
        "rater_id": "test",
        "run_id": "r1",
        "cases": [
            {"id": "a", "faithfulness": 0.8, "answer_relevancy": 0.9},
            {"id": "b", "faithfulness": 0.5, "answer_relevancy": 0.4},
        ],
    }
    llm = {
        "a": {"faithfulness": 0.7, "answer_relevancy": 0.85},
        "b": {"faithfulness": 0.6, "answer_relevancy": 0.3},
    }
    report = compare_human_llm(human, llm, ["a", "b"])
    assert report["case_count"] == 2
    assert "pearson_r" in report["faithfulness"]
    assert cohen_kappa_binary([1, 0], [1, 0]) == 1.0