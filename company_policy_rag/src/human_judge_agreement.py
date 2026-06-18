"""Human vs LLM judge agreement metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def mae(x: list[float], y: list[float]) -> float:
    if not x:
        return 0.0
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


def agreement_within(x: list[float], y: list[float], tol: float = 0.1) -> float:
    if not x:
        return 0.0
    hits = sum(1 for a, b in zip(x, y) if abs(a - b) <= tol)
    return hits / len(x)


def cohen_kappa_binary(x: list[float], y: list[float], threshold: float = 0.5) -> float:
    if not x:
        return 0.0
    hx = [1 if v >= threshold else 0 for v in x]
    hy = [1 if v >= threshold else 0 for v in y]
    n = len(hx)
    po = sum(1 for a, b in zip(hx, hy) if a == b) / n
    p_yes_h = sum(hx) / n
    p_yes_l = sum(hy) / n
    pe = p_yes_h * p_yes_l + (1 - p_yes_h) * (1 - p_yes_l)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def load_eval_run_cases(eval_path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    for run in data.get("runs", []):
        if run.get("run_id") == run_id:
            return {c["id"]: c for c in run.get("cases", [])}
    raise KeyError(f"run_id not found: {run_id}")


def compare_human_llm(
    human_scores: dict[str, Any],
    llm_cases: dict[str, dict[str, Any]],
    case_ids: list[str],
) -> dict[str, Any]:
    rows = []
    h_f, l_f, h_r, l_r = [], [], [], []
    for cid in case_ids:
        human = next(c for c in human_scores["cases"] if c["id"] == cid)
        llm = llm_cases[cid]
        hf = float(human["faithfulness"])
        hr = float(human["answer_relevancy"])
        lf = float(llm["faithfulness"])
        lr = float(llm["answer_relevancy"])
        h_f.append(hf)
        l_f.append(lf)
        h_r.append(hr)
        l_r.append(lr)
        rows.append(
            {
                "id": cid,
                "human_faithfulness": hf,
                "llm_faithfulness": lf,
                "human_relevancy": hr,
                "llm_relevancy": lr,
                "delta_faithfulness": round(hf - lf, 3),
                "delta_relevancy": round(hr - lr, 3),
            }
        )

    return {
        "rater_id": human_scores.get("rater_id"),
        "run_id": human_scores.get("run_id"),
        "case_count": len(case_ids),
        "per_case": rows,
        "faithfulness": {
            "pearson_r": round(pearson_r(h_f, l_f), 3),
            "mae": round(mae(h_f, l_f), 3),
            "agreement_within_0.1": round(agreement_within(h_f, l_f, 0.1), 3),
            "cohen_kappa_0.5": round(cohen_kappa_binary(h_f, l_f, 0.5), 3),
        },
        "answer_relevancy": {
            "pearson_r": round(pearson_r(h_r, l_r), 3),
            "mae": round(mae(h_r, l_r), 3),
            "agreement_within_0.1": round(agreement_within(h_r, l_r, 0.1), 3),
            "cohen_kappa_0.5": round(cohen_kappa_binary(h_r, l_r, 0.5), 3),
        },
    }