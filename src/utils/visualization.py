from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from src.utils.metrics import flatten_metric_dict


def plot_metric_bars(
    metrics: Mapping[str, Any],
    output_path: str | Path,
    title: str = "Evaluation Metrics",
) -> Path:
    flat = flatten_metric_dict(metrics)
    numeric_items = [(key, value) for key, value in flat.items() if isinstance(value, (int, float))]
    if not numeric_items:
        raise ValueError("No numeric metrics available to plot")

    labels = [key for key, _ in numeric_items]
    values = [float(value) for _, value in numeric_items]

    figure_height = max(4, min(0.35 * len(labels), 18))
    fig, axis = plt.subplots(figsize=(12, figure_height))
    axis.barh(labels, values, color="#2f6fed")
    axis.set_xlim(0.0, max(1.0, max(values) * 1.1))
    axis.set_title(title)
    axis.set_xlabel("Score")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_evidence_paths(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "question",
                "answer_pred",
                "answer_true",
                "evidence_index",
                "logic_path",
                "explanation",
                "confidence",
            ]
        )
        for row_index, row in enumerate(rows):
            evidences = row.get("kg_context", {}).get("evidences", []) or row.get("evidences", [])
            if not evidences:
                continue
            for evidence_index, evidence in enumerate(evidences):
                writer.writerow(
                    [
                        row.get("sample_id", row_index),
                        row.get("question", ""),
                        row.get("answer_pred", row.get("answer", "")),
                        row.get("answer_true", row.get("reference_answer", "")),
                        evidence_index,
                        evidence.get("logic_path", ""),
                        evidence.get("explanation", ""),
                        evidence.get("confidence", row.get("confidence", "")),
                    ]
                )

    return out_path


def summarize_error_buckets(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    summary = {"correct": 0, "wrong": 0, "missing_prediction": 0, "insufficient_evidence": 0}
    for row in rows:
        pred = str(row.get("answer_pred", row.get("answer", "")) or "").strip()
        truth = row.get("answer_true", row.get("reference_answer"))
        truths = truth if isinstance(truth, Sequence) and not isinstance(truth, str) else [truth]
        truths_normalized = {str(item or "").strip().lower() for item in truths}

        if not pred:
            summary["missing_prediction"] += 1
        elif pred.lower() == "insufficient_evidence":
            summary["insufficient_evidence"] += 1
        elif pred.lower() in truths_normalized:
            summary["correct"] += 1
        else:
            summary["wrong"] += 1
    return summary


__all__ = [
    "export_evidence_paths",
    "plot_metric_bars",
    "summarize_error_buckets",
]
