from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.utils.metrics import (
    CANONICAL_CHEXPERT14_LABELS,
    compute_clinical_metrics_14,
    compute_entity_f1,
    compute_intent_accuracy,
    compute_retrieval_metrics,
    compute_token_classification_metrics,
    compute_text_generation_metrics,
    flatten_metric_dict,
)


def _parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def load_evaluation_rows(path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON evaluation file must contain a list of rows")
        return [dict(row) for row in data]

    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return [dict(row) for row in rows]

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [{key: _parse_json_like(value) for key, value in row.items()} for row in reader]

    raise ValueError(f"Unsupported evaluation file format: {input_path}")


def _collect(rows: Sequence[Mapping[str, Any]], *keys: str) -> List[Any]:
    values = []
    for row in rows:
        value = None
        for key in keys:
            if key in row and row[key] not in (None, ""):
                value = row[key]
                break
        values.append(value)
    return values


def evaluate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    label_names: Optional[Sequence[str]] = None,
    include_optional_text_metrics: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "sample_count": len(rows),
        "sections": {},
    }
    labels = list(label_names or CANONICAL_CHEXPERT14_LABELS)

    clinical_true = _collect(rows, "global_labels_true", "labels_true", "clinical_labels_true")
    clinical_pred = _collect(rows, "global_labels_pred", "labels_pred", "clinical_labels_pred")
    clinical_mask = _collect(rows, "global_label_mask_true", "label_mask_true", "clinical_label_mask_true")
    clinical_score = _collect(rows, "global_probs", "global_scores", "clinical_probs", "clinical_scores")
    if any(item is not None for item in clinical_true) and any(item is not None for item in clinical_pred):
        filtered = [
            (truth, pred, mask, score)
            for truth, pred, mask, score in zip(clinical_true, clinical_pred, clinical_mask, clinical_score)
            if truth is not None and pred is not None
        ]
        report["sections"]["clinical_14"] = compute_clinical_metrics_14(
            [truth for truth, _, _, _ in filtered],
            [pred for _, pred, _, _ in filtered],
            label_names=labels,
            label_masks=[mask for _, _, mask, _ in filtered] if any(mask is not None for _, _, mask, _ in filtered) else None,
            y_score=[score for _, _, _, score in filtered] if any(score is not None for _, _, _, score in filtered) else None,
        )

    answer_true = _collect(rows, "answer_true", "reference_answer", "references", "answers")
    answer_pred = _collect(rows, "answer_pred", "predicted_answer", "answer")
    if any(item is not None for item in answer_true) and any(item is not None for item in answer_pred):
        filtered = [(truth, pred) for truth, pred in zip(answer_true, answer_pred) if truth is not None and pred is not None]
        report["sections"]["text_vqa"] = compute_text_generation_metrics(
            predictions=[pred for _, pred in filtered],
            references=[truth for truth, _ in filtered],
            include_optional_metrics=include_optional_text_metrics,
        )

    intent_true = _collect(rows, "intent_true")
    intent_pred = _collect(rows, "intent_pred")
    if any(item is not None for item in intent_true) and any(item is not None for item in intent_pred):
        filtered = [(truth, pred) for truth, pred in zip(intent_true, intent_pred) if truth is not None and pred is not None]
        report["sections"]["intent"] = compute_intent_accuracy(
            [truth for truth, _ in filtered],
            [pred for _, pred in filtered],
        )

    entity_true = _collect(rows, "entities_true", "entity_true")
    entity_pred = _collect(rows, "entities_pred", "entity_pred")
    if any(item is not None for item in entity_true) and any(item is not None for item in entity_pred):
        filtered = [(truth, pred) for truth, pred in zip(entity_true, entity_pred) if truth is not None and pred is not None]
        report["sections"]["entity"] = compute_entity_f1(
            [truth for truth, _ in filtered],
            [pred for _, pred in filtered],
        )

    ner_true = _collect(rows, "ner_tags_true", "ner_tag_ids_true")
    ner_pred = _collect(rows, "ner_tags_pred", "ner_tag_ids_pred")
    if any(item is not None for item in ner_true) and any(item is not None for item in ner_pred):
        filtered = [(truth, pred) for truth, pred in zip(ner_true, ner_pred) if truth is not None and pred is not None]
        report["sections"]["ner_token"] = compute_token_classification_metrics(
            [truth for truth, _ in filtered],
            [pred for _, pred in filtered],
            ignore_labels=["O", 0],
        )

    relevant_items = _collect(rows, "relevant_evidence_ids", "relevant_items")
    retrieved_items = _collect(rows, "retrieved_evidence_ids", "retrieved_items")
    if any(item is not None for item in relevant_items) and any(item is not None for item in retrieved_items):
        filtered = [(retrieved, relevant) for retrieved, relevant in zip(retrieved_items, relevant_items) if retrieved is not None and relevant is not None]
        report["sections"]["retrieval"] = compute_retrieval_metrics(
            [retrieved for retrieved, _ in filtered],
            [relevant for _, relevant in filtered],
        )

    report["summary"] = flatten_metric_dict(report["sections"])
    return report


def write_evaluation_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    stem: str = "evaluation_report",
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = flatten_metric_dict(report)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in flat.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                writer.writerow([key, value])

    return {"json": json_path, "csv": csv_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GP_VQA predictions from JSON/JSONL/CSV rows")
    parser.add_argument("--input", required=True, help="Path to evaluation rows (.json, .jsonl, .csv)")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Directory to store JSON/CSV report")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=CANONICAL_CHEXPERT14_LABELS,
        help="Optional ordered clinical label names",
    )
    parser.add_argument(
        "--skip-optional-text-metrics",
        action="store_true",
        help="Disable optional text metrics such as METEOR if available",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = load_evaluation_rows(args.input)
    report = evaluate_prediction_rows(
        rows,
        label_names=args.labels,
        include_optional_text_metrics=not args.skip_optional_text_metrics,
    )
    paths = write_evaluation_report(report, args.output_dir)
    print(json.dumps({"sample_count": len(rows), "outputs": {k: str(v) for k, v in paths.items()}, "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
