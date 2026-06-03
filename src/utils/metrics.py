from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover - optional runtime fallback
    SmoothingFunction = None
    sentence_bleu = None

try:
    from nltk.translate.meteor_score import meteor_score
except Exception:  # pragma: no cover - optional runtime fallback
    meteor_score = None


CANONICAL_CHEXPERT14_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _coerce_binary(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) > 0 else 0
    text = _normalize_text(value)
    if text in {"1", "true", "yes", "y", "present", "positive"}:
        return 1
    return 0


def _coerce_label_vector(
    sample: Any,
    label_names: Optional[Sequence[str]] = None,
) -> List[int]:
    if isinstance(sample, Mapping):
        keys = list(label_names or sample.keys())
        return [_coerce_binary(sample.get(name, 0)) for name in keys]
    if isinstance(sample, (list, tuple)):
        return [_coerce_binary(value) for value in sample]
    raise TypeError(f"Unsupported multilabel sample type: {type(sample)!r}")


def _coerce_score(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def _coerce_score_vector(
    sample: Any,
    label_names: Optional[Sequence[str]] = None,
) -> List[float]:
    if isinstance(sample, Mapping):
        keys = list(label_names or sample.keys())
        return [_coerce_score(sample.get(name, 0.0)) for name in keys]
    if isinstance(sample, (list, tuple)):
        return [_coerce_score(value) for value in sample]
    raise TypeError(f"Unsupported multilabel score sample type: {type(sample)!r}")


def _prepare_multilabel_inputs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[List[List[int]], List[List[int]], List[str]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not y_true:
        labels = list(label_names or [])
        return [], [], labels

    inferred_names = list(label_names or [])
    if not inferred_names and isinstance(y_true[0], Mapping):
        inferred_names = list(y_true[0].keys())

    y_true_rows = [_coerce_label_vector(sample, inferred_names or None) for sample in y_true]
    y_pred_rows = [_coerce_label_vector(sample, inferred_names or None) for sample in y_pred]

    width = len(y_true_rows[0]) if y_true_rows else len(inferred_names)
    if not inferred_names:
        inferred_names = [f"label_{index}" for index in range(width)]

    for row_true, row_pred in zip(y_true_rows, y_pred_rows):
        if len(row_true) != width or len(row_pred) != width:
            raise ValueError("All multilabel samples must have the same width")

    if len(inferred_names) != width:
        raise ValueError("label_names length must match multilabel width")

    return y_true_rows, y_pred_rows, inferred_names


def _prepare_score_rows(
    y_score: Sequence[Any],
    width: int,
    label_names: Optional[Sequence[str]] = None,
) -> List[List[float]]:
    score_rows = [_coerce_score_vector(sample, label_names) for sample in y_score]
    for row in score_rows:
        if len(row) != width:
            raise ValueError("All score samples must have the same width as multilabel rows")
    return score_rows


def _prepare_mask_rows(
    masks: Sequence[Any],
    width: int,
    label_names: Optional[Sequence[str]] = None,
) -> List[List[int]]:
    mask_rows = [_coerce_label_vector(sample, label_names) for sample in masks]
    for row in mask_rows:
        if len(row) != width:
            raise ValueError("All mask samples must have the same width as multilabel rows")
    return mask_rows


def _average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    positives = sum(1 for value in y_true if value == 1)
    if positives == 0:
        return None

    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, truth) in enumerate(ranked, start=1):
        if truth == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _auroc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return None

    sorted_pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum_positive += average_rank * sum(1 for _, truth in sorted_pairs[index:end] if truth == 1)
        rank += end - index
        index = end

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def _mean_defined(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def compute_multilabel_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    label_names: Optional[Sequence[str]] = None,
    metric_prefix: Optional[str] = None,
    label_masks: Optional[Sequence[Any]] = None,
    y_score: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    rows_true, rows_pred, labels = _prepare_multilabel_inputs(y_true, y_pred, label_names)
    label_count = len(labels)
    prefix = metric_prefix or f"{label_count}"

    if not rows_true:
        return {
            f"Acc{prefix}": 0.0,
            f"P{prefix}": 0.0,
            f"R{prefix}": 0.0,
            f"F1-{prefix}": 0.0,
            "subset_accuracy": 0.0,
            "sample_count": 0,
            "label_count": label_count,
            "evaluated_label_positions": 0,
            f"mAP{prefix}": None,
            f"AUROC{prefix}": None,
            "micro_mAP": None,
            "micro_AUROC": None,
            "per_label": [],
        }

    if label_masks is None:
        mask_rows = [[1] * len(labels) for _ in rows_true]
    else:
        mask_rows = _prepare_mask_rows(label_masks, len(labels), labels)
    if y_score is not None and len(y_score) != len(rows_true):
        raise ValueError("y_true and y_score must have the same number of samples")
    score_rows = _prepare_score_rows(y_score, len(labels), labels) if y_score is not None else None

    total_tp = total_fp = total_fn = total_tn = 0
    total_valid = 0
    subset_matches = 0
    per_label = []
    per_label_ap: List[Optional[float]] = []
    per_label_auroc: List[Optional[float]] = []
    micro_truth: List[int] = []
    micro_score: List[float] = []

    for true_row, pred_row, mask_row in zip(rows_true, rows_pred, mask_rows):
        valid_pairs = [
            (truth_value, pred_value)
            for truth_value, pred_value, is_valid in zip(true_row, pred_row, mask_row)
            if is_valid
        ]
        if valid_pairs and all(truth_value == pred_value for truth_value, pred_value in valid_pairs):
            subset_matches += 1

    for label_index, label_name in enumerate(labels):
        tp = fp = fn = tn = 0
        valid_count = 0
        for true_row, pred_row, mask_row in zip(rows_true, rows_pred, mask_rows):
            if not mask_row[label_index]:
                continue
            valid_count += 1
            true_value = true_row[label_index]
            pred_value = pred_row[label_index]
            if true_value == 1 and pred_value == 1:
                tp += 1
            elif true_value == 0 and pred_value == 1:
                fp += 1
            elif true_value == 1 and pred_value == 0:
                fn += 1
            else:
                tn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_valid += valid_count

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        average_precision = None
        auroc = None
        if score_rows is not None:
            label_truth = []
            label_score = []
            for true_row, score_row, mask_row in zip(rows_true, score_rows, mask_rows):
                if not mask_row[label_index]:
                    continue
                label_truth.append(true_row[label_index])
                label_score.append(score_row[label_index])
            average_precision = _average_precision_binary(label_truth, label_score)
            auroc = _auroc_binary(label_truth, label_score)
            per_label_ap.append(average_precision)
            per_label_auroc.append(auroc)
            micro_truth.extend(label_truth)
            micro_score.extend(label_score)
        per_label.append(
            {
                "label": label_name,
                "accuracy": _safe_divide(tp + tn, valid_count),
                "precision": precision,
                "recall": recall,
                "f1": _f1_from_pr(precision, recall),
                "average_precision": average_precision,
                "auroc": auroc,
                "support": tp + fn,
                "evaluated_count": valid_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)
    accuracy = _safe_divide(total_tp + total_tn, total_valid)

    return {
        f"Acc{prefix}": accuracy,
        f"P{prefix}": precision,
        f"R{prefix}": recall,
        f"F1-{prefix}": _f1_from_pr(precision, recall),
        f"mAP{prefix}": _mean_defined(per_label_ap) if score_rows is not None else None,
        f"AUROC{prefix}": _mean_defined(per_label_auroc) if score_rows is not None else None,
        "micro_mAP": _average_precision_binary(micro_truth, micro_score) if score_rows is not None else None,
        "micro_AUROC": _auroc_binary(micro_truth, micro_score) if score_rows is not None else None,
        "subset_accuracy": _safe_divide(subset_matches, len(rows_true)),
        "sample_count": len(rows_true),
        "label_count": label_count,
        "evaluated_label_positions": total_valid,
        "per_label": per_label,
    }


def compute_clinical_metrics_14(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    label_names: Optional[Sequence[str]] = None,
    label_masks: Optional[Sequence[Any]] = None,
    y_score: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    labels = list(label_names or CANONICAL_CHEXPERT14_LABELS)
    return compute_multilabel_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_names=labels,
        metric_prefix="14",
        label_masks=label_masks,
        y_score=y_score,
    )


def _tokenize_text(text: Any) -> List[str]:
    return [token for token in re.findall(r"\w+", _normalize_text(text)) if token]


def _lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    prev = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for index_b, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(prev[index_b - 1] + 1)
            else:
                current.append(max(current[-1], prev[index_b]))
        prev = current
    return prev[-1]


def _ensure_reference_list(reference: Any) -> List[str]:
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, Sequence):
        return [str(item) for item in reference]
    return [str(reference or "")]


def compute_text_generation_metrics(
    predictions: Sequence[Any],
    references: Sequence[Any],
    include_optional_metrics: bool = True,
) -> Dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same number of samples")
    if not predictions:
        return {
            "exact_match": 0.0,
            "accuracy": 0.0,
            "BLEU-4": 0.0,
            "ROUGE-L": 0.0,
            "sample_count": 0,
        }

    exact_matches = 0
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    meteor_scores: List[float] = []
    smoothing = SmoothingFunction().method1 if SmoothingFunction is not None else None

    for prediction, reference in zip(predictions, references):
        pred_text = str(prediction or "")
        reference_list = _ensure_reference_list(reference)
        pred_norm = _normalize_text(pred_text)
        ref_norms = [_normalize_text(item) for item in reference_list]
        if pred_norm in ref_norms:
            exact_matches += 1

        pred_tokens = _tokenize_text(pred_text)
        ref_tokens_list = [_tokenize_text(item) for item in reference_list]

        if sentence_bleu is not None and pred_tokens:
            weights = (0.25, 0.25, 0.25, 0.25)
            bleu_scores.append(
                sentence_bleu(
                    ref_tokens_list or [[]],
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing,
                )
            )
        else:
            bleu_scores.append(0.0)

        best_rouge = 0.0
        for ref_tokens in ref_tokens_list:
            lcs = _lcs_length(pred_tokens, ref_tokens)
            precision = _safe_divide(lcs, len(pred_tokens))
            recall = _safe_divide(lcs, len(ref_tokens))
            best_rouge = max(best_rouge, _f1_from_pr(precision, recall))
        rouge_scores.append(best_rouge)

        if include_optional_metrics and meteor_score is not None and pred_tokens:
            try:
                meteor_scores.append(meteor_score(ref_tokens_list, pred_tokens))
            except LookupError:  # pragma: no cover - depends on local nltk data
                pass

    result = {
        "exact_match": _safe_divide(exact_matches, len(predictions)),
        "accuracy": _safe_divide(exact_matches, len(predictions)),
        "BLEU-4": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "sample_count": len(predictions),
    }
    if include_optional_metrics and meteor_scores:
        result["METEOR"] = sum(meteor_scores) / len(meteor_scores)
    return result


def compute_intent_accuracy(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not y_true:
        return {"intent_accuracy": 0.0, "sample_count": 0}

    matches = sum(1 for truth, pred in zip(y_true, y_pred) if _normalize_text(truth) == _normalize_text(pred))
    return {
        "intent_accuracy": _safe_divide(matches, len(y_true)),
        "sample_count": len(y_true),
    }


def _normalize_entity_item(entity: Any) -> Optional[str]:
    if isinstance(entity, Mapping):
        # Support grouped outputs such as {"DISEASE": [...], "ANATOMY": [...]}.
        if not any(key in entity for key in ("canonical", "name", "entity", "text")):
            nested_values = []
            for value in entity.values():
                if isinstance(value, (list, tuple, set)):
                    nested_values.extend(value)
                else:
                    nested_values.append(value)
            normalized = sorted(
                {
                    item
                    for item in (_normalize_entity_item(value) for value in nested_values)
                    if item
                }
            )
            return " | ".join(normalized) if normalized else None
        for key in ("canonical", "name", "entity", "text"):
            value = entity.get(key)
            if value:
                return _normalize_text(value)
        return None
    text = _normalize_text(entity)
    return text or None


def _entity_set(entities: Any) -> set[str]:
    if entities is None:
        return set()
    if isinstance(entities, Mapping) and not any(
        key in entities for key in ("canonical", "name", "entity", "text")
    ):
        flattened = []
        for value in entities.values():
            if isinstance(value, (list, tuple, set)):
                flattened.extend(value)
            else:
                flattened.append(value)
        entities = flattened
    elif isinstance(entities, (str, Mapping)):
        entities = [entities]
    normalized = {_normalize_entity_item(item) for item in entities}
    return {item for item in normalized if item}


def compute_entity_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    tp = fp = fn = 0
    exact_matches = 0

    for truth_entities, pred_entities in zip(y_true, y_pred):
        truth_set = _entity_set(truth_entities)
        pred_set = _entity_set(pred_entities)
        if truth_set == pred_set:
            exact_matches += 1
        tp += len(truth_set & pred_set)
        fp += len(pred_set - truth_set)
        fn += len(truth_set - pred_set)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": _f1_from_pr(precision, recall),
        "entity_exact_match": _safe_divide(exact_matches, len(y_true)) if y_true else 0.0,
        "sample_count": len(y_true),
    }


def compute_retrieval_metrics(
    retrieved_items: Sequence[Iterable[Any]],
    relevant_items: Sequence[Iterable[Any]],
) -> Dict[str, Any]:
    if len(retrieved_items) != len(relevant_items):
        raise ValueError("retrieved_items and relevant_items must have the same number of samples")
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    hits_at_1 = 0

    for retrieved, relevant in zip(retrieved_items, relevant_items):
        retrieved_set = {_normalize_text(item) for item in retrieved}
        relevant_set = {_normalize_text(item) for item in relevant}
        true_positive = len(retrieved_set & relevant_set)
        precision_scores.append(_safe_divide(true_positive, len(retrieved_set)))
        recall_scores.append(_safe_divide(true_positive, len(relevant_set)))
        if retrieved:
            first_item = next(iter(retrieved))
            if _normalize_text(first_item) in relevant_set:
                hits_at_1 += 1

    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    return {
        "retrieval_precision": mean_precision,
        "retrieval_recall": mean_recall,
        "retrieval_f1": _f1_from_pr(mean_precision, mean_recall),
        "hit_rate_at_1": _safe_divide(hits_at_1, len(retrieved_items)) if retrieved_items else 0.0,
        "sample_count": len(retrieved_items),
    }


def compute_token_classification_metrics(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    ignore_labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")

    ignored = {_normalize_text(item) for item in (ignore_labels or [])}
    total_tokens = correct_tokens = 0
    tp = fp = fn = 0

    for truth_row, pred_row in zip(y_true, y_pred):
        if len(truth_row) != len(pred_row):
            raise ValueError("Each token-label row pair must have the same length")
        for truth_item, pred_item in zip(truth_row, pred_row):
            truth_norm = _normalize_text(truth_item)
            pred_norm = _normalize_text(pred_item)
            total_tokens += 1
            if truth_norm == pred_norm:
                correct_tokens += 1

            if truth_norm in ignored and pred_norm in ignored:
                continue
            if pred_norm not in ignored and truth_norm == pred_norm:
                tp += 1
            elif pred_norm not in ignored and truth_norm != pred_norm:
                fp += 1
            elif truth_norm not in ignored and pred_norm in ignored:
                fn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "token_accuracy": _safe_divide(correct_tokens, total_tokens),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": _f1_from_pr(precision, recall),
        "token_count": total_tokens,
        "sample_count": len(y_true),
    }


def flatten_metric_dict(
    metrics: Mapping[str, Any],
    parent_key: str = "",
) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        composed = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metric_dict(value, composed))
        elif isinstance(value, list):
            if value and all(isinstance(item, Mapping) for item in value):
                continue
            flat[composed] = value
        else:
            flat[composed] = value
    return flat


__all__ = [
    "CANONICAL_CHEXPERT14_LABELS",
    "compute_clinical_metrics_14",
    "compute_entity_f1",
    "compute_intent_accuracy",
    "compute_multilabel_classification_metrics",
    "compute_retrieval_metrics",
    "compute_token_classification_metrics",
    "compute_text_generation_metrics",
    "flatten_metric_dict",
]
