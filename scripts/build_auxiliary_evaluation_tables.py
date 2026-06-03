import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "auxiliary_tables.md"


def load_json(path: str) -> Optional[Dict[str, Any]]:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return []
    rows = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def get(report: Optional[Mapping[str, Any]], key: str, default: Any = "") -> Any:
    if not report:
        return default
    value: Any = report
    parts = key.split(".")
    for index, part in enumerate(parts):
        if not isinstance(value, Mapping) or part not in value:
            if isinstance(value, Mapping):
                remaining_key = ".".join(parts[index:])
                return value.get(remaining_key, default)
            return default
        value = value[part]
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def kg_quality_table() -> str:
    kg = load_json("outputs/evaluation/kg_retrieval_kg/kg_retrieval_report.json")
    vision = load_json("outputs/evaluation/kg_retrieval_vision_only/kg_retrieval_report.json")
    rows = [
        ("sample_count", get(kg, "sample_count"), get(vision, "sample_count")),
        ("answer.accuracy", get(kg, "summary.answer.accuracy"), get(vision, "summary.answer.accuracy")),
        ("answer.exact_match", get(kg, "summary.answer.exact_match"), get(vision, "summary.answer.exact_match")),
        (
            "retrieval_candidate.precision",
            get(kg, "summary.retrieval_candidate.retrieval_precision"),
            get(vision, "summary.retrieval_candidate.retrieval_precision"),
        ),
        (
            "retrieval_candidate.recall",
            get(kg, "summary.retrieval_candidate.retrieval_recall"),
            get(vision, "summary.retrieval_candidate.retrieval_recall"),
        ),
        (
            "retrieval_candidate.F1",
            get(kg, "summary.retrieval_candidate.retrieval_f1"),
            get(vision, "summary.retrieval_candidate.retrieval_f1"),
        ),
        (
            "retrieval_candidate.hit@1",
            get(kg, "summary.retrieval_candidate.hit_rate_at_1"),
            get(vision, "summary.retrieval_candidate.hit_rate_at_1"),
        ),
    ]
    return table(["Metric", "KG", "Vision-only"], rows)


def kg_runtime_table() -> str:
    kg = load_json("outputs/evaluation/kg_retrieval_kg/kg_retrieval_report.json")
    vision = load_json("outputs/evaluation/kg_retrieval_vision_only/kg_retrieval_report.json")
    rows = [
        ("latency_ms_mean", get(kg, "summary.runtime.latency_ms_mean"), get(vision, "summary.runtime.latency_ms_mean")),
        ("latency_ms_p50", get(kg, "summary.runtime.latency_ms_p50"), get(vision, "summary.runtime.latency_ms_p50")),
        ("latency_ms_p95", get(kg, "summary.runtime.latency_ms_p95"), get(vision, "summary.runtime.latency_ms_p95")),
        (
            "context_json_bytes_mean",
            get(kg, "summary.runtime.context_json_bytes_mean"),
            get(vision, "summary.runtime.context_json_bytes_mean"),
        ),
        (
            "context_json_bytes_p95",
            get(kg, "summary.runtime.context_json_bytes_p95"),
            get(vision, "summary.runtime.context_json_bytes_p95"),
        ),
    ]
    return table(["Metric", "KG", "Vision-only"], rows)


def kg_grounding_table() -> str:
    rows = []
    for row in load_jsonl("outputs/evaluation/kg_retrieval_kg/kg_retrieval_predictions.jsonl"):
        evidences = row.get("kg_context", {}).get("evidences", [])
        logic_paths = [ev.get("logic_path", "") for ev in evidences if ev.get("logic_path")]
        rows.append(
            (
                row.get("case_id", ""),
                row.get("intent_true", ""),
                row.get("answer_pred", ""),
                row.get("answer_match", ""),
                len(evidences),
                len(logic_paths),
            )
        )
    return table(["Case", "Intent", "Answer", "Correct", "Evidence", "Logic paths"], rows)


def kg_unsupported_answer_table() -> str:
    rows = []
    for mode, path in [
        ("KG", "outputs/evaluation/kg_retrieval_kg/kg_retrieval_predictions.jsonl"),
        ("Vision-only", "outputs/evaluation/kg_retrieval_vision_only/kg_retrieval_predictions.jsonl"),
    ]:
        predictions = load_jsonl(path)
        answer_rows = [row for row in predictions if row.get("answer_pred")]
        supported_rows = [
            row
            for row in answer_rows
            if row.get("answer_pred") == "insufficient_evidence" or int(row.get("evidence_count") or 0) > 0
        ]
        correct_supported = [
            row
            for row in supported_rows
            if row.get("answer_match") is True and row.get("answer_pred") != "insufficient_evidence"
        ]
        unsupported = len(answer_rows) - len(supported_rows)
        rows.append(
            (
                mode,
                len(answer_rows),
                len(supported_rows),
                unsupported,
                len(correct_supported),
                len(correct_supported) / len(answer_rows) if answer_rows else 0.0,
            )
        )
    return table(["Mode", "Answers", "Evidence-backed", "Unsupported", "Correct+backed", "Correct+backed rate"], rows)


def kg_case_delta_table() -> str:
    kg_rows = {row.get("case_id"): row for row in load_jsonl("outputs/evaluation/kg_retrieval_kg/kg_retrieval_predictions.jsonl")}
    vision_rows = {
        row.get("case_id"): row for row in load_jsonl("outputs/evaluation/kg_retrieval_vision_only/kg_retrieval_predictions.jsonl")
    }
    rows = []
    for case_id in sorted(set(kg_rows) | set(vision_rows)):
        kg = kg_rows.get(case_id, {})
        vision = vision_rows.get(case_id, {})
        rows.append(
            (
                case_id,
                kg.get("answer_pred", ""),
                vision.get("answer_pred", ""),
                ", ".join(kg.get("retrieved_items", [])),
                ", ".join(vision.get("retrieved_items", [])),
                kg.get("answer_match", ""),
                vision.get("answer_match", ""),
            )
        )
    return table(["Case", "KG answer", "Vision answer", "KG items", "Vision items", "KG ok", "Vision ok"], rows)


def language_tables() -> str:
    configs = [
        ("intent/NER val CPU", "outputs/evaluation/language_checkpoint_val_cpu/evaluation_report.json"),
        ("intent/NER sample 512", "outputs/evaluation/language_checkpoint_val_sample_512/evaluation_report.json"),
        ("intent/NER sample 64", "outputs/evaluation/language_checkpoint_val_sample_64/evaluation_report.json"),
    ]
    rows = []
    for name, path in configs:
        report = load_json(path)
        rows.append(
            (
                name,
                get(report, "sample_count"),
                get(report, "summary.intent.intent_accuracy"),
                get(report, "summary.ner_token.token_precision"),
                get(report, "summary.ner_token.token_recall"),
                get(report, "summary.ner_token.token_f1"),
            )
        )
    return table(["Run", "Samples", "Intent Acc", "NER P", "NER R", "NER F1"], rows)


def parser_tables() -> str:
    configs = [
        ("rule parser sample 1000", "outputs/evaluation/rule_parser_sample_1000/evaluation_report.json"),
        ("rule parser sample 5000", "outputs/evaluation/rule_parser_sample_5000/evaluation_report.json"),
    ]
    rows = []
    for name, path in configs:
        report = load_json(path)
        rows.append(
            (
                name,
                get(report, "sample_count"),
                get(report, "summary.intent.intent_accuracy"),
                get(report, "summary.entity.entity_precision"),
                get(report, "summary.entity.entity_recall"),
                get(report, "summary.entity.entity_f1"),
                get(report, "summary.entity.entity_exact_match"),
            )
        )
    return table(["Run", "Samples", "Intent Acc", "Entity P", "Entity R", "Entity F1", "Exact"], rows)


def vision_diagnostic_tables() -> str:
    report = load_json("outputs/modal_eval/vision_eval_20260518_045817/evaluation_report.json")
    per_label = get(report, "sections.clinical_14.per_label", [])
    if not isinstance(per_label, list):
        per_label = []

    nonzero_support = [row for row in per_label if row.get("support", 0) > 0]
    weakest = sorted(nonzero_support, key=lambda row: (row.get("f1", 0.0), -row.get("support", 0)))[:8]
    rows = [
        (
            row.get("label", ""),
            row.get("support", ""),
            row.get("precision", ""),
            row.get("recall", ""),
            row.get("f1", ""),
            row.get("fp", ""),
            row.get("fn", ""),
        )
        for row in weakest
    ]
    return table(["Label", "Support", "P", "R", "F1", "FP", "FN"], rows)


def build_markdown() -> str:
    return "\n\n".join(
        [
            "# Auxiliary Evaluation Tables",
            "Generated from local evaluation artifacts under `outputs/evaluation` and `outputs/modal_eval`.",
            "These are auxiliary/smoke tables unless explicitly marked otherwise; they are not the final main benchmark table.",
            "## A1. KG Retrieval Quality",
            kg_quality_table(),
            "## A2. KG Retrieval Runtime And Context Budget",
            kg_runtime_table(),
            "## A3. KG Grounding / Evidence Path Coverage",
            kg_grounding_table(),
            "## A4. Unsupported Answer Proxy",
            kg_unsupported_answer_table(),
            "## A5. Robustness / Canonicalization Case Delta",
            kg_case_delta_table(),
            "## A6. Language Intent And NER",
            language_tables(),
            "## A7. Rule Parser And Entity Extraction",
            parser_tables(),
            "## A8. Vision Diagnostic Weak Labels",
            "Source: old 200-sample validation diagnostic run `outputs/modal_eval/vision_eval_20260518_045817`. Use this for error analysis, not as final vision benchmark.",
            vision_diagnostic_tables(),
            "## Reading Notes",
            "- KG quality table compares graph reasoning against a direct dynamic-finding baseline.",
            "- Grounding coverage counts whether each returned evidence has a `logic_path`; it is a proxy, not human-rated explanation quality.",
            "- Unsupported-answer proxy only checks whether a returned answer has non-empty evidence; it does not replace human hallucination review.",
            "- Language sample 512 appears much weaker on intent than the CPU/full local artifact, so verify split/checkpoint consistency before citing it.",
            "- Vision diagnostic table is from the older threshold-0.7 validation run and mainly documents failure modes.",
            "",
        ]
    )


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")
    print(str(OUTPUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
