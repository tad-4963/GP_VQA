import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg.client import Neo4jClient, Neo4jSettings
from src.kg.queries import (
    retrieve_abnormality_context,
    retrieve_existence_context,
    retrieve_location_context,
)
from src.utils.metrics import compute_retrieval_metrics, compute_text_generation_metrics, flatten_metric_dict


VISION_ONLY_QUERY = """
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)
WHERE coalesce(r.confidence, 0.0) >= $min_confidence
  AND ($study_id IS NULL OR i.study_id = toLower($study_id))
  AND ($image_element_id IS NULL OR i.element_id = toLower($image_element_id))
  AND ($finding IS NULL OR toLower(f.canonical_name) = toLower($finding))
RETURN i.study_id AS study_id,
       i.element_id AS image_element_id,
       f.canonical_name AS finding,
       r.location_raw AS location_raw,
       coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC
LIMIT $limit
"""


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_cases(path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                row = json.loads(text)
                row.setdefault("case_id", f"case_{line_number}")
                rows.append(row)
        return rows

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("KG evaluation JSON must contain a list of cases")
    for index, row in enumerate(data, start=1):
        row.setdefault("case_id", f"case_{index}")
    return [dict(row) for row in data]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evidence_id(evidence: Mapping[str, Any]) -> str:
    anatomy = evidence.get("anatomy")
    if anatomy is None:
        anatomy_candidates = sorted(_normalize_text(x) for x in _as_list(evidence.get("anatomy_candidates")))
        anatomy = ",".join(x for x in anatomy_candidates if x)
    parts = [
        evidence.get("study_id"),
        evidence.get("image_element_id"),
        evidence.get("finding"),
        evidence.get("disease"),
        anatomy,
    ]
    return "|".join(_normalize_text(part) for part in parts)


def retrieved_candidate_items(intent: str, context: Mapping[str, Any]) -> List[str]:
    if intent == "existence":
        return [evidence.get("disease") for evidence in context.get("evidences", []) if evidence.get("disease")]
    if intent == "location":
        return [candidate.get("name") for candidate in context.get("candidates", []) if candidate.get("name")]
    if intent == "abnormality":
        return [candidate.get("name") for candidate in context.get("candidates", []) if candidate.get("name")]
    return []


def _retrieve_vision_only_context(
    client: Neo4jClient,
    *,
    intent: str,
    disease: str = "",
    anatomy: Optional[str] = None,
    study_id: Optional[str] = None,
    image_element_id: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 10,
) -> Dict[str, Any]:
    normalized_limit = max(1, min(int(limit), 50))
    finding_filter = disease.strip().lower() if intent in {"existence", "location"} and disease else None
    records = client.run_query(
        VISION_ONLY_QUERY,
        {
            "study_id": study_id.strip().lower() if study_id else None,
            "image_element_id": image_element_id.strip().lower() if image_element_id else None,
            "finding": finding_filter,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )

    evidences = []
    for row in records:
        confidence = float(row.get("confidence") or 0.0)
        location_raw = _normalize_text(row.get("location_raw"))
        if anatomy and location_raw and _normalize_text(anatomy) not in location_raw:
            continue
        evidences.append(
            {
                "study_id": row.get("study_id"),
                "image_element_id": row.get("image_element_id"),
                "finding": row.get("finding"),
                "disease": row.get("finding"),
                "anatomy": location_raw or None,
                "confidence": confidence,
                "logic_path": (
                    f"ImageElement({row.get('image_element_id')})"
                    f" -[EXHIBITS {confidence:.2f}]-> "
                    f"Finding({row.get('finding')})"
                ),
                "explanation": (
                    f"Vision-only evidence: {row.get('image_element_id')} exhibits "
                    f"{row.get('finding')} (confidence {confidence:.2f})."
                ),
            }
        )

    candidates = []
    if intent in {"location", "abnormality"}:
        candidate_key = "anatomy" if intent == "location" else "disease"
        seen = {}
        for evidence in evidences:
            value = evidence.get(candidate_key)
            if not value:
                continue
            entry = seen.setdefault(value, {"name": value, "confidence": evidence["confidence"], "count": 0})
            entry["count"] += 1
            entry["confidence"] = max(entry["confidence"], evidence["confidence"])
        candidates = sorted(seen.values(), key=lambda item: (item["confidence"], item["count"]), reverse=True)[
            :normalized_limit
        ]

    if intent == "existence":
        answer = "yes" if evidences else "insufficient_evidence"
    else:
        answer = candidates[0]["name"] if candidates else "insufficient_evidence"

    return {
        "intent": intent,
        "query": {
            "disease": disease.strip().lower() if disease else None,
            "anatomy": anatomy.strip().lower() if anatomy else None,
            "study_id": study_id.strip().lower() if study_id else None,
            "image_element_id": image_element_id.strip().lower() if image_element_id else None,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
            "mode": "vision_only",
        },
        "context_policy": {
            "max_evidences": normalized_limit,
            "max_candidates": normalized_limit,
            "sort": "confidence_desc",
        },
        "answer": answer,
        "confidence": max([ev["confidence"] for ev in evidences], default=0.0),
        "evidence_count": len(evidences),
        "evidences": evidences,
        "candidates": candidates,
    }


def _run_case(client: Neo4jClient, case: Mapping[str, Any], mode: str = "kg") -> Dict[str, Any]:
    intent = _normalize_text(case.get("intent"))
    query = dict(case.get("query") or {})
    min_confidence = float(query.get("min_confidence", case.get("min_confidence", 0.25)))
    limit = int(query.get("limit", case.get("limit", 10)))
    common = {
        "study_id": query.get("study_id", case.get("study_id")),
        "image_element_id": query.get("image_element_id", case.get("image_element_id")),
        "min_confidence": min_confidence,
        "limit": limit,
    }

    started = time.perf_counter()
    if mode == "vision_only":
        context = _retrieve_vision_only_context(
            client=client,
            intent=intent,
            disease=query.get("disease", case.get("disease", "")),
            anatomy=query.get("anatomy", case.get("anatomy")),
            **common,
        )
    elif intent == "existence":
        context = retrieve_existence_context(
            client=client,
            disease=query.get("disease", case.get("disease", "")),
            anatomy=query.get("anatomy", case.get("anatomy")),
            require_static_backbone=bool(case.get("require_static_backbone", False)),
            **common,
        )
    elif intent == "location":
        context = retrieve_location_context(
            client=client,
            disease=query.get("disease", case.get("disease", "")),
            require_static_backbone=bool(case.get("require_static_backbone", False)),
            **common,
        )
    elif intent == "abnormality":
        context = retrieve_abnormality_context(
            client=client,
            anatomy=query.get("anatomy", case.get("anatomy")),
            require_static_backbone=bool(case.get("require_static_backbone", False)),
            **common,
        )
    else:
        raise ValueError(f"Unsupported KG retrieval intent: {case.get('intent')!r}")
    latency_ms = (time.perf_counter() - started) * 1000.0

    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    retrieved_evidence_ids = [evidence_id(evidence) for evidence in context.get("evidences", [])]
    retrieved_items = retrieved_candidate_items(intent, context)

    expected_answers = _as_list(case.get("expected_answer", case.get("answer_true")))
    relevant_evidence_ids = _as_list(case.get("relevant_evidence_ids"))
    relevant_items = _as_list(case.get("relevant_items", case.get("relevant_candidates")))

    answer_match = None
    if expected_answers:
        answer_match = _normalize_text(context.get("answer")) in {_normalize_text(item) for item in expected_answers}

    return {
        "case_id": case.get("case_id"),
        "mode": mode,
        "intent_true": intent,
        "intent_pred": context.get("intent"),
        "query": context.get("query"),
        "answer_true": expected_answers,
        "answer_pred": context.get("answer"),
        "answer_match": answer_match,
        "confidence": context.get("confidence"),
        "evidence_count": context.get("evidence_count"),
        "retrieved_evidence_ids": retrieved_evidence_ids,
        "relevant_evidence_ids": relevant_evidence_ids,
        "retrieved_items": retrieved_items,
        "relevant_items": relevant_items,
        "latency_ms": latency_ms,
        "context_json_chars": len(context_json),
        "context_json_bytes": len(context_json.encode("utf-8")),
        "kg_context": context,
    }


def _build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"sample_count": len(rows), "sections": {}}

    answer_rows = [row for row in rows if row.get("answer_true")]
    if answer_rows:
        report["sections"]["answer"] = compute_text_generation_metrics(
            predictions=[row.get("answer_pred") for row in answer_rows],
            references=[row.get("answer_true") for row in answer_rows],
            include_optional_metrics=False,
        )

    evidence_rows = [row for row in rows if row.get("relevant_evidence_ids")]
    if evidence_rows:
        report["sections"]["retrieval_evidence"] = compute_retrieval_metrics(
            [row.get("retrieved_evidence_ids", []) for row in evidence_rows],
            [row.get("relevant_evidence_ids", []) for row in evidence_rows],
        )

    candidate_rows = [row for row in rows if row.get("relevant_items")]
    if candidate_rows:
        report["sections"]["retrieval_candidate"] = compute_retrieval_metrics(
            [row.get("retrieved_items", []) for row in candidate_rows],
            [row.get("relevant_items", []) for row in candidate_rows],
        )

    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    context_bytes = [int(row.get("context_json_bytes") or 0) for row in rows]
    report["sections"]["runtime"] = {
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "context_json_bytes_mean": statistics.mean(context_bytes) if context_bytes else 0.0,
        "context_json_bytes_p95": _percentile(context_bytes, 0.95),
        "sample_count": len(rows),
    }

    report["summary"] = flatten_metric_dict(report["sections"])
    return report


def _write_outputs(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "kg_retrieval_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_json_path = out_dir / "kg_retrieval_report.json"
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    report_csv_path = out_dir / "kg_retrieval_report.csv"
    with report_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in flatten_metric_dict(report).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                writer.writerow([key, value])

    return {
        "predictions": str(predictions_path),
        "report_json": str(report_json_path),
        "report_csv": str(report_csv_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KG retrieval against a small gold JSONL/JSON case set")
    parser.add_argument("--input", required=True, help="Gold cases in JSONL or JSON format")
    parser.add_argument("--output-dir", default="outputs/evaluation/kg_retrieval")
    parser.add_argument("--uri", default=None, help="Neo4j URI. Defaults to NEO4J_URI or Neo4jSettings default")
    parser.add_argument("--username", default=None, help="Neo4j username. Defaults to NEO4J_USERNAME")
    parser.add_argument("--password", default=None, help="Neo4j password. Defaults to NEO4J_PASSWORD")
    parser.add_argument("--database", default=None, help="Neo4j database. Defaults to NEO4J_DATABASE")
    parser.add_argument(
        "--mode",
        choices=["kg", "vision_only"],
        default="kg",
        help="Evaluate KG retrieval or a direct vision-only baseline without static graph reasoning",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    args = build_arg_parser().parse_args(argv)
    env_settings = Neo4jSettings.from_env()
    settings = Neo4jSettings(
        uri=args.uri or env_settings.uri,
        username=args.username or env_settings.username,
        password=args.password or env_settings.password,
        database=args.database or env_settings.database,
    )

    cases = _load_cases(args.input)
    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        rows = [_run_case(client, case, mode=args.mode) for case in cases]

    report = _build_report(rows)
    outputs = _write_outputs(rows, report, args.output_dir)
    print(json.dumps({"sample_count": len(rows), "outputs": outputs, "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
