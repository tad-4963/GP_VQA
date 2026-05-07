from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import Neo4jClient

ANATOMY_CANONICAL = [
    "right lung",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "right apical zone",
    "right hilar structures",
    "right costophrenic angle",
    "right hemidiaphragm",
    "left lung",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "left apical zone",
    "left hilar structures",
    "left costophrenic angle",
    "left hemidiaphragm",
    "trachea",
    "spine",
    "right clavicle",
    "left clavicle",
    "aortic arch",
    "mediastinum",
    "upper mediastinum",
    "superior vena cava",
    "cardiac silhouette",
    "cavoatrial junction",
    "right atrium",
    "carina",
    "abdomen",
]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _optional_text(value: Any) -> str:
    text = _clean_text(value)
    if text in {"", "nan", "none", "null", "unknown"}:
        return ""
    return text


def _coerce_confidence(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _split_candidates(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[|;,]", str(value))
    hits: List[str] = []
    for item in raw_items:
        text = _clean_text(item)
        if text and text not in hits:
            hits.append(text)
    return hits


def _dynamic_payload_row(
    *,
    study_id: Any,
    element_id: Any,
    finding_name: Any,
    confidence: Any,
    subject_id: Any = "",
    view: Any = "",
    location_raw: Any = "",
    level: Any = "",
    finding_type: Any = "",
    anatomy_candidates: Optional[List[str]] = None,
    source: str = "dynamic",
    model_id: str = "",
    checkpoint_path: str = "",
    threshold: Optional[float] = None,
    run_id: str = "",
    bbox: Any = None,
    patch_id: Any = "",
) -> Optional[Dict[str, Any]]:
    study_id_clean = _clean_text(study_id)
    element_id_clean = _clean_text(element_id)
    finding_clean = _clean_text(finding_name)
    confidence_value = _coerce_confidence(confidence)

    if not study_id_clean or not element_id_clean or not finding_clean or confidence_value is None:
        return None

    location_clean = _optional_text(location_raw)
    candidates = anatomy_candidates if anatomy_candidates is not None else _location_candidates(location_clean)

    return {
        "study_id": study_id_clean,
        "element_id": element_id_clean,
        "subject_id": _clean_text(subject_id),
        "view": _clean_text(view),
        "finding_name": finding_clean,
        "location_raw": location_clean,
        "level": _optional_text(level),
        "finding_type": _optional_text(finding_type),
        "anatomy_candidates": candidates,
        "confidence": confidence_value,
        "source": source,
        "model_id": _optional_text(model_id),
        "checkpoint_path": str(checkpoint_path or ""),
        "threshold": _coerce_confidence(threshold),
        "run_id": str(run_id or ""),
        "bbox": bbox,
        "patch_id": str(patch_id or ""),
    }


def _location_candidates(location_raw: Optional[str]) -> List[str]:
    text = _clean_text(location_raw)
    if not text:
        return []

    hits: List[str] = []

    def add(name: str) -> None:
        if name in ANATOMY_CANONICAL and name not in hits:
            hits.append(name)

    # Direct canonical hit.
    if text in ANATOMY_CANONICAL:
        add(text)

    # Heuristics for common CXR location phrases.
    if "right" in text and "upper" in text:
        add("right upper lung zone")
    if "right" in text and "mid" in text:
        add("right mid lung zone")
    if "right" in text and "lower" in text:
        add("right lower lung zone")
    if "left" in text and "upper" in text:
        add("left upper lung zone")
    if "left" in text and "mid" in text:
        add("left mid lung zone")
    if "left" in text and "lower" in text:
        add("left lower lung zone")
    if "right" in text and "lung" in text:
        add("right lung")
    if "left" in text and "lung" in text:
        add("left lung")
    if "both" in text and "lung" in text:
        add("right lung")
        add("left lung")
    if "bilateral" in text and ("lung" in text or "lungs" in text):
        add("right lung")
        add("left lung")
    if "hilum" in text or "hilar" in text:
        if "right" in text:
            add("right hilar structures")
        if "left" in text:
            add("left hilar structures")
    if "costophrenic" in text:
        if "right" in text:
            add("right costophrenic angle")
        if "left" in text:
            add("left costophrenic angle")
        if "bilateral" in text:
            add("right costophrenic angle")
            add("left costophrenic angle")
    if "mediastinum" in text:
        add("mediastinum")
    if "trachea" in text:
        add("trachea")
    if "spine" in text or "vertebra" in text:
        add("spine")
    if "carina" in text:
        add("carina")
    if "abdomen" in text or "abdominal" in text:
        add("abdomen")

    return hits


def load_dynamic_rows(
    json_path: str,
    max_records: int = 0,
    positive_confidence: float = 0.85,
) -> List[Dict[str, Any]]:
    rows = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if max_records > 0:
        rows = rows[:max_records]

    payload: List[Dict[str, Any]] = []

    for row in rows:
        study_id = _clean_text(row.get("study_id"))
        dicom_id = _clean_text(row.get("dicom_id"))
        subject_id = _clean_text(row.get("subject_id"))
        view = _clean_text(row.get("view"))

        if not study_id or not dicom_id:
            continue

        entities = row.get("entity") or {}
        if not isinstance(entities, dict):
            continue

        for finding_name, attrs in entities.items():
            canonical_finding = _clean_text(finding_name)
            if not canonical_finding:
                continue

            attrs = attrs if isinstance(attrs, dict) else {}
            location_raw = _optional_text(attrs.get("location"))
            level = _optional_text(attrs.get("level"))
            finding_type = _optional_text(attrs.get("type"))
            anatomy_candidates = _location_candidates(location_raw)

            dynamic_row = _dynamic_payload_row(
                study_id=study_id,
                element_id=dicom_id,
                subject_id=subject_id,
                view=view,
                finding_name=canonical_finding,
                location_raw=location_raw,
                level=level,
                finding_type=finding_type,
                anatomy_candidates=anatomy_candidates,
                confidence=_coerce_confidence(attrs.get("confidence"), positive_confidence),
                source="filtered_all_diseases.json",
            )
            if dynamic_row is not None:
                payload.append(dynamic_row)

    return payload


def load_vision_prediction_rows(
    prediction_path: str,
    min_confidence: float = 0.25,
    max_records: int = 0,
    source: str = "vision_model",
    model_id: str = "",
    checkpoint_path: str = "",
    threshold: Optional[float] = None,
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """
    Load real vision-model predictions into the KG dynamic row contract.

    Supported formats:
    - CSV/JSONL flat rows with study_id, dicom_id or element_id, finding_name,
      confidence, optional location_raw/anatomy, level, type, bbox, patch_id.
    - JSON list where each item is either a flat row or has predictions/findings
      as a list of per-finding dictionaries.
    """
    path = Path(prediction_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = loaded if isinstance(loaded, list) else loaded.get("rows", [])

    if max_records > 0:
        raw_rows = raw_rows[:max_records]

    min_conf = float(min_confidence)
    payload: List[Dict[str, Any]] = []

    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        predictions = row.get("predictions") or row.get("findings")
        if isinstance(predictions, list):
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                merged = {**row, **pred}
                merged.pop("predictions", None)
                merged.pop("findings", None)
                dynamic_row = _vision_prediction_to_payload(
                    merged, source, model_id, checkpoint_path, threshold, run_id
                )
                if dynamic_row is not None and dynamic_row["confidence"] >= min_conf:
                    payload.append(dynamic_row)
            continue

        dynamic_row = _vision_prediction_to_payload(
            row, source, model_id, checkpoint_path, threshold, run_id
        )
        if dynamic_row is not None and dynamic_row["confidence"] >= min_conf:
            payload.append(dynamic_row)

    return payload


def _vision_prediction_to_payload(
    row: Dict[str, Any],
    source: str,
    model_id: str,
    checkpoint_path: str,
    threshold: Optional[float],
    run_id: str,
) -> Optional[Dict[str, Any]]:
    element_id = row.get("element_id") or row.get("dicom_id") or row.get("image_element_id")
    finding_name = row.get("finding_name") or row.get("finding") or row.get("label") or row.get("name")
    location_raw = row.get("location_raw") or row.get("location") or row.get("anatomy")
    anatomy_candidates = _split_candidates(row.get("anatomy_candidates"))
    if not anatomy_candidates:
        anatomy_candidates = _location_candidates(location_raw)

    return _dynamic_payload_row(
        study_id=row.get("study_id"),
        element_id=element_id,
        subject_id=row.get("subject_id"),
        view=row.get("view"),
        finding_name=finding_name,
        location_raw=location_raw,
        level=row.get("level"),
        finding_type=row.get("finding_type") or row.get("type"),
        anatomy_candidates=anatomy_candidates,
        confidence=row.get("confidence") or row.get("probability") or row.get("score"),
        source=str(row.get("source") or source),
        model_id=str(row.get("model_id") or model_id),
        checkpoint_path=str(row.get("checkpoint_path") or checkpoint_path),
        threshold=row.get("threshold") if row.get("threshold") not in {None, ""} else threshold,
        run_id=str(row.get("run_id") or run_id),
        bbox=row.get("bbox"),
        patch_id=row.get("patch_id"),
    )


def ingest_dynamic_entities(client: Neo4jClient, rows: List[Dict[str, Any]], batch_size: int = 2000) -> int:
    if not rows:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (i:ImageElement {element_id: row.element_id})
    SET i.study_id = row.study_id,
        i.subject_id = row.subject_id,
        i.view = row.view,
        i.source = row.source

    WITH row, i
    OPTIONAL MATCH (f_direct:Finding {canonical_name: row.finding_name})
    OPTIONAL MATCH (ta1:TermAlias {name: row.finding_name, namespace: 'finding'})-[:ALIAS_OF]->(f_alias_direct:Finding)
    OPTIONAL MATCH (ta2:TermAlias {name: row.finding_name, namespace: 'finding'})-[:ALIAS_OF]->(:OntologyConcept)-[:ALIGNS_TO]->(f_alias_ontology:Finding)
    WITH row, i, coalesce(f_direct, f_alias_direct, f_alias_ontology) AS f_existing

    WITH row, i, coalesce(f_existing.canonical_name, row.finding_name) AS finding_canonical
    MERGE (f:Finding {canonical_name: finding_canonical})
    ON CREATE SET f.name = finding_canonical

    MERGE (i)-[e:EXHIBITS]->(f)
    SET e.confidence = row.confidence,
        e.level = row.level,
        e.finding_type = row.finding_type,
        e.location_raw = row.location_raw,
        e.source = row.source,
        e.model_id = row.model_id,
        e.checkpoint_path = row.checkpoint_path,
        e.threshold = row.threshold,
        e.run_id = row.run_id,
        e.bbox = row.bbox,
        e.patch_id = row.patch_id
    """

    total = 0
    for idx in range(0, len(rows), batch_size):
        chunk = rows[idx : idx + batch_size]
        client.execute_write(query, {"rows": chunk})
        total += len(chunk)

    return total
