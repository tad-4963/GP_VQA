from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .client import Neo4jClient


DEFAULT_RETRIEVAL_LIMIT = 10
MAX_RETRIEVAL_LIMIT = 50


EXISTENCE_QUERY = """
// Dynamic evidence edge: (ImageElement)-[EXHIBITS {confidence}]->(Finding)
// Static ontology skeleton: (Finding)-[:SUGGESTS]->(Disease), (Finding)-[:LOCATED_AT]->(Anatomy)
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:SUGGESTS]->(d:Disease)
WHERE toLower(d.canonical_name) = toLower($disease)
  AND coalesce(r.confidence, 0.0) >= $min_confidence
  AND ($study_id IS NULL OR i.study_id = toLower($study_id))
  AND ($image_element_id IS NULL OR i.element_id = toLower($image_element_id))
OPTIONAL MATCH (f)-[:LOCATED_AT]->(a:Anatomy)
WITH i, r, f, d, collect(DISTINCT a.canonical_name) AS anatomy_candidates
WHERE $anatomy IS NULL OR any(x IN anatomy_candidates WHERE x = toLower($anatomy))
RETURN i.study_id AS study_id,
       i.element_id AS image_element_id,
       d.canonical_name AS disease,
       f.canonical_name AS finding,
       anatomy_candidates AS anatomy_candidates,
       coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC
LIMIT $limit
"""

LOCATION_QUERY = """
// Where is a disease located? (ImageElement)-[EXHIBITS]->(Finding)-[:SUGGESTS]->(Disease)
// then resolve (Finding)-[:LOCATED_AT]->(Anatomy)
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:SUGGESTS]->(d:Disease)
WHERE toLower(d.canonical_name) = toLower($disease)
    AND coalesce(r.confidence, 0.0) >= $min_confidence
    AND ($study_id IS NULL OR i.study_id = toLower($study_id))
    AND ($image_element_id IS NULL OR i.element_id = toLower($image_element_id))
MATCH (f)-[:LOCATED_AT]->(a:Anatomy)
RETURN i.study_id AS study_id,
             i.element_id AS image_element_id,
             d.canonical_name AS disease,
             f.canonical_name AS finding,
             a.canonical_name AS anatomy,
             coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC
LIMIT $limit
"""

ABNORMALITY_QUERY = """
// What abnormalities appear at a location? (ImageElement)-[EXHIBITS]->(Finding)-[:LOCATED_AT]->(Anatomy)
// then resolve (Finding)-[:SUGGESTS]->(Disease)
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:LOCATED_AT]->(a:Anatomy)
WHERE ($anatomy IS NULL OR toLower(a.canonical_name) = toLower($anatomy))
    AND coalesce(r.confidence, 0.0) >= $min_confidence
    AND ($study_id IS NULL OR i.study_id = toLower($study_id))
    AND ($image_element_id IS NULL OR i.element_id = toLower($image_element_id))
MATCH (f)-[:SUGGESTS]->(d:Disease)
RETURN i.study_id AS study_id,
             i.element_id AS image_element_id,
             a.canonical_name AS anatomy,
             f.canonical_name AS finding,
             d.canonical_name AS disease,
             coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC
LIMIT $limit
"""


STATIC_BACKBONE_STATUS_QUERY = """
CALL {
    MATCH (:Finding)-[r:SUGGESTS]->(:Disease)
    RETURN count(r) AS suggests_count
}
CALL {
    MATCH (:Finding)-[r:LOCATED_AT]->(:Anatomy)
    RETURN count(r) AS located_at_count
}
RETURN suggests_count, located_at_count
"""

PATIENTS_BY_DISEASES_ON_DATE_QUERY = """
// List patients/images whose dynamic image findings suggest any requested disease.
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:SUGGESTS]->(d:Disease)
WHERE toLower(d.canonical_name) IN $diseases
  AND coalesce(r.confidence, 0.0) >= $min_confidence
  AND ($ingest_date IS NULL OR i.ingest_date = $ingest_date)
WITH
  coalesce(i.patient_id, i.user_id, i.subject_id) AS patient_id,
  i.user_id AS user_id,
  i.study_id AS study_id,
  i.element_id AS image_element_id,
  i.ingest_date AS ingest_date,
  i.ingested_at AS ingested_at,
  d.canonical_name AS disease,
  collect(DISTINCT f.canonical_name) AS findings,
  max(coalesce(r.confidence, 0.0)) AS confidence
RETURN patient_id,
       user_id,
       study_id,
       image_element_id,
       ingest_date,
       ingested_at,
       disease,
       findings,
       confidence
ORDER BY ingest_date DESC, confidence DESC, patient_id, image_element_id
LIMIT $limit
"""

PATIENT_HISTORY_QUERY = """
// List disease evidence for one patient/user across ingested images.
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:SUGGESTS]->(d:Disease)
WHERE (
    toLower(coalesce(i.patient_id, "")) = $patient_id
    OR toLower(coalesce(i.user_id, "")) = $patient_id
    OR toLower(coalesce(i.subject_id, "")) = $patient_id
  )
  AND (size($diseases) = 0 OR toLower(d.canonical_name) IN $diseases)
  AND coalesce(r.confidence, 0.0) >= $min_confidence
  AND ($ingest_date IS NULL OR i.ingest_date = $ingest_date)
WITH
  coalesce(i.patient_id, i.user_id, i.subject_id) AS patient_id,
  i.user_id AS user_id,
  i.study_id AS study_id,
  i.element_id AS image_element_id,
  i.ingest_date AS ingest_date,
  i.ingested_at AS ingested_at,
  d.canonical_name AS disease,
  collect(DISTINCT f.canonical_name) AS findings,
  max(coalesce(r.confidence, 0.0)) AS confidence
RETURN patient_id,
       user_id,
       study_id,
       image_element_id,
       ingest_date,
       ingested_at,
       disease,
       findings,
       confidence
ORDER BY ingest_date DESC, confidence DESC, disease, image_element_id
LIMIT $limit
"""

COHORT_COUNTS_QUERY = """
// Aggregate patient/image counts by disease and optional ingest date.
MATCH (i:ImageElement)-[r:EXHIBITS]->(f:Finding)-[:SUGGESTS]->(d:Disease)
WHERE (size($diseases) = 0 OR toLower(d.canonical_name) IN $diseases)
  AND coalesce(r.confidence, 0.0) >= $min_confidence
  AND ($ingest_date IS NULL OR i.ingest_date = $ingest_date)
WITH
  d.canonical_name AS disease,
  coalesce(i.patient_id, i.user_id, i.subject_id) AS patient_id,
  i.element_id AS image_element_id,
  max(coalesce(r.confidence, 0.0)) AS confidence
RETURN disease,
       count(DISTINCT patient_id) AS patient_count,
       count(DISTINCT image_element_id) AS image_count,
       max(confidence) AS max_confidence
ORDER BY patient_count DESC, image_count DESC, disease
LIMIT $limit
"""


def get_static_backbone_status(client: Neo4jClient) -> Dict[str, int]:
    row = client.run_query(STATIC_BACKBONE_STATUS_QUERY)[0]
    return {
        "suggests_count": int(row.get("suggests_count") or 0),
        "located_at_count": int(row.get("located_at_count") or 0),
    }


def _normalize_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_RETRIEVAL_LIMIT
    if value < 1:
        return 1
    if value > MAX_RETRIEVAL_LIMIT:
        return MAX_RETRIEVAL_LIMIT
    return value

def _normalize_disease_list(diseases: Any) -> List[str]:
    if isinstance(diseases, str):
        values = [diseases]
    else:
        values = list(diseases or [])
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized

def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "unknown"}:
        return None
    return text

def retrieve_patients_by_diseases_on_date(
    client: Neo4jClient,
    diseases: Any,
    ingest_date: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 50,
) -> Dict[str, Any]:
    normalized_diseases = _normalize_disease_list(diseases)
    normalized_date = str(ingest_date).strip() if ingest_date else None
    normalized_limit = _normalize_limit(limit)
    if not normalized_diseases:
        return {
            "query": {
                "diseases": [],
                "ingest_date": normalized_date,
                "min_confidence": float(min_confidence),
                "limit": normalized_limit,
            },
            "patient_count": 0,
            "image_count": 0,
            "rows": [],
        }

    records = client.run_query(
        PATIENTS_BY_DISEASES_ON_DATE_QUERY,
        {
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )
    rows = [
        {
            "patient_id": row.get("patient_id"),
            "user_id": row.get("user_id"),
            "study_id": row.get("study_id"),
            "image_element_id": row.get("image_element_id"),
            "ingest_date": row.get("ingest_date"),
            "ingested_at": row.get("ingested_at"),
            "disease": row.get("disease"),
            "findings": row.get("findings") or [],
            "confidence": float(row.get("confidence") or 0.0),
        }
        for row in records
    ]
    return {
        "query": {
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "patient_count": len({row["patient_id"] for row in rows if row.get("patient_id")}),
        "image_count": len({row["image_element_id"] for row in rows if row.get("image_element_id")}),
        "rows": rows,
    }

def retrieve_patient_history(
    client: Neo4jClient,
    patient_id: str,
    diseases: Any = None,
    ingest_date: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 50,
) -> Dict[str, Any]:
    normalized_patient_id = _normalize_optional_text(patient_id)
    normalized_diseases = _normalize_disease_list(diseases)
    normalized_date = str(ingest_date).strip() if ingest_date else None
    normalized_limit = _normalize_limit(limit)
    if not normalized_patient_id:
        return {
            "query": {
                "patient_id": None,
                "diseases": normalized_diseases,
                "ingest_date": normalized_date,
                "min_confidence": float(min_confidence),
                "limit": normalized_limit,
            },
            "patient_count": 0,
            "image_count": 0,
            "disease_count": 0,
            "rows": [],
        }

    records = client.run_query(
        PATIENT_HISTORY_QUERY,
        {
            "patient_id": normalized_patient_id,
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )
    rows = [
        {
            "patient_id": row.get("patient_id"),
            "user_id": row.get("user_id"),
            "study_id": row.get("study_id"),
            "image_element_id": row.get("image_element_id"),
            "ingest_date": row.get("ingest_date"),
            "ingested_at": row.get("ingested_at"),
            "disease": row.get("disease"),
            "findings": row.get("findings") or [],
            "confidence": float(row.get("confidence") or 0.0),
        }
        for row in records
    ]
    return {
        "query": {
            "patient_id": normalized_patient_id,
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "patient_count": len({row["patient_id"] for row in rows if row.get("patient_id")}),
        "image_count": len({row["image_element_id"] for row in rows if row.get("image_element_id")}),
        "disease_count": len({row["disease"] for row in rows if row.get("disease")}),
        "rows": rows,
    }

def retrieve_cohort_counts(
    client: Neo4jClient,
    diseases: Any = None,
    ingest_date: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 50,
) -> Dict[str, Any]:
    normalized_diseases = _normalize_disease_list(diseases)
    normalized_date = str(ingest_date).strip() if ingest_date else None
    normalized_limit = _normalize_limit(limit)
    records = client.run_query(
        COHORT_COUNTS_QUERY,
        {
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )
    rows = [
        {
            "disease": row.get("disease"),
            "patient_count": int(row.get("patient_count") or 0),
            "image_count": int(row.get("image_count") or 0),
            "max_confidence": float(row.get("max_confidence") or 0.0),
        }
        for row in records
    ]
    return {
        "query": {
            "diseases": normalized_diseases,
            "ingest_date": normalized_date,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "patient_count": sum(row["patient_count"] for row in rows),
        "image_count": sum(row["image_count"] for row in rows),
        "disease_count": len(rows),
        "rows": rows,
    }

def execute_routed_kg_query(
    client: Neo4jClient,
    route: Dict[str, Any],
    min_confidence: float = 0.25,
    limit: int = 50,
) -> Dict[str, Any]:
    intent = str(route.get("intent") or "").strip().lower()
    if intent == "patient_disease_query":
        result = retrieve_patients_by_diseases_on_date(
            client=client,
            diseases=route.get("diseases") or [],
            ingest_date=route.get("ingest_date"),
            min_confidence=min_confidence,
            limit=limit,
        )
    elif intent == "patient_history":
        result = retrieve_patient_history(
            client=client,
            patient_id=str(route.get("patient_id") or route.get("user_id") or ""),
            diseases=route.get("diseases") or [],
            ingest_date=route.get("ingest_date"),
            min_confidence=min_confidence,
            limit=limit,
        )
    elif intent == "cohort_count":
        result = retrieve_cohort_counts(
            client=client,
            diseases=route.get("diseases") or [],
            ingest_date=route.get("ingest_date"),
            min_confidence=min_confidence,
            limit=limit,
        )
    else:
        return {"status": "unsupported_route", "route": route}

    result["intent"] = intent
    result["route"] = route
    return result


def _context_policy(limit: int) -> Dict[str, Any]:
    return {
        "max_evidences": limit,
        "max_candidates": limit,
        "sort": "confidence_desc",
    }


def _summarize_candidates(evidences: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for evidence in evidences:
        value = evidence.get(key)
        if not value:
            continue
        entry = summary.get(value)
        if entry is None:
            summary[value] = {"name": value, "confidence": evidence["confidence"], "count": 1}
        else:
            entry["count"] += 1
            if evidence["confidence"] > entry["confidence"]:
                entry["confidence"] = evidence["confidence"]

    candidates = list(summary.values())
    candidates.sort(key=lambda item: (item["confidence"], item["count"]), reverse=True)
    return candidates[:limit]


def retrieve_existence_context(
    client: Neo4jClient,
    disease: str,
    anatomy: Optional[str] = None,
    study_id: Optional[str] = None,
    image_element_id: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 10,
    require_static_backbone: bool = False,
) -> Dict[str, Any]:
    if require_static_backbone:
        status = get_static_backbone_status(client)
        if status["suggests_count"] <= 0 or status["located_at_count"] <= 0:
            raise RuntimeError(
                "Static ontology backbone is missing. "
                "Require Finding-[:SUGGESTS]->Disease and Finding-[:LOCATED_AT]->Anatomy before existence retrieval."
            )

    normalized_disease = disease.strip().lower()
    normalized_anatomy = anatomy.strip().lower() if anatomy else None
    normalized_study_id = study_id.strip().lower() if study_id else None
    normalized_image_element_id = image_element_id.strip().lower() if image_element_id else None
    normalized_limit = _normalize_limit(limit)

    records = client.run_query(
        EXISTENCE_QUERY,
        {
            "disease": normalized_disease,
            "anatomy": normalized_anatomy,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )

    evidences: List[Dict[str, Any]] = []
    for row in records:
        anatomy_candidates = [x for x in (row.get("anatomy_candidates") or []) if x]
        confidence = float(row.get("confidence") or 0.0)
        logic_path = (
            f"ImageElement({row.get('image_element_id')})"
            f" -[EXHIBITS {confidence:.2f}]-> "
            f"Finding({row.get('finding')})"
            f" -[:SUGGESTS]-> Disease({row.get('disease')})"
        )
        evidences.append(
            {
                "study_id": row.get("study_id"),
                "image_element_id": row.get("image_element_id"),
                "disease": row.get("disease"),
                "finding": row.get("finding"),
                "anatomy_candidates": anatomy_candidates,
                "confidence": confidence,
                "logic_path": logic_path,
                "explanation": (
                    f"Evidence: {row.get('finding')} suggests {row.get('disease')} and "
                    f"{row.get('image_element_id')} exhibits the finding (confidence {confidence:.2f})."
                ),
            }
        )

    top_confidence = max([ev["confidence"] for ev in evidences], default=0.0)
    answer = "yes" if evidences else "insufficient_evidence"

    return {
        "intent": "existence",
        "query": {
            "disease": normalized_disease,
            "anatomy": normalized_anatomy,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "context_policy": _context_policy(normalized_limit),
        "answer": answer,
        "confidence": top_confidence,
        "evidence_count": len(evidences),
        "evidences": evidences,
        "candidates": [],
    }


def retrieve_location_context(
    client: Neo4jClient,
    disease: str,
    study_id: Optional[str] = None,
    image_element_id: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 10,
    require_static_backbone: bool = False,
) -> Dict[str, Any]:
    if require_static_backbone:
        status = get_static_backbone_status(client)
        if status["suggests_count"] <= 0 or status["located_at_count"] <= 0:
            raise RuntimeError(
                "Static ontology backbone is missing. "
                "Require Finding-[:SUGGESTS]->Disease and Finding-[:LOCATED_AT]->Anatomy before location retrieval."
            )

    normalized_disease = disease.strip().lower()
    normalized_study_id = study_id.strip().lower() if study_id else None
    normalized_image_element_id = image_element_id.strip().lower() if image_element_id else None
    normalized_limit = _normalize_limit(limit)

    records = client.run_query(
        LOCATION_QUERY,
        {
            "disease": normalized_disease,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )

    evidences: List[Dict[str, Any]] = []
    for row in records:
        confidence = float(row.get("confidence") or 0.0)
        logic_path = (
            f"ImageElement({row.get('image_element_id')})"
            f" -[EXHIBITS {confidence:.2f}]-> "
            f"Finding({row.get('finding')})"
            f" -[:SUGGESTS]-> Disease({row.get('disease')})"
            f" -[:LOCATED_AT]-> Anatomy({row.get('anatomy')})"
        )
        evidences.append(
            {
                "study_id": row.get("study_id"),
                "image_element_id": row.get("image_element_id"),
                "disease": row.get("disease"),
                "finding": row.get("finding"),
                "anatomy": row.get("anatomy"),
                "confidence": confidence,
                "logic_path": logic_path,
                "explanation": (
                    f"Evidence: {row.get('finding')} suggests {row.get('disease')} and "
                    f"is located at {row.get('anatomy')} (confidence {confidence:.2f})."
                ),
            }
        )

    candidates = _summarize_candidates(evidences, "anatomy", normalized_limit)
    top_confidence = max([ev["confidence"] for ev in evidences], default=0.0)
    answer = candidates[0]["name"] if candidates else "insufficient_evidence"

    return {
        "intent": "location",
        "query": {
            "disease": normalized_disease,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "context_policy": _context_policy(normalized_limit),
        "answer": answer,
        "confidence": top_confidence,
        "evidence_count": len(evidences),
        "evidences": evidences,
        "candidates": candidates,
    }


def retrieve_abnormality_context(
    client: Neo4jClient,
    anatomy: Optional[str] = None,
    study_id: Optional[str] = None,
    image_element_id: Optional[str] = None,
    min_confidence: float = 0.25,
    limit: int = 10,
    require_static_backbone: bool = False,
) -> Dict[str, Any]:
    if require_static_backbone:
        status = get_static_backbone_status(client)
        if status["suggests_count"] <= 0 or status["located_at_count"] <= 0:
            raise RuntimeError(
                "Static ontology backbone is missing. "
                "Require Finding-[:SUGGESTS]->Disease and Finding-[:LOCATED_AT]->Anatomy before abnormality retrieval."
            )

    normalized_anatomy = anatomy.strip().lower() if anatomy else None
    normalized_study_id = study_id.strip().lower() if study_id else None
    normalized_image_element_id = image_element_id.strip().lower() if image_element_id else None
    normalized_limit = _normalize_limit(limit)

    records = client.run_query(
        ABNORMALITY_QUERY,
        {
            "anatomy": normalized_anatomy,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
    )

    evidences: List[Dict[str, Any]] = []
    for row in records:
        confidence = float(row.get("confidence") or 0.0)
        logic_path = (
            f"ImageElement({row.get('image_element_id')})"
            f" -[EXHIBITS {confidence:.2f}]-> "
            f"Finding({row.get('finding')})"
            f" -[:LOCATED_AT]-> Anatomy({row.get('anatomy')})"
            f" -[:SUGGESTS]-> Disease({row.get('disease')})"
        )
        evidences.append(
            {
                "study_id": row.get("study_id"),
                "image_element_id": row.get("image_element_id"),
                "anatomy": row.get("anatomy"),
                "finding": row.get("finding"),
                "disease": row.get("disease"),
                "confidence": confidence,
                "logic_path": logic_path,
                "explanation": (
                    f"Evidence: {row.get('finding')} at {row.get('anatomy')} suggests "
                    f"{row.get('disease')} (confidence {confidence:.2f})."
                ),
            }
        )

    candidates = _summarize_candidates(evidences, "disease", normalized_limit)
    top_confidence = max([ev["confidence"] for ev in evidences], default=0.0)
    answer = candidates[0]["name"] if candidates else "insufficient_evidence"

    return {
        "intent": "abnormality",
        "query": {
            "anatomy": normalized_anatomy,
            "study_id": normalized_study_id,
            "image_element_id": normalized_image_element_id,
            "min_confidence": float(min_confidence),
            "limit": normalized_limit,
        },
        "context_policy": _context_policy(normalized_limit),
        "answer": answer,
        "confidence": top_confidence,
        "evidence_count": len(evidences),
        "evidences": evidences,
        "candidates": candidates,
    }
