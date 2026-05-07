from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

if TYPE_CHECKING:
    from .client import Neo4jClient

DEFAULT_BATCH_SIZE = 1000


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json_rows(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {file_path}")
    return [x for x in data if isinstance(x, dict)]


def _iter_batches(rows: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[List[Dict[str, Any]]]:
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(rows), safe_batch_size):
        yield rows[start : start + safe_batch_size]


def _merge_alias_rows(
    client: Neo4jClient,
    rows: List[Dict[str, str]],
    namespace: str,
    target_label: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    if not rows:
        return 0

    query = f"""
    UNWIND $rows AS row
    MERGE (oc:OntologyConcept {{source: row.source, source_id: row.source_id}})
    SET oc.name = row.preferred_name,
        oc.namespace = row.namespace,
        oc.semantic_type = row.semantic_type,
        oc.source = row.source

    WITH oc, row
    WHERE row.alias <> ''
    MERGE (alias:TermAlias {{name: row.alias, namespace: row.namespace}})
    MERGE (alias)-[:ALIAS_OF]->(oc)

    WITH oc, row
    WHERE row.canonical_name <> ''
    MERGE (n:{target_label} {{canonical_name: row.canonical_name}})
    ON CREATE SET n.name = row.canonical_name
    MERGE (oc)-[:ALIGNS_TO]->(n)
    """
    for batch in _iter_batches(rows, batch_size):
        client.execute_write(query, {"rows": batch})
    return len(rows)


def ingest_umls_aliases(client: Neo4jClient, csv_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Expected columns:
    - cui
    - alias_term
    - canonical_name
    - namespace: disease|anatomy|finding
    - semantic_type (optional)
    - source (optional, default umls)
    """
    raw_rows = _read_csv_rows(csv_path)

    grouped: Dict[str, List[Dict[str, str]]] = {
        "disease": [],
        "anatomy": [],
        "finding": [],
    }
    for row in raw_rows:
        namespace = _clean(row.get("namespace"))
        if namespace not in grouped:
            continue

        source_id = _clean(row.get("cui"))
        alias = _clean(row.get("alias_term"))
        canonical_name = _clean(row.get("canonical_name"))
        if not source_id or (not alias and not canonical_name):
            continue

        grouped[namespace].append(
            {
                "source": _clean(row.get("source")) or "umls",
                "source_id": source_id,
                "preferred_name": canonical_name or alias,
                "alias": alias,
                "canonical_name": canonical_name,
                "namespace": namespace,
                "semantic_type": _clean(row.get("semantic_type")),
            }
        )

    total = 0
    total += _merge_alias_rows(client, grouped["disease"], "disease", "Disease", batch_size=batch_size)
    total += _merge_alias_rows(client, grouped["anatomy"], "anatomy", "Anatomy", batch_size=batch_size)
    total += _merge_alias_rows(client, grouped["finding"], "finding", "Finding", batch_size=batch_size)
    return total


def ingest_snomed_hierarchy(client: Neo4jClient, csv_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Expected columns:
    - source_id
    - target_id
    - rel_type (example: IS_A, PART_OF, ASSOCIATED_WITH)
    - source (optional, default snomedct_us)
    """
    rows = _read_csv_rows(csv_path)
    payload = []
    for row in rows:
        source_id = _clean(row.get("source_id"))
        target_id = _clean(row.get("target_id"))
        rel_type = _clean(row.get("rel_type")) or "is_a"
        source = _clean(row.get("source")) or "snomedct_us"
        if not source_id or not target_id:
            continue
        payload.append(
            {
                "source_name": source,
                "source": source_id,
                "target": target_id,
                "rel_type": rel_type,
            }
        )

    if not payload:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (s:OntologyConcept {source: row.source_name, source_id: row.source})
    ON CREATE SET s.name = row.source
    MERGE (t:OntologyConcept {source: row.source_name, source_id: row.target})
    ON CREATE SET t.name = row.target
    MERGE (s)-[r:ONTOLOGY_REL {type: row.rel_type}]->(t)
    SET r.source = row.source_name
    """
    for batch in _iter_batches(payload, batch_size):
        client.execute_write(query, {"rows": batch})
    return len(payload)


def ingest_radgraph_priors(client: Neo4jClient, path: str) -> int:
    """
    Expected CSV or JSON rows with keys:
    - finding
    - anatomy
    - disease (optional)
    - confidence (optional)

    Note:
    - SUGGESTS/LOCATED_AT created here are static ontology backbone edges.
    - Runtime confidence must come from dynamic EXHIBITS edges.
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        rows = _read_json_rows(path)
    else:
        rows = _read_csv_rows(path)

    payload = []
    for row in rows:
        finding = _clean(row.get("finding"))
        anatomy = _clean(row.get("anatomy"))
        disease = _clean(row.get("disease"))
        confidence_text = _clean(row.get("confidence"))
        try:
            confidence = float(confidence_text) if confidence_text else 0.7
        except ValueError:
            confidence = 0.7

        if not finding:
            continue

        payload.append(
            {
                "finding": finding,
                "anatomy": anatomy,
                "disease": disease,
                "confidence": confidence,
            }
        )

    if not payload:
        return 0

    query = """
    UNWIND $rows AS row
    MERGE (f:Finding {canonical_name: row.finding})
    ON CREATE SET f.name = row.finding

    FOREACH (_ IN CASE WHEN row.anatomy = '' THEN [] ELSE [1] END |
        MERGE (a:Anatomy {canonical_name: row.anatomy})
        ON CREATE SET a.name = row.anatomy
        MERGE (f)-[loc:LOCATED_AT]->(a)
        SET loc.source = 'static_ontology'
    )

    FOREACH (_ IN CASE WHEN row.disease = '' THEN [] ELSE [1] END |
        MERGE (d:Disease {canonical_name: row.disease})
        ON CREATE SET d.name = row.disease
        MERGE (f)-[s:SUGGESTS]->(d)
        SET s.source = 'static_ontology'
    )
    """
    client.execute_write(query, {"rows": payload})
    return len(payload)
