from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .client import Neo4jClient


DEFAULT_DROPPED_FINDINGS = {
    "normal",
    "blunted",
    "clear",
    "feeding tube",
    "icd",
    "missing part",
    "poorly defined",
    "sternal wire",
    "swan-ganz catheter",
    "tracheostomy tube",
    "unfolding",
    "apical capping",
    "hardware failure",
    "peribronchial cuffing",
    "picc",
    "pigtail catheter",
    "widened",
    "chest tube",
    "drain",
    "obscured",
    "silhouette sign",
    "pacemaker",
    "valve prosthesis",
    "distended",
    "endotracheal tube",
    "enteric tube",
    "crowded",
    "artifact",
    "surgical clip",
    "airspace disease",
    "rotated",
    "cvp line",
    "linear band",
}


def default_finding_labels() -> List[str]:
    raw = [
        "consolidation", "pleural effusion", "pneumothorax", "atelectasis", "pulmonary edema", "cardiomegaly", "pneumonia",
        "emphysema", "hernia", "mass", "nodule", "lung opacity", "pleural thickening", "calcification", "granuloma", "fracture",
        "pneumomediastinum", "pneumoperitoneum", "subcutaneous emphysema", "hyperaeration", "cyst / cystic", "scarring / fibrotic",
        "linear band", "infiltration", "vascular congestion", "vascular redistribution", "cavitation", "bronchiectasis", "enlarged",
        "tortuous", "elevated", "blunted", "shifted", "prominent", "abnormal", "obscured", "clear", "distended",
        "collapsed", "widened", "crowded", "rotated", "low lung volumes", "unfolding", "engorgement", "eventration", "lucency",
        "peribronchial cuffing", "airspace disease", "interstitial lung disease", "opacification", "silhouette sign", "apical capping",
        "plural abnormality", "scoliosis", "kyphosis", "degenerative change", "osteopenia", "osteophyte", "arthritic change",
        "surgical material", "hardware failure", "missing part", "artifact", "asymmetry", "poorly defined", "endotracheal tube",
        "enteric tube", "cvp line", "picc", "chest tube", "pacemaker", "icd", "hardware", "sternal wire", "surgical clip",
        "valve prosthesis", "tracheostomy tube", "drain", "pigtail catheter", "swan-ganz catheter", "feeding tube",
    ]
    return [x for x in raw if x not in DEFAULT_DROPPED_FINDINGS]


def default_cxr_static_priors() -> List[Dict[str, str]]:
    """
    Conservative built-in CXR ontology backbone used when RadGraph is unavailable.

    These rows are intentionally broad and task-oriented:
    - LOCATED_AT gives retrieval a stable anatomy bridge.
    - SUGGESTS is only added for disease-like findings where the finding name is
      commonly used as the target abnormality/disease label.
    """
    rows = [
        {"finding": "atelectasis", "anatomy": "lung", "disease": "atelectasis"},
        {"finding": "cardiomegaly", "anatomy": "heart", "disease": "cardiomegaly"},
        {"finding": "consolidation", "anatomy": "lung", "disease": "consolidation"},
        {"finding": "pulmonary edema", "anatomy": "lung", "disease": "pulmonary edema"},
        {"finding": "edema", "anatomy": "lung", "disease": "pulmonary edema"},
        {"finding": "enlarged", "anatomy": "cardiomediastinum", "disease": "enlarged cardiomediastinum"},
        {"finding": "fracture", "anatomy": "rib", "disease": "fracture"},
        {"finding": "lung opacity", "anatomy": "lung", "disease": "lung opacity"},
        {"finding": "opacity", "anatomy": "lung", "disease": "lung opacity"},
        {"finding": "opacification", "anatomy": "lung", "disease": "lung opacity"},
        {"finding": "infiltration", "anatomy": "lung", "disease": "lung opacity"},
        {"finding": "airspace disease", "anatomy": "lung", "disease": "lung opacity"},
        {"finding": "mass", "anatomy": "lung", "disease": "lung lesion"},
        {"finding": "nodule", "anatomy": "lung", "disease": "lung lesion"},
        {"finding": "granuloma", "anatomy": "lung", "disease": "lung lesion"},
        {"finding": "cavitation", "anatomy": "lung", "disease": "lung lesion"},
        {"finding": "pleural effusion", "anatomy": "pleura", "disease": "pleural effusion"},
        {"finding": "pleural thickening", "anatomy": "pleura", "disease": "pleural abnormality"},
        {"finding": "plural abnormality", "anatomy": "pleura", "disease": "pleural abnormality"},
        {"finding": "pneumonia", "anatomy": "lung", "disease": "pneumonia"},
        {"finding": "pneumonia", "anatomy": "lung", "disease": "infection"},
        {"finding": "consolidation", "anatomy": "lung", "disease": "infection"},
        {"finding": "lung opacity", "anatomy": "lung", "disease": "infection"},
        {"finding": "opacity", "anatomy": "lung", "disease": "infection"},
        {"finding": "pulmonary edema", "anatomy": "lung", "disease": "interstitial edema"},
        {"finding": "edema", "anatomy": "lung", "disease": "interstitial edema"},
        {"finding": "mass", "anatomy": "lung", "disease": "metastatic disease"},
        {"finding": "nodule", "anatomy": "lung", "disease": "metastatic disease"},
        {"finding": "pneumothorax", "anatomy": "pleura", "disease": "pneumothorax"},
        {"finding": "pneumomediastinum", "anatomy": "mediastinum", "disease": "pneumomediastinum"},
        {"finding": "pneumoperitoneum", "anatomy": "abdomen", "disease": "pneumoperitoneum"},
        {"finding": "subcutaneous emphysema", "anatomy": "chest wall", "disease": "subcutaneous emphysema"},
        {"finding": "emphysema", "anatomy": "lung", "disease": "emphysema"},
        {"finding": "hyperaeration", "anatomy": "lung", "disease": "emphysema"},
        {"finding": "hernia", "anatomy": "diaphragm", "disease": "hernia"},
        {"finding": "scarring / fibrotic", "anatomy": "lung", "disease": "fibrosis"},
        {"finding": "interstitial lung disease", "anatomy": "lung", "disease": "interstitial lung disease"},
        {"finding": "bronchiectasis", "anatomy": "lung", "disease": "bronchiectasis"},
        {"finding": "vascular congestion", "anatomy": "lung", "disease": "pulmonary edema"},
        {"finding": "vascular redistribution", "anatomy": "lung", "disease": "pulmonary edema"},
        {"finding": "low lung volumes", "anatomy": "lung", "disease": "low lung volumes"},
        {"finding": "collapsed", "anatomy": "lung", "disease": "atelectasis"},
        {"finding": "elevated", "anatomy": "diaphragm", "disease": ""},
        {"finding": "eventration", "anatomy": "diaphragm", "disease": ""},
        {"finding": "scoliosis", "anatomy": "spine", "disease": "scoliosis"},
        {"finding": "kyphosis", "anatomy": "spine", "disease": "kyphosis"},
        {"finding": "degenerative change", "anatomy": "spine", "disease": "degenerative change"},
        {"finding": "osteopenia", "anatomy": "bone", "disease": "osteopenia"},
        {"finding": "osteophyte", "anatomy": "spine", "disease": "degenerative change"},
        {"finding": "arthritic change", "anatomy": "bone", "disease": "degenerative change"},
        {"finding": "calcification", "anatomy": "chest", "disease": "calcification"},
        {"finding": "lucency", "anatomy": "lung", "disease": ""},
        {"finding": "asymmetry", "anatomy": "chest", "disease": ""},
        {"finding": "abnormal", "anatomy": "chest", "disease": "abnormality"},
    ]
    return rows


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def ingest_diseases(client: Neo4jClient, csv_path: str) -> int:
    rows = _load_rows(Path(csv_path))
    payload = []
    for row in rows:
        mapped = (row.get("Mapped_Disease") or "").strip().lower()
        if not mapped or mapped == "unknown":
            continue
        payload.append(
            {
                "canonical_name": mapped,
                "name": mapped,
                "raw_term": (row.get("Raw_Term") or "").strip().lower(),
                "source": "normalized_diseases.csv",
            }
        )

    query = """
    UNWIND $rows AS row
    MERGE (d:Disease {canonical_name: row.canonical_name})
    SET d.name = row.name,
        d.source = row.source
    WITH d, row
    WHERE row.raw_term <> ''
    MERGE (alias:TermAlias {name: row.raw_term, namespace: 'disease'})
    MERGE (alias)-[:ALIAS_OF]->(d)
    """
    client.execute_write(query, {"rows": payload})
    return len(payload)


def ingest_anatomy(client: Neo4jClient, csv_path: str) -> int:
    rows = _load_rows(Path(csv_path))
    payload = []
    for row in rows:
        mapped = (row.get("Mapped_Anatomy") or "").strip().lower()
        if not mapped or mapped == "unknown":
            continue
        payload.append(
            {
                "canonical_name": mapped,
                "name": mapped,
                "raw_term": (row.get("Raw_Term") or "").strip().lower(),
                "source": "normalized_anatomy.csv",
            }
        )

    query = """
    UNWIND $rows AS row
    MERGE (a:Anatomy {canonical_name: row.canonical_name})
    SET a.name = row.name,
        a.source = row.source
    WITH a, row
    WHERE row.raw_term <> ''
    MERGE (alias:TermAlias {name: row.raw_term, namespace: 'anatomy'})
    MERGE (alias)-[:ALIAS_OF]->(a)
    """
    client.execute_write(query, {"rows": payload})
    return len(payload)


def ingest_findings(client: Neo4jClient, findings: List[str]) -> int:
    payload = []
    seen = set()
    for name in findings:
        canonical = name.strip().lower()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        payload.append({"canonical_name": canonical, "name": canonical})

    query = """
    UNWIND $rows AS row
    MERGE (f:Finding {canonical_name: row.canonical_name})
    SET f.name = row.name
    """
    client.execute_write(query, {"rows": payload})
    return len(payload)


def ingest_default_cxr_priors(client: "Neo4jClient", priors: List[Dict[str, str]] | None = None) -> int:
    rows = priors if priors is not None else default_cxr_static_priors()
    payload = []
    seen = set()
    for row in rows:
        finding = (row.get("finding") or "").strip().lower()
        anatomy = (row.get("anatomy") or "").strip().lower()
        disease = (row.get("disease") or "").strip().lower()
        key = (finding, anatomy, disease)
        if not finding or key in seen:
            continue
        seen.add(key)
        payload.append({"finding": finding, "anatomy": anatomy, "disease": disease})

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
        SET loc.source = 'default_cxr_static_priors'
    )

    FOREACH (_ IN CASE WHEN row.disease = '' THEN [] ELSE [1] END |
        MERGE (d:Disease {canonical_name: row.disease})
        ON CREATE SET d.name = row.disease
        MERGE (f)-[s:SUGGESTS]->(d)
        SET s.source = 'default_cxr_static_priors'
    )
    """
    client.execute_write(query, {"rows": payload})
    return len(payload)
