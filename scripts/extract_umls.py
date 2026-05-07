from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_SOURCES = ("SNOMEDCT_US", "RADLEX")
DEFAULT_OUTPUT_DIR = Path("data/label")

MRCONSO_COLUMNS = {
    "cui": 0,
    "lat": 1,
    "ispref": 6,
    "tty": 12,
    "code": 13,
    "str": 14,
    "suppress": 16,
}

MRREL_COLUMNS = {
    "cui1": 0,
    "rel": 3,
    "cui2": 4,
    "rela": 7,
    "sab": 10,
    "suppress": 14,
}

DISEASE_STYS = {
    "T019",  # Congenital Abnormality
    "T020",  # Acquired Abnormality
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T049",  # Cell or Molecular Dysfunction
    "T050",  # Experimental Model of Disease
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
}
ANATOMY_STYS = {
    "T017",  # Anatomical Structure
    "T021",  # Fully Formed Anatomical Structure
    "T022",  # Body System
    "T023",  # Body Part, Organ, or Organ Component
    "T024",  # Tissue
    "T025",  # Cell
    "T026",  # Cell Component
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
}
FINDING_STYS = {
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T184",  # Sign or Symptom
}

CHEST_KEYWORDS = {
    "airspace",
    "atelectasis",
    "cardiac",
    "cardiomediastinal",
    "cardiomegaly",
    "chest",
    "consolidation",
    "costophrenic",
    "edema",
    "effusion",
    "enlarged cardiomediastinum",
    "fracture",
    "heart",
    "hemithorax",
    "hilar",
    "hyperinflation",
    "infiltrate",
    "lung",
    "mediastinal",
    "mediastinum",
    "nodule",
    "opacity",
    "pleura",
    "pleural",
    "pneumonia",
    "pneumothorax",
    "pulmonary",
    "rib",
    "thoracic",
}

ANATOMY_HINTS = {
    "airway",
    "apex",
    "base",
    "bronchus",
    "cardiomediastinal",
    "chest",
    "clavicle",
    "costophrenic",
    "diaphragm",
    "heart",
    "hemidiaphragm",
    "hemithorax",
    "hilum",
    "hilar",
    "lung",
    "mediastinum",
    "pleura",
    "rib",
    "thorax",
    "trachea",
}

FINDING_HINTS = {
    "abnormality",
    "atelectasis",
    "consolidation",
    "edema",
    "effusion",
    "finding",
    "infiltrate",
    "lesion",
    "nodule",
    "opacity",
    "pain",
    "pneumonia",
    "pneumothorax",
}

PROCEDURE_HINTS = {
    "angiography",
    "biopsy",
    "ct",
    "mri",
    "procedure",
    "radiograph",
    "radiography",
    "scan",
    "ultrasound",
    "xray",
    "x-ray",
}


def _split_sources(value: str) -> Set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _clean(value: str) -> str:
    return " ".join((value or "").strip().split())


def _lower(value: str) -> str:
    return _clean(value).lower()


def _tokens(value: str) -> Set[str]:
    normalized = _lower(value)
    for char in "-/(),;:":
        normalized = normalized.replace(char, " ")
    return set(normalized.split())


def _contains_phrase_or_token(term: str, keywords: Set[str]) -> bool:
    normalized = _lower(term)
    tokens = _tokens(term)
    for keyword in keywords:
        if " " in keyword:
            if keyword in normalized:
                return True
        elif keyword in tokens:
            return True
    return False


def _canonical_score(term: str, is_preferred: bool) -> Tuple[int, int, int, int, str]:
    normalized = _lower(term)
    noisy = int(
        " nos" in normalized
        or normalized.endswith(" nos")
        or normalized.endswith(" (qualifier value)")
        or normalized.endswith(" (attribute)")
    )
    parenthetical = int("(" in normalized or ")" in normalized)
    preferred = 0 if is_preferred else 1
    return noisy, parenthetical, preferred, len(normalized), normalized


def _validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def load_semantic_types(mrsty_path: Optional[Path]) -> Dict[str, Set[str]]:
    if not mrsty_path:
        return {}
    _validate_file(mrsty_path, "MRSTY.RRF")

    semantic_types: Dict[str, Set[str]] = {}
    with mrsty_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            cui = parts[0].strip()
            tui = parts[1].strip()
            if cui and tui:
                semantic_types.setdefault(cui, set()).add(tui)
    return semantic_types


def infer_namespace(term: str, semantic_types: Iterable[str]) -> str:
    sty_set = set(semantic_types)
    if sty_set & ANATOMY_STYS:
        return "anatomy"
    if sty_set & DISEASE_STYS:
        return "disease"
    if sty_set & FINDING_STYS:
        return "finding"

    normalized = _lower(term)
    tokens = _tokens(term)
    if tokens & {"disease", "disorder", "syndrome", "pneumonia", "edema"} or normalized.endswith(
        (" (disorder)", " (disease)", " (syndrome)")
    ):
        return "disease"
    if tokens & FINDING_HINTS or normalized.endswith(" (finding)"):
        return "finding"
    if tokens & ANATOMY_HINTS:
        return "anatomy"
    return "finding"


def is_chest_relevant(term: str, source: str, semantic_types: Iterable[str], keep_all_radlex: bool) -> bool:
    if keep_all_radlex and source.upper() == "RADLEX":
        return True

    if _tokens(term) & PROCEDURE_HINTS:
        return False

    if _contains_phrase_or_token(term, CHEST_KEYWORDS):
        return True

    sty_set = set(semantic_types)
    return bool(sty_set & (ANATOMY_STYS | DISEASE_STYS | FINDING_STYS)) and _contains_phrase_or_token(
        term, CHEST_KEYWORDS
    )


def extract_umls_aliases(
    mrconso_path: Path,
    ontology_csv: Path,
    sources: Set[str],
    semantic_types: Dict[str, Set[str]],
    keep_all_radlex: bool,
    max_rows: int,
) -> Tuple[int, Set[str]]:
    selected_rows: List[Dict[str, str]] = []
    preferred_by_cui: Dict[str, Tuple[Tuple[int, int, int, int, str], str]] = {}
    seen = set()

    with mrconso_path.open("r", encoding="utf-8", errors="replace") as source_handle:
        for line in source_handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= MRCONSO_COLUMNS["suppress"]:
                continue

            cui = parts[MRCONSO_COLUMNS["cui"]].strip()
            lang = parts[MRCONSO_COLUMNS["lat"]].strip()
            is_preferred = parts[MRCONSO_COLUMNS["ispref"]].strip() == "Y"
            source = parts[11].strip().upper()
            term = _clean(parts[MRCONSO_COLUMNS["str"]])
            suppress = parts[MRCONSO_COLUMNS["suppress"]].strip()

            if lang != "ENG" or source not in sources or suppress == "Y" or not cui or not term:
                continue

            sty_set = semantic_types.get(cui, set())
            if not is_chest_relevant(term, source, sty_set, keep_all_radlex):
                continue

            namespace = infer_namespace(term, sty_set)
            semantic_type = ";".join(sorted(sty_set))
            key = (cui, _lower(term), namespace, source)
            if key in seen:
                continue
            seen.add(key)

            score = _canonical_score(term, is_preferred)
            if cui not in preferred_by_cui or score < preferred_by_cui[cui][0]:
                preferred_by_cui[cui] = (score, term)
            selected_rows.append(
                {
                    "cui": cui,
                    "alias": term,
                    "namespace": namespace,
                    "semantic_type": semantic_type,
                    "source": source,
                }
            )
            if max_rows and len(selected_rows) >= max_rows:
                break

    with ontology_csv.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(["cui", "alias_term", "canonical_name", "namespace", "semantic_type", "source"])
        for row in selected_rows:
            writer.writerow(
                [
                    row["cui"],
                    row["alias"],
                    preferred_by_cui.get(row["cui"], ((0, 0, 0, 0, ""), row["alias"]))[1],
                    row["namespace"],
                    row["semantic_type"],
                    row["source"],
                ]
            )

    return len(selected_rows), set(preferred_by_cui)


def extract_snomed_relations(mrrel_path: Path, snomed_csv: Path, allowed_cuis: Set[str], max_rows: int) -> int:
    written = 0
    seen = set()

    with mrrel_path.open("r", encoding="utf-8", errors="replace") as source_handle, snomed_csv.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(["source_id", "target_id", "rel_type", "source"])

        for line in source_handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= MRREL_COLUMNS["suppress"]:
                continue

            cui1 = parts[MRREL_COLUMNS["cui1"]].strip()
            rel = parts[MRREL_COLUMNS["rel"]].strip()
            cui2 = parts[MRREL_COLUMNS["cui2"]].strip()
            rela = parts[MRREL_COLUMNS["rela"]].strip()
            source = parts[MRREL_COLUMNS["sab"]].strip().upper()
            suppress = parts[MRREL_COLUMNS["suppress"]].strip()

            if source != "SNOMEDCT_US" or suppress == "Y" or not cui1 or not cui2 or cui1 == cui2:
                continue
            if allowed_cuis and (cui1 not in allowed_cuis or cui2 not in allowed_cuis):
                continue

            rel_type = rela or rel or "related_to"
            key = (cui2, cui1, rel_type)
            if key in seen:
                continue
            seen.add(key)

            writer.writerow([cui2, cui1, rel_type, source])
            written += 1
            if max_rows and written >= max_rows:
                break

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UMLS MRCONSO/MRREL files into CSVs accepted by scripts/kg_ingest_ontology.py"
    )
    parser.add_argument("--mrconso", default="data/label/MRCONSO.RRF", help="Path to UMLS META/MRCONSO.RRF")
    parser.add_argument("--mrrel", default="data/label/MRREL.RRF", help="Path to UMLS META/MRREL.RRF")
    parser.add_argument("--mrsty", default="", help="Optional path to UMLS META/MRSTY.RRF")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated UMLS SAB sources to keep, for example SNOMEDCT_US,RADLEX",
    )
    parser.add_argument("--max-alias-rows", type=int, default=0, help="Debug limit; 0 means no limit")
    parser.add_argument("--max-rel-rows", type=int, default=0, help="Debug limit; 0 means no limit")
    parser.add_argument("--keep-all-radlex", action="store_true", help="Keep every English RADLEX row")
    parser.add_argument("--skip-relations", action="store_true", help="Only write ontology aliases; skip MRREL scan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mrconso_path = Path(args.mrconso)
    mrrel_path = Path(args.mrrel)
    mrsty_path = Path(args.mrsty) if args.mrsty else None
    output_dir = Path(args.output_dir)
    ontology_csv = output_dir / "ontology_ingest_template.csv"
    snomed_csv = output_dir / "snomed_rel_template.csv"

    _validate_file(mrconso_path, "MRCONSO.RRF")
    _validate_file(mrrel_path, "MRREL.RRF")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _split_sources(args.sources)
    semantic_types = load_semantic_types(mrsty_path)

    print(f"Reading MRCONSO: {mrconso_path}")
    alias_count, alias_cuis = extract_umls_aliases(
        mrconso_path=mrconso_path,
        ontology_csv=ontology_csv,
        sources=sources,
        semantic_types=semantic_types,
        keep_all_radlex=args.keep_all_radlex,
        max_rows=args.max_alias_rows,
    )
    print(f"Wrote aliases: {ontology_csv} ({alias_count} rows)")

    if args.skip_relations:
        with snomed_csv.open("w", newline="", encoding="utf-8") as output_handle:
            csv.writer(output_handle).writerow(["source_id", "target_id", "rel_type", "source"])
        rel_count = 0
        print(f"Skipped MRREL scan; wrote empty SNOMED relation CSV: {snomed_csv}")
    else:
        print(f"Reading MRREL: {mrrel_path}")
        rel_count = extract_snomed_relations(
            mrrel_path=mrrel_path,
            snomed_csv=snomed_csv,
            allowed_cuis=alias_cuis,
            max_rows=args.max_rel_rows,
        )
        print(f"Wrote SNOMED relations: {snomed_csv} ({rel_count} rows)")

    print(
        "Next:",
        "python scripts/kg_ingest_ontology.py",
        f"--umls-csv {ontology_csv}",
        f"--snomed-rel-csv {snomed_csv}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
