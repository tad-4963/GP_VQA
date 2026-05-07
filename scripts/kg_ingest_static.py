import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg.ingest_static import (
    default_finding_labels,
    ingest_default_cxr_priors,
    ingest_anatomy,
    ingest_diseases,
    ingest_findings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest static ontology (Disease/Anatomy/Finding) into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="MedVQA2026!")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument(
        "--disease-csv",
        default=str(PROJECT_ROOT / "data" / "label" / "normalized_diseases.csv"),
    )
    parser.add_argument(
        "--anatomy-csv",
        default=str(PROJECT_ROOT / "data" / "label" / "normalized_anatomy.csv"),
    )
    parser.add_argument(
        "--findings-json",
        default="",
        help="Optional JSON list of finding labels. If empty, use default_finding_labels().",
    )
    parser.add_argument(
        "--skip-default-priors",
        action="store_true",
        help="Do not create built-in CXR Finding-LOCATED_AT/SUGGESTS edges.",
    )
    return parser.parse_args()


def _load_findings(path: str):
    if not path:
        return default_finding_labels()
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    if not isinstance(values, list):
        raise ValueError("--findings-json must be a JSON array of strings")
    return [str(x) for x in values]


def main() -> int:
    args = parse_args()

    from src.kg.client import Neo4jClient, Neo4jSettings
    from src.kg.schema import create_schema

    settings = Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    findings = _load_findings(args.findings_json)

    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        schema_result = create_schema(client)

        disease_rows = ingest_diseases(client, args.disease_csv)
        anatomy_rows = ingest_anatomy(client, args.anatomy_csv)
        finding_rows = ingest_findings(client, findings)
        prior_rows = 0 if args.skip_default_priors else ingest_default_cxr_priors(client)

        counts_query = """
        CALL {
            MATCH (d:Disease)
            RETURN count(d) AS disease_count
        }
        CALL {
            MATCH (a:Anatomy)
            RETURN count(a) AS anatomy_count
        }
        CALL {
            MATCH (f:Finding)
            RETURN count(f) AS finding_count
        }
        CALL {
            MATCH (t:TermAlias)
            RETURN count(t) AS alias_count
        }
        CALL {
            MATCH (:Finding)-[r:SUGGESTS]->(:Disease)
            RETURN count(r) AS suggests_count
        }
        CALL {
            MATCH (:Finding)-[r:LOCATED_AT]->(:Anatomy)
            RETURN count(r) AS located_at_count
        }
        RETURN disease_count, anatomy_count, finding_count, alias_count, suggests_count, located_at_count
        """
        counts = client.run_query(counts_query)[0]

        duplicate_check = client.run_query(
            """
            MATCH (d:Disease)
            WITH d.canonical_name AS key, count(*) AS c
            WHERE c > 1
            RETURN count(*) AS disease_dup_keys
            """
        )[0]["disease_dup_keys"]

        print(
            "Schema OK:",
            f"constraints={len(schema_result['constraints'])}",
            f"indexes={len(schema_result['indexes'])}",
        )
        print(
            "Ingested rows:",
            f"disease={disease_rows}",
            f"anatomy={anatomy_rows}",
            f"finding={finding_rows}",
            f"default_priors={prior_rows}",
        )
        print(
            "Graph counts:",
            f"Disease={counts['disease_count']}",
            f"Anatomy={counts['anatomy_count']}",
            f"Finding={counts['finding_count']}",
            f"TermAlias={counts['alias_count']}",
            f"SUGGESTS={counts['suggests_count']}",
            f"LOCATED_AT={counts['located_at_count']}",
        )
        print("Duplicate check: disease_dup_keys=", duplicate_check)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
