import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg.client import Neo4jClient, Neo4jSettings
from src.kg.ingest_dynamic import (
    ingest_dynamic_entities,
    load_dynamic_rows,
    load_vision_prediction_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest dynamic EXHIBITS edges from entity JSON. "
            "Static SUGGESTS/LOCATED_AT edges should come from static/default priors or optional ontology ingest."
        )
    )
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="MedVQA2026!")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument(
        "--json-path",
        default=str(PROJECT_ROOT / "data" / "medical_cxr" / "filtered_all_diseases.json"),
    )
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--positive-confidence", type=float, default=0.85)
    parser.add_argument(
        "--vision-predictions",
        default="",
        help=(
            "Optional CSV/JSON/JSONL from real vision inference. "
            "Rows must include study_id, dicom_id/element_id, finding_name and confidence."
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--source", default="vision_model")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    settings = Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if args.vision_predictions:
        rows = load_vision_prediction_rows(
            prediction_path=args.vision_predictions,
            min_confidence=args.min_confidence,
            max_records=args.max_records,
            source=args.source,
            model_id=args.model_id,
            checkpoint_path=args.checkpoint_path,
            threshold=args.min_confidence,
            run_id=args.run_id,
        )
    else:
        rows = load_dynamic_rows(
            json_path=args.json_path,
            max_records=args.max_records,
            positive_confidence=args.positive_confidence,
        )

    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        ingested = ingest_dynamic_entities(client, rows, batch_size=args.batch_size)

        counts = client.run_query(
            """
            CALL {
              MATCH (:ImageElement)-[r:EXHIBITS]->(:Finding)
              RETURN count(r) AS exhibits_count
            }
            CALL {
              MATCH (:Finding)-[r:SUGGESTS]->(:Disease)
              RETURN count(r) AS suggests_count
            }
            CALL {
              MATCH (:Finding)-[r:LOCATED_AT]->(:Anatomy)
              RETURN count(r) AS located_at_count
            }
            RETURN exhibits_count, suggests_count, located_at_count
            """
        )[0]

    print(
        "Dynamic ingest done:",
        f"dynamic_rows={ingested}",
        f"EXHIBITS_total={counts['exhibits_count']}",
        f"SUGGESTS_static_backbone_total={counts['suggests_count']}",
        f"LOCATED_AT_static_backbone_total={counts['located_at_count']}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
