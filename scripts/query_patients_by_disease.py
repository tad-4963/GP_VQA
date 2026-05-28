import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg.client import Neo4jClient, Neo4jSettings
from src.kg.queries import execute_routed_kg_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query patients/images by KG-inferred disease and ingest date.")
    parser.add_argument("--question", default="", help="Natural-language query. Parsed with the OpenAI router.")
    parser.add_argument("--disease", action="append", default=[], help="Disease name. Repeat for multiple diseases.")
    parser.add_argument("--patient-id", default="", help="Patient/user id for patient history queries.")
    parser.add_argument("--count", action="store_true", help="Return aggregate cohort counts instead of patient rows.")
    parser.add_argument("--date", default=None, help="Image ingest date in YYYY-MM-DD. Omit to search all dates.")
    parser.add_argument("--current-date", default=None, help="YYYY-MM-DD anchor for relative dates in --question.")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="MedVQA2026!")
    parser.add_argument("--database", default="neo4j")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diseases = args.disease
    ingest_date = args.date
    route = None

    if args.question:
        from src.models.language.llm_answer_generator import OpenAICompatibleQuestionRouter

        router = OpenAICompatibleQuestionRouter.from_env()
        route = router.route(args.question, current_date=args.current_date)
    else:
        if args.count:
            route = {
                "intent": "cohort_count",
                "diseases": diseases,
                "ingest_date": ingest_date,
            }
        elif args.patient_id:
            route = {
                "intent": "patient_history",
                "patient_id": args.patient_id,
                "diseases": diseases,
                "ingest_date": ingest_date,
            }
        else:
            if not diseases:
                raise SystemExit("Provide --disease, --patient-id, --count, or --question for the OpenAI router.")
            route = {
                "intent": "patient_disease_query",
                "diseases": diseases,
                "ingest_date": ingest_date,
            }

    settings = Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )
    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        result = execute_routed_kg_query(
            client=client,
            route=route,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
