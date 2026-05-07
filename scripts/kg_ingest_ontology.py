import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest optional ontology priors from UMLS/SNOMED/RadGraph")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="MedVQA2026!")
    parser.add_argument("--database", default="neo4j")

    parser.add_argument("--umls-csv", default="", help="CSV with UMLS aliases")
    parser.add_argument("--snomed-rel-csv", default="", help="CSV with SNOMED relations")
    parser.add_argument(
        "--radgraph-priors",
        default="",
        help="Optional CSV/JSON with RadGraph-style priors. Not required if default static CXR priors were ingested.",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per Neo4j transaction for UMLS/SNOMED ingest")
    parser.add_argument("--skip-schema", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.kg.client import Neo4jClient, Neo4jSettings
    from src.kg.ingest_ontology import ingest_radgraph_priors, ingest_snomed_hierarchy, ingest_umls_aliases
    from src.kg.schema import create_schema

    settings = Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        if not args.skip_schema:
            create_schema(client)

        umls_count = ingest_umls_aliases(client, args.umls_csv, batch_size=args.batch_size) if args.umls_csv else 0
        snomed_count = (
            ingest_snomed_hierarchy(client, args.snomed_rel_csv, batch_size=args.batch_size)
            if args.snomed_rel_csv
            else 0
        )
        radgraph_count = ingest_radgraph_priors(client, args.radgraph_priors) if args.radgraph_priors else 0

        rel_types = client.run_query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels"
        )[0].get("rels") or []

        ontology_concepts = client.run_query(
            "MATCH (oc:OntologyConcept) RETURN count(oc) AS ontology_concepts"
        )[0].get("ontology_concepts")

        if "ONTOLOGY_REL" in rel_types:
            ontology_edges = client.run_query(
                "MATCH (:OntologyConcept)-[r:ONTOLOGY_REL]->(:OntologyConcept) RETURN count(r) AS ontology_edges"
            )[0].get("ontology_edges")
        else:
            ontology_edges = 0

        if "ALIGNS_TO" in rel_types:
            align_edges = client.run_query(
                "MATCH (:OntologyConcept)-[r:ALIGNS_TO]->() RETURN count(r) AS align_edges"
            )[0].get("align_edges")
        else:
            align_edges = 0

    print(
        "Ontology ingest done:",
        f"umls_rows={umls_count}",
        f"snomed_rows={snomed_count}",
        f"radgraph_rows={radgraph_count}",
        f"OntologyConcept={ontology_concepts}",
        f"ONTOLOGY_REL={ontology_edges}",
        f"ALIGNS_TO={align_edges}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
