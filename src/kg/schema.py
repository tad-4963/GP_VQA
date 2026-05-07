from __future__ import annotations

from typing import Dict, List

from .client import Neo4jClient

CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT anatomy_name_unique IF NOT EXISTS FOR (a:Anatomy) REQUIRE a.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT finding_name_unique IF NOT EXISTS FOR (f:Finding) REQUIRE f.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT image_element_id_unique IF NOT EXISTS FOR (i:ImageElement) REQUIRE i.element_id IS UNIQUE",
    "CREATE CONSTRAINT text_chunk_id_unique IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT term_alias_unique IF NOT EXISTS FOR (t:TermAlias) REQUIRE (t.name, t.namespace) IS UNIQUE",
    "CREATE CONSTRAINT ontology_concept_unique IF NOT EXISTS FOR (o:OntologyConcept) REQUIRE (o.source, o.source_id) IS UNIQUE",
]

INDEX_QUERIES = [
    "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (d:Disease) ON (d.name)",
    "CREATE INDEX anatomy_name_idx IF NOT EXISTS FOR (a:Anatomy) ON (a.name)",
    "CREATE INDEX finding_name_idx IF NOT EXISTS FOR (f:Finding) ON (f.name)",
    "CREATE INDEX image_study_idx IF NOT EXISTS FOR (i:ImageElement) ON (i.study_id)",
    "CREATE INDEX text_study_idx IF NOT EXISTS FOR (t:TextChunk) ON (t.study_id)",
    "CREATE INDEX ontology_name_idx IF NOT EXISTS FOR (o:OntologyConcept) ON (o.name)",
]


def create_schema(client: Neo4jClient) -> Dict[str, List[str]]:
    executed_constraints: List[str] = []
    executed_indexes: List[str] = []

    for query in CONSTRAINT_QUERIES:
        client.execute_write(query)
        executed_constraints.append(query)

    for query in INDEX_QUERIES:
        client.execute_write(query)
        executed_indexes.append(query)

    return {
        "constraints": executed_constraints,
        "indexes": executed_indexes,
    }
