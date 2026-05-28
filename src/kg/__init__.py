from __future__ import annotations

__all__ = [
    "Neo4jClient",
    "Neo4jSettings",
    "create_schema",
    "ingest_default_cxr_priors",
    "load_dynamic_rows",
    "ingest_dynamic_entities",
    "ingest_umls_aliases",
    "ingest_snomed_hierarchy",
    "ingest_radgraph_priors",
    "retrieve_existence_context",
    "retrieve_location_context",
    "retrieve_abnormality_context",
    "retrieve_patients_by_diseases_on_date",
    "retrieve_patient_history",
    "retrieve_cohort_counts",
    "execute_routed_kg_query",
]


def __getattr__(name: str):
    if name in {"Neo4jClient", "Neo4jSettings"}:
        from .client import Neo4jClient, Neo4jSettings

        return {"Neo4jClient": Neo4jClient, "Neo4jSettings": Neo4jSettings}[name]
    if name == "create_schema":
        from .schema import create_schema

        return create_schema
    if name == "ingest_default_cxr_priors":
        from .ingest_static import ingest_default_cxr_priors

        return ingest_default_cxr_priors
    if name in {"load_dynamic_rows", "ingest_dynamic_entities"}:
        from .ingest_dynamic import ingest_dynamic_entities, load_dynamic_rows

        return {"load_dynamic_rows": load_dynamic_rows, "ingest_dynamic_entities": ingest_dynamic_entities}[name]
    if name in {"ingest_umls_aliases", "ingest_snomed_hierarchy", "ingest_radgraph_priors"}:
        from .ingest_ontology import ingest_radgraph_priors, ingest_snomed_hierarchy, ingest_umls_aliases

        return {
            "ingest_umls_aliases": ingest_umls_aliases,
            "ingest_snomed_hierarchy": ingest_snomed_hierarchy,
            "ingest_radgraph_priors": ingest_radgraph_priors,
        }[name]
    if name in {
        "retrieve_existence_context",
        "retrieve_location_context",
        "retrieve_abnormality_context",
        "retrieve_patients_by_diseases_on_date",
        "retrieve_patient_history",
        "retrieve_cohort_counts",
        "execute_routed_kg_query",
    }:
        from .queries import (
            execute_routed_kg_query,
            retrieve_abnormality_context,
            retrieve_cohort_counts,
            retrieve_existence_context,
            retrieve_location_context,
            retrieve_patient_history,
            retrieve_patients_by_diseases_on_date,
        )

        return {
            "retrieve_existence_context": retrieve_existence_context,
            "retrieve_location_context": retrieve_location_context,
            "retrieve_abnormality_context": retrieve_abnormality_context,
            "retrieve_patients_by_diseases_on_date": retrieve_patients_by_diseases_on_date,
            "retrieve_patient_history": retrieve_patient_history,
            "retrieve_cohort_counts": retrieve_cohort_counts,
            "execute_routed_kg_query": execute_routed_kg_query,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
