from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


@dataclass
class Neo4jSettings:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            username=os.getenv("NEO4J_USERNAME", cls.username),
            password=os.getenv("NEO4J_PASSWORD", cls.password),
            database=os.getenv("NEO4J_DATABASE", cls.database),
        )


class Neo4jClient:
    def __init__(self, settings: Optional[Neo4jSettings] = None):
        self.settings = settings or Neo4jSettings.from_env()
        self.driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.username, self.settings.password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def run_query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        db = database or self.settings.database
        with self.driver.session(database=db) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def execute_write(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        db = database or self.settings.database
        with self.driver.session(database=db) as session:
            result = session.execute_write(
                lambda tx: tx.run(cypher, parameters or {}).data()
            )
        return result

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
