from __future__ import annotations

import re
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.kg.queries import (
    retrieve_abnormality_context,
    retrieve_existence_context,
    retrieve_location_context,
)


SUPPORTED_INTENTS = {"existence", "location", "abnormality"}


@dataclass
class QuestionUnderstanding:
    question: str
    intent: str
    diseases: List[str]
    anatomies: List[str]


@dataclass
class VQARequest:
    question: str
    study_id: Optional[str] = None
    image_element_id: Optional[str] = None
    image_path: Optional[str] = None
    patient_id: Optional[str] = None
    user_id: Optional[str] = None


class RuleBasedQuestionParser:
    """Lightweight parser used before the trained intent/NER model is wired in."""

    def __init__(
        self,
        diseases_csv_path: str = "data/label/normalized_diseases.csv",
        anatomy_csv_path: str = "data/label/normalized_anatomy.csv",
    ):
        self.entity_extractor = None
        disease_path = Path(diseases_csv_path)
        anatomy_path = Path(anatomy_csv_path)
        if disease_path.exists() and anatomy_path.exists():
            from src.models.language.clinical_ner import ClinicalEntityExtractor

            self.entity_extractor = ClinicalEntityExtractor(
                diseases_csv_path=str(disease_path),
                anatomy_csv_path=str(anatomy_path),
            )

    def parse(self, question: str) -> QuestionUnderstanding:
        text = self._clean(question)
        entities = self._extract_entities(text)
        return QuestionUnderstanding(
            question=question,
            intent=self._infer_intent(text, entities),
            diseases=entities["DISEASE"],
            anatomies=entities["ANATOMY"],
        )

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {"DISEASE": [], "ANATOMY": []}
        if self.entity_extractor is not None:
            grouped = self.entity_extractor.extract_grouped(text)

        # Fallbacks for common CXR terms that should be recognized even if the
        # normalized dictionaries are incomplete for a demo question.
        disease_terms = [
            "pneumonia",
            "pleural effusion",
            "pneumothorax",
            "edema",
            "atelectasis",
            "consolidation",
            "cardiomegaly",
            "lung opacity",
            "opacity",
        ]
        anatomy_terms = [
            "right lung",
            "left lung",
            "lung",
            "lungs",
            "right lower lung",
            "left lower lung",
            "right upper lung",
            "left upper lung",
            "mediastinum",
            "cardiac silhouette",
        ]

        for term in disease_terms:
            self._append_if_present(grouped["DISEASE"], text, term)
        for term in anatomy_terms:
            canonical = "lung" if term == "lungs" else term
            self._append_if_present(grouped["ANATOMY"], text, term, canonical)

        return grouped

    def _infer_intent(self, text: str, entities: Dict[str, List[str]]) -> str:
        if re.search(r"\b(where|location|located|which side|which lung)\b", text):
            return "location"
        if re.search(r"\b(what|which)\b.*\b(abnormalit|finding|disease|condition)", text):
            return "abnormality"
        if re.search(r"\b(abnormalit|finding)s?\b.*\b(seen|present|there)\b", text):
            return "abnormality"
        if entities["DISEASE"]:
            return "existence"
        return "abnormality"

    def _append_if_present(
        self,
        values: List[str],
        text: str,
        term: str,
        canonical: Optional[str] = None,
    ) -> None:
        value = canonical or term
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) and value not in values:
            values.append(value)

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())


class OpenAIQuestionParser:
    """Adapter that lets the OpenAI router replace the local rule-based parser."""

    INTENT_MAP = {
        "image_abnormality": "abnormality",
        "image_disease_existence": "existence",
        "image_disease_location": "location",
    }

    def __init__(self, router: Any):
        self.router = router

    @classmethod
    def from_env(cls) -> "OpenAIQuestionParser":
        from src.models.language.llm_answer_generator import OpenAICompatibleQuestionRouter

        return cls(OpenAICompatibleQuestionRouter.from_env())

    def parse(self, question: str) -> QuestionUnderstanding:
        route = self.router.route(question)
        routed_intent = str(route.get("intent") or "unsupported").strip().lower()
        intent = self.INTENT_MAP.get(routed_intent, routed_intent)
        return QuestionUnderstanding(
            question=question,
            intent=intent,
            diseases=list(route.get("diseases") or []),
            anatomies=list(route.get("anatomies") or []),
        )


class KnowledgeGraphVQAModel:
    """
    Baseline end-to-end VQA orchestrator.

    This version assumes dynamic vision findings were already ingested into Neo4j.
    It uses question parsing + KG retrieval + deterministic answer templates.
    """

    def __init__(
        self,
        kg_client: Any,
        parser: Optional[RuleBasedQuestionParser] = None,
        answer_generator: Optional[Any] = None,
        min_confidence: float = 0.25,
        limit: int = 5,
        require_static_backbone: bool = True,
        vision_predictor: Optional[Any] = None,
        ingest_on_the_fly_vision: bool = True,
        dynamic_ingester: Optional[Any] = None,
    ):
        self.kg_client = kg_client
        self.parser = parser or RuleBasedQuestionParser()
        self.answer_generator = answer_generator
        self.min_confidence = float(min_confidence)
        self.limit = int(limit)
        self.require_static_backbone = require_static_backbone
        self.vision_predictor = vision_predictor
        self.ingest_on_the_fly_vision = ingest_on_the_fly_vision
        self.dynamic_ingester = dynamic_ingester

    def answer(
        self,
        question: str,
        study_id: Optional[str] = None,
        image_element_id: Optional[str] = None,
        image_path: Optional[str] = None,
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = VQARequest(
            question=question,
            study_id=self._optional_clean(study_id),
            image_element_id=self._optional_clean(image_element_id),
            image_path=image_path,
            patient_id=self._optional_clean(patient_id),
            user_id=self._optional_clean(user_id),
        )
        vision_rows = self._maybe_predict_and_ingest(request)
        understanding = self.parser.parse(question)
        context = self.retrieve_context(request, understanding)
        generated = self.generate_answer(understanding, context, question=question)

        return {
            "question": question,
            "request": asdict(request),
            "understanding": asdict(understanding),
            "answer": generated["answer"],
            "confidence": context.get("confidence", 0.0),
            "explanation": generated["explanation"],
            "answer_source": generated.get("source", "template"),
            "vision_dynamic_rows": len(vision_rows),
            "kg_context": context,
        }

    def _maybe_predict_and_ingest(self, request: VQARequest) -> List[Dict[str, Any]]:
        if self.vision_predictor is None or not request.image_path:
            return []

        predict_kwargs = {
            "image_path": request.image_path,
            "study_id": request.study_id,
            "image_element_id": request.image_element_id,
            "subject_id": request.patient_id or request.user_id or "",
            "patient_id": request.patient_id or "",
            "user_id": request.user_id or "",
        }
        supported_params = inspect.signature(self.vision_predictor.predict).parameters
        rows = self.vision_predictor.predict(
            **{key: value for key, value in predict_kwargs.items() if key in supported_params}
        )
        if rows and self.ingest_on_the_fly_vision:
            ingester = self.dynamic_ingester
            if ingester is None:
                from src.kg.ingest_dynamic import ingest_dynamic_entities

                ingester = ingest_dynamic_entities

            ingester(self.kg_client, rows)
        return rows

    def retrieve_context(
        self,
        request: VQARequest,
        understanding: QuestionUnderstanding,
    ) -> Dict[str, Any]:
        disease = understanding.diseases[0] if understanding.diseases else None
        anatomy = understanding.anatomies[0] if understanding.anatomies else None
        common = {
            "client": self.kg_client,
            "study_id": request.study_id,
            "image_element_id": request.image_element_id,
            "min_confidence": self.min_confidence,
            "limit": self.limit,
            "require_static_backbone": self.require_static_backbone,
        }

        if understanding.intent == "location":
            if not disease:
                return self._empty_context("location", "missing_disease", request, understanding)
            return retrieve_location_context(disease=disease, **common)

        if understanding.intent == "existence":
            if not disease:
                return self._empty_context("existence", "missing_disease", request, understanding)
            return retrieve_existence_context(disease=disease, anatomy=anatomy, **common)

        if understanding.intent == "abnormality":
            return retrieve_abnormality_context(anatomy=anatomy, **common)

        return self._empty_context(understanding.intent, "unsupported_intent", request, understanding)

    def generate_answer(
        self,
        understanding: QuestionUnderstanding,
        context: Dict[str, Any],
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        template = self._generate_template_answer(understanding, context)
        if self.answer_generator is None:
            return {**template, "source": "template"}

        generated = self.answer_generator.generate(
            question=question or understanding.question,
            understanding=asdict(understanding),
            kg_context=context,
            fallback=template,
        )
        return {**generated, "source": "llm"}

    def _generate_template_answer(
        self,
        understanding: QuestionUnderstanding,
        context: Dict[str, Any],
    ) -> Dict[str, str]:
        if context.get("answer") == "insufficient_evidence":
            return {
                "answer": "insufficient_evidence",
                "explanation": self._insufficient_explanation(context),
            }

        answer = str(context.get("answer", "insufficient_evidence"))
        evidences = context.get("evidences") or []
        top = evidences[0] if evidences else {}

        if understanding.intent == "existence":
            disease = (understanding.diseases or [top.get("disease", "the target disease")])[0]
            anatomy = (understanding.anatomies or top.get("anatomy_candidates") or [None])[0]
            answer_text = "yes"
            location_text = f" in {anatomy}" if anatomy else ""
            explanation = (
                f"{top.get('finding', 'A finding')} was detected{location_text} and is linked "
                f"to {disease} in the KG. Confidence={context.get('confidence', 0.0):.2f}."
            )
            return {"answer": answer_text, "explanation": explanation}

        if understanding.intent == "location":
            explanation = (
                f"The strongest KG evidence places {top.get('disease', 'the disease')} at "
                f"{answer} via {top.get('finding', 'a finding')}. "
                f"Confidence={context.get('confidence', 0.0):.2f}."
            )
            return {"answer": answer, "explanation": explanation}

        explanation = (
            f"The most supported abnormality is {answer}, associated with "
            f"{top.get('finding', 'a finding')} at {top.get('anatomy', 'the queried anatomy')}. "
            f"Confidence={context.get('confidence', 0.0):.2f}."
        )
        return {"answer": answer, "explanation": explanation}

    def _empty_context(
        self,
        intent: str,
        reason: str,
        request: VQARequest,
        understanding: QuestionUnderstanding,
    ) -> Dict[str, Any]:
        return {
            "intent": intent,
            "query": {
                "reason": reason,
                "study_id": request.study_id,
                "image_element_id": request.image_element_id,
                "diseases": understanding.diseases,
                "anatomies": understanding.anatomies,
            },
            "answer": "insufficient_evidence",
            "confidence": 0.0,
            "evidence_count": 0,
            "evidences": [],
            "candidates": [],
        }

    def _insufficient_explanation(self, context: Dict[str, Any]) -> str:
        reason = context.get("query", {}).get("reason")
        if reason == "missing_disease":
            return "No disease entity was found in the question, so KG retrieval could not run."
        if reason == "unsupported_intent":
            return "The parsed question intent is not supported by the current KG retrieval baseline."
        return "No KG evidence met the query constraints and confidence threshold."

    def _optional_clean(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value).strip().lower())
        return text or None


__all__ = [
    "KnowledgeGraphVQAModel",
    "QuestionUnderstanding",
    "RuleBasedQuestionParser",
    "SUPPORTED_INTENTS",
    "VQARequest",
]
