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


def is_anatomy_compatible(finding: str, anatomy: str) -> bool:
    """Helper to check if a chest X-ray finding is anatomically compatible with a target anatomy."""
    f_clean = str(finding).lower().strip()
    a_clean = str(anatomy).lower().strip()

    # 1. Cardiomegaly / Heart silhouette / Mediastinum
    if "cardiomegaly" in f_clean or "heart" in f_clean:
        return any(x in a_clean for x in ["heart", "cardiac", "mediastinum", "silhouette", "chest"])

    # 2. Bone / Spine / Rib / Clavicle
    if "fracture" in f_clean or "degenerative" in f_clean or "scoliosis" in f_clean:
        return any(x in a_clean for x in ["rib", "spine", "clavicle", "bone", "vertebra", "back", "chest wall"])

    # 3. Hernia
    if "hernia" in f_clean:
        return any(x in a_clean for x in ["diaphragm", "abdomen", "lower lung"])

    # 4. Lung and Pleural parenchymal findings (effusion, pneumothorax, atelectasis, consolidation, pneumonia, edema, opacity, etc.)
    # These should NOT be located in the heart, spine, or clavicle
    if any(x in a_clean for x in ["heart", "spine", "clavicle"]):
        return False

    return True


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
        
        # Map colloquial anatomy terms to canonical graph labels
        anatomy_mapping = {
            "left base": ["left lower lung zone"],
            "left base of lung": ["left lower lung zone"],
            "right base": ["right lower lung zone"],
            "right base of lung": ["right lower lung zone"],
            "left apical": ["left apical zone"],
            "right apical": ["right apical zone"],
            "retrocardiac": ["left lower lung zone"],
            "left lower lobe retrocardiac": ["left lower lung zone"],
            "bibasilar": ["left lower lung zone", "right lower lung zone"],
            "bilateral upper lobe": ["left upper lung zone", "right upper lung zone"],
            "bibasilar area": ["left lower lung zone", "right lower lung zone"],
            "bilateral area": ["left lung", "right lung"],
            "bilateral": ["left lung", "right lung"],
        }
        mapped_anatomies = []
        for ant in entities["ANATOMY"]:
            if ant in anatomy_mapping:
                mapped_anatomies.extend(anatomy_mapping[ant])
            else:
                mapped_anatomies.append(ant)
                
        # Deduplicate while preserving order
        seen_ant = set()
        entities["ANATOMY"] = [x for x in mapped_anatomies if not (x in seen_ant or seen_ant.add(x))]

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
            "left base",
            "right base",
            "left apical",
            "right apical",
            "retrocardiac",
            "left lower lobe retrocardiac",
            "bibasilar",
            "bilateral upper lobe",
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
        language: Optional[str] = None,
    ):
        import os
        self.kg_client = kg_client
        self.parser = parser or RuleBasedQuestionParser()
        self.answer_generator = answer_generator
        self.min_confidence = float(min_confidence)
        self.limit = int(limit)
        self.require_static_backbone = require_static_backbone
        self.vision_predictor = vision_predictor
        self.ingest_on_the_fly_vision = ingest_on_the_fly_vision
        self.dynamic_ingester = dynamic_ingester
        self.language = language or os.getenv("VQA_LLM_LANGUAGE", "vi")

    def answer(
        self,
        question: str,
        study_id: Optional[str] = None,
        image_element_id: Optional[str] = None,
        image_path: Optional[str] = None,
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
        hybrid_rows: Optional[List[Dict[str, Any]]] = None,
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
        context = self.retrieve_context(request, understanding, hybrid_rows=hybrid_rows)
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

    def _re_rank_evidences(
        self,
        evidences: List[Dict[str, Any]],
        understanding: QuestionUnderstanding,
    ) -> List[Dict[str, Any]]:
        if not evidences:
            return []

        query_diseases = {str(d).lower().strip() for d in (understanding.diseases or []) if d}
        query_anatomies = {str(a).lower().strip() for a in (understanding.anatomies or []) if a}

        re_ranked = []
        for ev in evidences:
            score = 0

            # Extract evidence values
            ev_disease = str(ev.get("disease") or "").lower().strip()
            ev_finding = str(ev.get("finding") or "").lower().strip()
            ev_anatomy = str(ev.get("anatomy") or "").lower().strip()
            ev_anatomy_candidates = {str(ac).lower().strip() for ac in (ev.get("anatomy_candidates") or []) if ac}

            # Check disease matching (disease or finding)
            for q_dis in query_diseases:
                if q_dis in ev_disease or ev_disease in q_dis:
                    score += 1
                if q_dis in ev_finding or ev_finding in q_dis:
                    score += 1

            # Check anatomy matching (anatomy or anatomy_candidates)
            for q_anat in query_anatomies:
                if q_anat in ev_anatomy or ev_anatomy in q_anat:
                    score += 1
                for ac in ev_anatomy_candidates:
                    if q_anat in ac or ac in q_anat:
                        score += 1

            re_ranked.append((ev, score))

        # Sort by relevance_score * 10 + confidence descending
        re_ranked.sort(key=lambda item: (item[1] * 10 + item[0].get("confidence", 0.0)), reverse=True)
        return [item[0] for item in re_ranked]

    def _check_ambiguity(self, evidences: List[Dict[str, Any]]) -> bool:
        if len(evidences) < 2:
            return False

        ev1 = evidences[0]
        ev2 = evidences[1]

        dis1 = str(ev1.get("disease") or "").lower().strip()
        dis2 = str(ev2.get("disease") or "").lower().strip()
        fin1 = str(ev1.get("finding") or "").lower().strip()
        fin2 = str(ev2.get("finding") or "").lower().strip()

        different = (dis1 != dis2) or (fin1 != fin2)
        conf_diff = abs(ev1.get("confidence", 0.0) - ev2.get("confidence", 0.0))

        return different and (conf_diff <= 0.05)

    def retrieve_context(
        self,
        request: VQARequest,
        understanding: QuestionUnderstanding,
        hybrid_rows: Optional[List[Dict[str, Any]]] = None,
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
                context = self._empty_context("location", "missing_disease", request, understanding)
            else:
                context = retrieve_location_context(disease=disease, **common)

        elif understanding.intent == "existence":
            if not disease:
                context = self._empty_context("existence", "missing_disease", request, understanding)
            else:
                if len(understanding.anatomies) > 1:
                    merged_evidences = []
                    other_findings = []
                    for anat in understanding.anatomies:
                        sub_context = retrieve_existence_context(disease=disease, anatomy=anat, **common)
                        merged_evidences.extend(sub_context.get("evidences") or [])
                        other_findings.extend(sub_context.get("other_findings") or [])
                    
                    seen_ev = set()
                    deduped_evidences = []
                    for ev in merged_evidences:
                        key = (ev.get("image_element_id"), ev.get("finding"), ev.get("disease"))
                        if key not in seen_ev:
                            seen_ev.add(key)
                            deduped_evidences.append(ev)
                            
                    seen_other = set()
                    deduped_other = []
                    for f in other_findings:
                        key = (f.get("finding"), f.get("disease"), tuple(f.get("anatomy_candidates") or []))
                        if key not in seen_other:
                            seen_other.add(key)
                            deduped_other.append(f)
                            
                    context = retrieve_existence_context(disease=disease, anatomy=understanding.anatomies[0], **common)
                    context["evidences"] = deduped_evidences
                    context["other_findings"] = deduped_other
                    context["evidence_count"] = len(deduped_evidences)
                else:
                    context = retrieve_existence_context(disease=disease, anatomy=anatomy, **common)

        elif understanding.intent == "abnormality":
            if len(understanding.anatomies) > 1:
                merged_evidences = []
                other_findings = []
                for anat in understanding.anatomies:
                    sub_context = retrieve_abnormality_context(anatomy=anat, **common)
                    merged_evidences.extend(sub_context.get("evidences") or [])
                    other_findings.extend(sub_context.get("other_findings") or [])
                
                seen_ev = set()
                deduped_evidences = []
                for ev in merged_evidences:
                    key = (ev.get("image_element_id"), ev.get("finding"), ev.get("anatomy"), ev.get("disease"))
                    if key not in seen_ev:
                        seen_ev.add(key)
                        deduped_evidences.append(ev)
                        
                seen_other = set()
                deduped_other = []
                for f in other_findings:
                    key = (f.get("finding"), f.get("disease"), tuple(f.get("anatomy_candidates") or []))
                    if key not in seen_other:
                        seen_other.add(key)
                        deduped_other.append(f)
                        
                context = retrieve_abnormality_context(anatomy=understanding.anatomies[0], **common)
                context["evidences"] = deduped_evidences
                context["other_findings"] = deduped_other
                context["evidence_count"] = len(deduped_evidences)
            else:
                context = retrieve_abnormality_context(anatomy=anatomy, **common)

        else:
            context = self._empty_context(understanding.intent, "unsupported_intent", request, understanding)

        # Re-rank evidences based on semantic relevance
        evidences = context.get("evidences") or []
        if not evidences and hybrid_rows and understanding.intent in ("abnormality", "existence", "location"):
            # Construct evidences from hybrid_rows in-memory as a fallback
            for r in hybrid_rows:
                confidence = float(r.get("confidence") or 0.0)
                if confidence < self.min_confidence:
                    continue
                finding = r.get("finding")
                disease = r.get("disease") or finding
                anatomy = r.get("anatomy") or (r.get("anatomy_candidates")[0] if r.get("anatomy_candidates") else None)
                
                logic_path = (
                    f"ImageElement({r.get('image_element_id') or request.image_element_id})"
                    f" -[EXHIBITS {confidence:.2f}]-> "
                    f"Finding({finding})"
                    f" -[:LOCATED_AT]-> Anatomy({anatomy})"
                    f" -[:SUGGESTS]-> Disease({disease})"
                )
                evidences.append({
                    "study_id": r.get("study_id") or request.study_id,
                    "image_element_id": r.get("image_element_id") or request.image_element_id,
                    "anatomy": anatomy,
                    "finding": finding,
                    "disease": disease,
                    "confidence": confidence,
                    "parent_concepts": [],
                    "logic_path": logic_path,
                    "explanation": f"Evidence: {finding} at {anatomy} suggests {disease} (confidence {confidence:.2f})."
                })
        if understanding.anatomies:
            filtered_evidences = []
            for ev in evidences:
                finding = ev.get("finding") or ev.get("disease") or ""
                if any(is_anatomy_compatible(finding, anat) for anat in understanding.anatomies):
                    filtered_evidences.append(ev)
            evidences = filtered_evidences
        re_ranked = self._re_rank_evidences(evidences, understanding)
        context["evidences"] = re_ranked

        # Update answer, confidence and candidates
        if re_ranked:
            context["confidence"] = re_ranked[0]["confidence"]
            if understanding.intent == "existence":
                context["answer"] = "yes"
            elif understanding.intent == "location":
                seen = set()
                candidates = []
                for ev in re_ranked:
                    val = ev.get("anatomy")
                    if val and val not in seen:
                        seen.add(val)
                        matching_evs = [e for e in re_ranked if e.get("anatomy") == val]
                        candidates.append({
                            "name": val,
                            "confidence": max(e["confidence"] for e in matching_evs),
                            "count": len(matching_evs)
                        })
                context["candidates"] = candidates
                context["answer"] = candidates[0]["name"] if candidates else "insufficient_evidence"
            elif understanding.intent == "abnormality":
                seen = set()
                candidates = []
                for ev in re_ranked:
                    val = ev.get("disease")
                    if val and val not in seen:
                        seen.add(val)
                        matching_evs = [e for e in re_ranked if e.get("disease") == val]
                        candidates.append({
                            "name": val,
                            "confidence": max(e["confidence"] for e in matching_evs),
                            "count": len(matching_evs)
                        })
                context["candidates"] = candidates
                context["answer"] = ", ".join([c["name"] for c in candidates]) if candidates else "insufficient_evidence"
        else:
            context["answer"] = "insufficient_evidence"
            context["confidence"] = 0.0
            context["candidates"] = []

        context["is_ambiguous"] = self._check_ambiguity(re_ranked)

        # Retrieve all image findings to extract other findings (warnings)
        if request.study_id or request.image_element_id:
            from src.kg.queries import retrieve_all_image_findings
            all_findings = retrieve_all_image_findings(
                client=self.kg_client,
                study_id=request.study_id,
                image_element_id=request.image_element_id,
                min_confidence=self.min_confidence,
            )
            main_findings = {ev.get("finding") for ev in re_ranked if ev.get("finding")}
            other_findings = [f for f in all_findings if f.get("finding") not in main_findings]
            context["other_findings"] = other_findings
        else:
            context["other_findings"] = []

        return context

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
        is_vi = self.language.lower() == "vi"

        if is_vi:
            vi_answers = {"yes": "có", "no": "không"}
            answer_text = vi_answers.get(answer, answer)
            if understanding.intent == "existence":
                disease = (understanding.diseases or [top.get("disease", "bệnh mục tiêu")])[0]
                anatomy = (understanding.anatomies or top.get("anatomy_candidates") or [None])[0]
                location_text = f" ở {anatomy}" if anatomy else ""
                explanation = (
                    f"Hình ảnh ghi nhận có {top.get('finding', 'tổn thương')}{location_text}, "
                    f"gợi ý nhiều đến bệnh lý {disease} (mức độ tin cậy khoảng {context.get('confidence', 0.0)*100:.0f}%)."
                )
                if context.get("is_ambiguous"):
                    explanation += " Chẩn đoán hiện tại chưa hoàn toàn rõ ràng. Bạn có thể cho biết thêm bệnh nhân có kèm theo triệu chứng ho, sốt hoặc đau ngực không?"
                return {"answer": answer_text, "explanation": explanation}

            if understanding.intent == "location":
                explanation = (
                    f"Bất thường {top.get('finding', 'tổn thương')} liên quan đến bệnh lý {top.get('disease', 'bệnh')} "
                    f"được ghi nhận chủ yếu ở vị trí {answer_text} (độ tin cậy khoảng {context.get('confidence', 0.0)*100:.0f}%)."
                )
                if context.get("is_ambiguous"):
                    explanation += " Vị trí tổn thương chưa rõ nét hoàn toàn. Bệnh nhân có triệu chứng ho, sốt hay đau ngực đi kèm không?"
                return {"answer": answer_text, "explanation": explanation}

            explanation = (
                f"Ghi nhận tổn thương dạng {answer_text} liên quan đến {top.get('finding', 'bất thường')} "
                f"tại vị trí {top.get('anatomy', 'ngực')} (độ tin cậy khoảng {context.get('confidence', 0.0)*100:.0f}%)."
            )
            if context.get("is_ambiguous"):
                explanation += " Khuyến nghị kiểm tra thêm lâm sàng. Bệnh nhân có biểu hiện đau ngực, ho hay sốt không?"
            return {"answer": answer_text, "explanation": explanation}

        if understanding.intent == "existence":
            disease = (understanding.diseases or [top.get("disease", "the target disease")])[0]
            anatomy = (understanding.anatomies or top.get("anatomy_candidates") or [None])[0]
            answer_text = "yes"
            location_text = f" in the {anatomy}" if anatomy else ""
            explanation = (
                f"Chest X-ray shows {top.get('finding', 'a finding')}{location_text}, "
                f"which is suggestive of {disease} (confidence level approx. {context.get('confidence', 0.0)*100:.0f}%)."
            )
            if context.get("is_ambiguous"):
                explanation += " The presentation is somewhat ambiguous. Does the patient present with clinical symptoms like cough, fever, or chest pain?"
            return {"answer": answer_text, "explanation": explanation}

        if understanding.intent == "location":
            explanation = (
                f"The pathology of {top.get('disease', 'the disease')} is observed at "
                f"{answer} via {top.get('finding', 'a finding')} (confidence level approx. {context.get('confidence', 0.0)*100:.0f}%)."
            )
            if context.get("is_ambiguous"):
                explanation += " The exact localization is slightly ambiguous. Does the patient have matching symptoms like cough, fever, or chest pain?"
            return {"answer": answer, "explanation": explanation}

        explanation = (
            f"The primary finding is {answer}, associated with "
            f"{top.get('finding', 'a finding')} at the {top.get('anatomy', 'queried anatomy')} "
            f"(confidence level approx. {context.get('confidence', 0.0)*100:.0f}%)."
        )
        if context.get("is_ambiguous"):
            explanation += " Please correlate clinically. Is the patient experiencing cough, fever, or chest pain?"
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
            "is_ambiguous": False,
        }

    def _insufficient_explanation(self, context: Dict[str, Any]) -> str:
        reason = context.get("query", {}).get("reason")
        is_vi = self.language.lower() == "vi"
        if is_vi:
            if reason == "missing_disease":
                base = "Không tìm thấy thực thể bệnh nào trong câu hỏi, do đó không thể thực hiện truy vấn KG."
            elif reason == "unsupported_intent":
                base = "Ý định của câu hỏi đã được phân tích không được hỗ trợ bởi phương pháp truy vấn KG baseline hiện tại."
            else:
                base = "Không có minh chứng KG nào đáp ứng các ràng buộc truy vấn và ngưỡng độ tin cậy."
            return base + " Bạn có muốn kiểm tra các triệu chứng cụ thể của bệnh nhân như ho, sốt hoặc khó thở không?"

        if reason == "missing_disease":
            base = "No disease entity was found in the question, so KG retrieval could not run."
        elif reason == "unsupported_intent":
            base = "The parsed question intent is not supported by the current KG retrieval baseline."
        else:
            base = "No KG evidence met the query constraints and confidence threshold."
        return base + " Would you like to check for specific patient symptoms such as fever, cough, or dyspnea?"

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
