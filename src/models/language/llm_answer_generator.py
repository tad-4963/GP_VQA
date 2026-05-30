from __future__ import annotations

import json
import os
from datetime import date
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class LLMAnswer:
    answer: str
    explanation: str
    raw_response: Optional[str] = None


class KGContextPromptBuilder:
    """Builds a compact, evidence-grounded prompt for answer generation."""

    def __init__(self, max_evidences: int = 5, max_candidates: int = 5):
        self.max_evidences = max(1, int(max_evidences))
        self.max_candidates = max(1, int(max_candidates))

    def build_messages(
        self,
        question: str,
        understanding: Mapping[str, Any],
        kg_context: Mapping[str, Any],
    ) -> List[Dict[str, str]]:
        payload = {
            "question": question,
            "understanding": {
                "intent": understanding.get("intent"),
                "diseases": list(understanding.get("diseases") or []),
                "anatomies": list(understanding.get("anatomies") or []),
            },
            "kg_context": self._compact_context(kg_context),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You answer chest X-ray visual questions using only the supplied "
                    "knowledge graph evidence and logic paths. Do not add findings, "
                    "locations, or diagnoses that are absent from the evidence. If evidence is insufficient, answer exactly "
                    "'insufficient_evidence'. Return strict JSON with keys "
                    "'answer' and 'explanation'. The answer must be a concise string, "
                    "not an array or nested object. The explanation should be 2-4 "
                    "sentences and must explicitly cover: (1) what was observed on "
                    "the image from the Finding/Anatomy evidence, (2) what disease "
                    "or abnormality this supports from the Disease evidence, and "
                    "(3) the confidence/evidence strength. Use the supplied logic_path "
                    "values as the reasoning chain, but keep the wording concise. "
                    "Briefly caveat that the response is based only on the provided KG evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    def _compact_context(self, kg_context: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "intent": kg_context.get("intent"),
            "answer": kg_context.get("answer"),
            "confidence": kg_context.get("confidence", 0.0),
            "evidence_count": kg_context.get("evidence_count", 0),
            "evidences": self._compact_evidences(kg_context.get("evidences"), self.max_evidences),
            "candidates": self._compact_candidates(kg_context.get("candidates"), self.max_candidates),
            "query": kg_context.get("query") or {},
            "context_policy": kg_context.get("context_policy") or {},
        }

    def _take(self, values: Any, limit: int) -> List[Any]:
        if not isinstance(values, list):
            return []
        return values[:limit]

    def _compact_evidences(self, values: Any, limit: int) -> List[Dict[str, Any]]:
        evidences = self._take(values, limit)
        compacted = []
        for evidence in evidences:
            if not isinstance(evidence, Mapping):
                continue
            anatomy = evidence.get("anatomy")
            anatomy_candidates = evidence.get("anatomy_candidates")
            if not anatomy and isinstance(anatomy_candidates, list) and anatomy_candidates:
                anatomy = anatomy_candidates[0]
            compacted.append(
                {
                    "finding": evidence.get("finding"),
                    "observed_at": anatomy,
                    "anatomy_candidates": anatomy_candidates or [],
                    "diagnosis": evidence.get("disease") or evidence.get("answer"),
                    "confidence": evidence.get("confidence"),
                    "logic_path": evidence.get("logic_path", ""),
                    "kg_explanation": evidence.get("explanation", ""),
                }
            )
        return compacted

    def _compact_candidates(self, values: Any, limit: int) -> List[Dict[str, Any]]:
        candidates = self._take(values, limit)
        compacted = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            compacted.append(
                {
                    "finding": candidate.get("finding"),
                    "observed_at": candidate.get("anatomy"),
                    "diagnosis": candidate.get("disease") or candidate.get("answer"),
                    "confidence": candidate.get("confidence"),
                    "logic_path": candidate.get("logic_path", ""),
                }
            )
        return compacted


class QuestionRoutingPromptBuilder:
    """Builds a compact prompt for routing natural-language KG/VQA questions."""

    def build_messages(self, question: str, current_date: Optional[str] = None) -> List[Dict[str, str]]:
        payload = {
            "question": question,
            "current_date": current_date or date.today().isoformat(),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You classify user questions for a chest X-ray VQA/KG system. "
                    "Return strict JSON only with keys: intent, diseases, findings, anatomies, "
                    "patient_id, user_id, study_id, image_element_id, ingest_date, aggregate, explanation. "
                    "Allowed intents are: image_abnormality, image_disease_existence, image_disease_location, "
                    "patient_disease_query, patient_history, cohort_count, unsupported. "
                    "Use image_abnormality for questions asking what findings/diseases are in one image. "
                    "Use image_disease_existence for questions asking whether one image has a named disease. "
                    "Use image_disease_location for questions asking where a disease/finding is located in one image. "
                    "Use patient_disease_query when the user asks to list/find/retrieve patients or images by disease. "
                    "Use patient_history when the user asks what diseases/findings a specific patient/user has or had. "
                    "Use cohort_count when the user asks how many patients/images match a disease/date condition. "
                    "diseases, findings, and anatomies must be lowercase arrays. patient_id/user_id/study_id/"
                    "image_element_id should be copied from the question when present, otherwise null. "
                    "ingest_date must be YYYY-MM-DD when the user gives a date, asks for today, yesterday, "
                    "or a relative date based on current_date; otherwise null. aggregate is true for count/statistics. "
                    "Do not invent diseases, ids, or dates that are not present or implied by the question."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]


class OpenAICompatibleAnswerGenerator:
    """
    Optional LLM answer generator for OpenAI-compatible chat-completions servers.

    This intentionally uses httpx directly so the core project does not require a
    provider SDK. It can target hosted APIs or local servers that implement
    /v1/chat/completions.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_builder: Optional[KGContextPromptBuilder] = None,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.prompt_builder = prompt_builder or KGContextPromptBuilder()

    @classmethod
    def from_env(cls) -> "OpenAICompatibleAnswerGenerator":
        model = os.getenv("VQA_LLM_MODEL")
        if not model:
            raise ValueError("VQA_LLM_MODEL is required when --llm-provider openai-compatible is used")
        return cls(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("VQA_LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout=float(os.getenv("VQA_LLM_TIMEOUT", "30")),
            temperature=float(os.getenv("VQA_LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("VQA_LLM_MAX_TOKENS", "256")),
        )

    def generate(
        self,
        question: str,
        understanding: Mapping[str, Any],
        kg_context: Mapping[str, Any],
        fallback: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        import httpx

        messages = self.prompt_builder.build_messages(question, understanding, kg_context)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json_content(content, fallback=fallback)
        parsed["llm_raw_response"] = content
        return parsed

    def _parse_json_content(
        self,
        content: str,
        fallback: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            if fallback:
                return {
                    "answer": str(fallback.get("answer", "insufficient_evidence")),
                    "explanation": str(fallback.get("explanation", "")),
                }
            return {"answer": "insufficient_evidence", "explanation": content.strip()}

        answer = self._normalize_answer(payload.get("answer"))
        explanation = str(payload.get("explanation", "")).strip()
        if not explanation and fallback:
            explanation = str(fallback.get("explanation", ""))
        return {"answer": answer, "explanation": explanation}

    def _normalize_answer(self, value: Any) -> str:
        if value is None:
            return "insufficient_evidence"
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(parts) if parts else "insufficient_evidence"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value).strip() or "insufficient_evidence"


class OpenAICompatibleQuestionRouter:
    """
    LLM router for natural-language questions that are outside the trained
    intent model, especially patient/date retrieval queries.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        prompt_builder: Optional[QuestionRoutingPromptBuilder] = None,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.prompt_builder = prompt_builder or QuestionRoutingPromptBuilder()

    @classmethod
    def from_env(cls) -> "OpenAICompatibleQuestionRouter":
        model = os.getenv("VQA_ROUTER_MODEL") or os.getenv("VQA_LLM_MODEL")
        if not model:
            raise ValueError("VQA_LLM_MODEL or VQA_ROUTER_MODEL is required for the OpenAI question router")
        return cls(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("VQA_LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout=float(os.getenv("VQA_ROUTER_TIMEOUT", os.getenv("VQA_LLM_TIMEOUT", "30"))),
            temperature=float(os.getenv("VQA_ROUTER_TEMPERATURE", os.getenv("VQA_LLM_TEMPERATURE", "0"))),
            max_tokens=int(os.getenv("VQA_ROUTER_MAX_TOKENS", "256")),
        )

    def route(self, question: str, current_date: Optional[str] = None) -> Dict[str, Any]:
        import httpx

        messages = self.prompt_builder.build_messages(question, current_date=current_date)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json_content(content)
        parsed["llm_raw_response"] = content
        return parsed

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {
                "intent": "unsupported",
                "diseases": [],
                "findings": [],
                "anatomies": [],
                "patient_id": None,
                "user_id": None,
                "study_id": None,
                "image_element_id": None,
                "ingest_date": None,
                "aggregate": False,
                "explanation": content.strip(),
            }

        intent = str(payload.get("intent") or "unsupported").strip().lower()
        allowed_intents = {
            "image_abnormality",
            "image_disease_existence",
            "image_disease_location",
            "patient_disease_query",
            "patient_history",
            "cohort_count",
            "unsupported",
        }
        if intent == "image_vqa":
            intent = "image_abnormality"
        if intent not in allowed_intents:
            intent = "unsupported"

        def normalize_list(value: Any) -> List[str]:
            values = [value] if isinstance(value, str) else list(value or [])
            normalized: List[str] = []
            for item in values:
                text = str(item or "").strip().lower()
                if text and text not in normalized:
                    normalized.append(text)
            return normalized

        def normalize_optional(value: Any) -> Optional[str]:
            text = str(value or "").strip().lower()
            if text in {"", "null", "none", "unknown"}:
                return None
            return text

        ingest_date = payload.get("ingest_date")
        ingest_date = str(ingest_date).strip() if ingest_date else None
        if ingest_date in {"null", "none", ""}:
            ingest_date = None

        return {
            "intent": intent,
            "diseases": normalize_list(payload.get("diseases")),
            "findings": normalize_list(payload.get("findings")),
            "anatomies": normalize_list(payload.get("anatomies")),
            "patient_id": normalize_optional(payload.get("patient_id")),
            "user_id": normalize_optional(payload.get("user_id")),
            "study_id": normalize_optional(payload.get("study_id")),
            "image_element_id": normalize_optional(payload.get("image_element_id") or payload.get("dicom_id")),
            "ingest_date": ingest_date,
            "aggregate": bool(payload.get("aggregate")) or intent == "cohort_count",
            "explanation": str(payload.get("explanation") or "").strip(),
        }


__all__ = [
    "KGContextPromptBuilder",
    "LLMAnswer",
    "OpenAICompatibleAnswerGenerator",
    "OpenAICompatibleQuestionRouter",
    "QuestionRoutingPromptBuilder",
]
