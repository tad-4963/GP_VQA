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

    def __init__(self, max_evidences: int = 5, max_candidates: int = 5, language: str = "vi"):
        self.max_evidences = max(1, int(max_evidences))
        self.max_candidates = max(1, int(max_candidates))
        self.language = str(language).lower()

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

        language_instruction = ""
        if self.language == "vi":
            language_instruction = (
                " IMPORTANT: You MUST generate the 'answer' and 'explanation' in Vietnamese (Tiếng Việt). "
                "The 'answer' must be a concise string (e.g. 'có', 'không', or the name of the abnormality/location in Vietnamese). "
                "The 'explanation' MUST be written in a natural, professional, and empathetic clinical Vietnamese tone, exactly like a senior radiologist. "
                "Translate medical terms to standard Vietnamese: 'pneumonia' -> 'viêm phổi', 'pleural effusion' -> 'tràn dịch màng phổi', "
                "'cardiomegaly' -> 'bóng tim to/phì đại tim', 'fracture' -> 'gãy xương sườn/xương đòn', 'atelectasis' -> 'xẹp phổi', "
                "'consolidation' -> 'đông đặc phổi', 'pulmonary edema' -> 'phù phổi', 'pneumothorax' -> 'tràn khí màng phổi', 'lung opacity' -> 'vùng mờ ở phổi'. "
                "Do NOT use robotic translations like 'dựa trên bằng chứng đồ thị tri thức' or 'mức độ tin cậy là 0.55'. "
                "You MUST specify the anatomy locations (observed_at) of the findings if available (e.g., 'đáy phổi', 'phổi phải', 'xương sườn'). "
                "If evidence is insufficient, return 'insufficient_evidence' (do not translate this specific string for the 'answer' key). "
                "If the diagnosis is ambiguous or evidence is insufficient, suggest specific next diagnostic steps (e.g. suggest sputum test or chest CT for pneumonia, suggest diagnostic pleural tap for pleural effusion, or recommend checking history of chest trauma/fall for fracture). "
                "CRITICAL: If 'other_findings' are provided, you MUST explicitly list and warn about ALL of these secondary abnormalities at the end of the explanation in a structured way (e.g., 'Bên cạnh đó, hình ảnh còn ghi nhận các bất thường đi kèm bao gồm: [bất thường] ở [vị trí giải phẫu]...'), ensuring completeness of the radiologist report. "
                "However, do NOT include secondary findings from 'other_findings' in the 'answer' key. The 'answer' must ONLY contain findings directly supported by the main 'evidences' list for the queried anatomy."
            )
        else:
            language_instruction = (
                " Return strict JSON with keys 'answer' and 'explanation' in English. "
                "The 'answer' must be a concise string or a comma-separated list of findings/locations if there are multiple (e.g. 'yes', 'no', 'pneumothorax', 'right lung', or 'cardiomegaly, pleural effusion'). "
                "The 'explanation' should be written in a natural, professional clinical tone, like a doctor explaining findings to a colleague or patient. "
                "Do NOT use database-centric or robotic phrases like 'according to the provided knowledge graph evidence' or 'confidence level is 0.55'. "
                "If the context is flagged as ambiguous (is_ambiguous: true) or if the evidence is insufficient, "
                "you MUST proactively suggest one or more clinical follow-up questions in English, asking about "
                "symptoms like chest pain, cough, or fever to help narrow down the diagnosis. "
                "Additionally, if there are other findings/pathologies provided under 'other_findings', you MUST briefly warn or mention these secondary findings at the end of the explanation in English, focusing primarily on answering the user's question first. "
                "CRITICAL: Do NOT include secondary findings from 'other_findings' in the 'answer' key. The 'answer' must ONLY contain findings directly supported by the main 'evidences' list for the queried anatomy. If the main 'evidences' list is empty, the 'answer' key MUST be 'insufficient_evidence'."
            )



        return [
            {
                "role": "system",
                "content": (
                    "You are an expert radiologist and medical doctor. You answer chest X-ray visual questions using only the supplied "
                    "knowledge graph evidence and logic paths. Do not invent entirely unrelated findings. "
                    "If evidence is insufficient, answer exactly 'insufficient_evidence'. Return strict JSON with keys 'answer' and 'explanation'. "
                    "The answer must be a concise string, not an array or nested object. "
                    "The explanation should be 2-4 sentences and written in a natural, professional clinical tone. "
                    "Instead of citing raw numerical confidence levels or database logic paths directly, describe what was observed "
                    "and suspected diagnoses as a clinical narrative (e.g., describing findings at specific locations, suspected diseases, "
                    "and qualitative clinical certainty like 'highly likely', 'suggestive', or 'moderate sign').\n\n"
                    "MEDICAL REASONING RULES:\n"
                    "1. If the question asks 'is there evidence of any abnormalities?' or asks if the image is normal/abnormal in general, and there are findings present, the 'answer' key MUST be exactly 'yes'. Do not list diseases in the 'answer' key for this specific question type.\n\n"
                    + language_instruction
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
            "is_ambiguous": bool(kg_context.get("is_ambiguous", False)),
            "evidences": self._compact_evidences(kg_context.get("evidences"), self.max_evidences),
            "candidates": self._compact_candidates(kg_context.get("candidates"), self.max_candidates),
            "other_findings": self._compact_other_findings(kg_context.get("other_findings"), self.max_evidences),
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
                    "parent_concepts": evidence.get("parent_concepts") or [],
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

    def _compact_other_findings(self, values: Any, limit: int) -> List[Dict[str, Any]]:
        findings = self._take(values, limit)
        compacted = []
        for f in findings:
            if not isinstance(f, Mapping):
                continue
            compacted.append(
                {
                    "finding": f.get("finding"),
                    "observed_at": f.get("anatomy_candidates") or [],
                    "confidence": f.get("confidence"),
                    "diagnosis": f.get("disease"),
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
                    "Do not invent diseases, ids, or dates that are not present or implied by the question. "
                    "CRITICAL: The input question may be in Vietnamese. You MUST translate any extracted diseases, "
                    "findings, and anatomies into their canonical English medical terms (e.g. 'viêm phổi' or 'phổi bị viêm' -> 'pneumonia', "
                    "'phổi phải' -> 'right lung', 'tràn dịch màng phổi' -> 'pleural effusion', 'bóng tim to' -> 'cardiomegaly') "
                    "so they match the English knowledge graph. All list elements must be in English. "
                    "The explanation should be in the same language as the user's question."
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
        language: str = "vi",
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.language = language
        self.prompt_builder = prompt_builder or KGContextPromptBuilder(language=language)

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
            language=os.getenv("VQA_LLM_LANGUAGE", "vi"),
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
