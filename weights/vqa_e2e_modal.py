import json
import os
import base64
from pathlib import Path
from typing import Any, Dict, Optional

import modal
from pydantic import BaseModel

app = modal.App("med-vqa-e2e")

vol_weights = modal.Volume.from_name("med-vqa-weights")
vol_data = modal.Volume.from_name("med-vqa-data", create_if_missing=True)
neo4j_secret = modal.Secret.from_name("neo4j-kg")
openai_secret = modal.Secret.from_name("openai-api")
hf_secret = modal.Secret.from_name("my-huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.10")
    # Cache bust: 2026-06-28 12:59
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers",
        "huggingface_hub",
        "peft",
        "Pillow",
        "numpy<2",
        "iopath",
        "einops",
        "decord",
        "hydra-core",
        "omegaconf",
        "submitit",
        "open-clip-torch",
        "ftfy",
        "regex",
        "psutil",
        "neo4j",
        "httpx",
        "fastapi[standard]",
    )
    .add_local_dir("/home/laptopdev/GP_VQA/src", remote_path="/root/src")
    .add_local_file(
        "/home/laptopdev/GP_VQA/data/label/normalized_diseases.csv",
        remote_path="/root/data/label/normalized_diseases.csv",
    )
    .add_local_file(
        "/home/laptopdev/GP_VQA/data/label/normalized_anatomy.csv",
        remote_path="/root/data/label/normalized_anatomy.csv",
    )
)

_medsam_predictor = None
_rad_dino_predictor = None

class E2EAnswerRequest(BaseModel):
    image_path: str = ""
    image_base64: str = ""
    image_filename: str = "upload.png"
    question: str = "What abnormalities are seen?"
    study_id: str = "modal_e2e_study"
    dicom_id: str = "modal_e2e_image"
    subject_id: str = ""
    patient_id: str = ""
    user_id: str = ""
    view: str = ""
    threshold: float = 0.5
    anatomy_threshold: float = 0.5
    min_confidence: float = 0.25
    limit: int = 5
    use_llm: bool = True
    use_llm_router: bool = False
    use_global: bool = True
    global_threshold: float = 0.5

class PatientsByDiseaseRequest(BaseModel):
    diseases: list[str]
    ingest_date: str = ""
    min_confidence: float = 0.25
    limit: int = 50

class NaturalLanguageKGQueryRequest(BaseModel):
    question: str
    current_date: str = ""
    min_confidence: float = 0.25
    limit: int = 50

def _settings_from_env():
    from src.kg.client import Neo4jSettings

    return Neo4jSettings.from_env()

def _counts_query() -> str:
    return """
    CALL () {
        MATCH (d:Disease)
        RETURN count(d) AS disease
    }
    CALL () {
        MATCH (a:Anatomy)
        RETURN count(a) AS anatomy
    }
    CALL () {
        MATCH (f:Finding)
        RETURN count(f) AS finding
    }
    CALL () {
        MATCH (t:TermAlias)
        RETURN count(t) AS aliases
    }
    CALL () {
        MATCH (:Finding)-[s:SUGGESTS]->(:Disease)
        RETURN count(s) AS suggests
    }
    CALL () {
        MATCH (:Finding)-[l:LOCATED_AT]->(:Anatomy)
        RETURN count(l) AS located
    }
    CALL () {
        MATCH (:ImageElement)-[e:EXHIBITS]->(:Finding)
        RETURN count(e) AS exhibits
    }
    RETURN disease, anatomy, finding, aliases, suggests, located, exhibits
    """

def _resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return str(path)

    candidates = [
        Path("/data/dataset/mimic-cxr-kaggle/images") / image_path,
        Path("/data/dataset/images") / image_path,
        Path("/data/dataset/valid") / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    checked = "\n  - ".join([str(path), *[str(c) for c in candidates]])
    raise FileNotFoundError(f"Image not found. Checked:\n  - {checked}")

def _materialize_image_input(
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_filename: str = "upload.png",
) -> str:
    if image_base64:
        payload = image_base64
        if "," in payload and payload.split(",", 1)[0].startswith("data:"):
            payload = payload.split(",", 1)[1]

        suffix = Path(image_filename or "upload.png").suffix or ".png"
        output_dir = Path("/tmp/med_vqa_uploads")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"upload_{abs(hash(payload))}{suffix}"
        output_path.write_bytes(base64.b64decode(payload))
        return str(output_path)

    if not image_path:
        raise ValueError("Either image_path or image_base64 is required")
    return _resolve_image_path(image_path)

def _get_medsam_predictor(
    threshold: float,
    anatomy_threshold: float,
    weights_path: str,
    sam_checkpoint: str,
    encoder_backend: str,
    model_id: str,
):
    global _medsam_predictor
    cache_key = (
        float(threshold),
        float(anatomy_threshold),
        weights_path,
        sam_checkpoint,
        encoder_backend,
        model_id,
    )
    if _medsam_predictor is not None and getattr(_medsam_predictor, "_modal_cache_key", None) == cache_key:
        return _medsam_predictor

    from src.models.vision.dynamic_predictor import MedVQAVisionDynamicPredictor

    predictor = MedVQAVisionDynamicPredictor(
        weights_path=weights_path,
        sam_checkpoint=sam_checkpoint,
        threshold=threshold,
        anatomy_threshold=anatomy_threshold,
        device="cuda",
        encoder_backend=encoder_backend,
        model_id=model_id,
    )
    predictor._modal_cache_key = cache_key
    _medsam_predictor = predictor
    return predictor

def _get_rad_dino_predictor(
    checkpoint_path: str,
    threshold: float,
    model_id: str,
):
    global _rad_dino_predictor
    cache_key = (
        checkpoint_path,
        float(threshold),
        model_id,
    )
    if _rad_dino_predictor is not None and getattr(_rad_dino_predictor, "_modal_cache_key", None) == cache_key:
        return _rad_dino_predictor

    from src.models.vision.dynamic_predictor import RadDinoGlobalPredictor

    predictor = RadDinoGlobalPredictor(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device="cuda",
        model_id=model_id,
    )
    predictor._modal_cache_key = cache_key
    _rad_dino_predictor = predictor
    return predictor

@app.function(image=image, secrets=[neo4j_secret], timeout=300)
def aura_counts() -> Dict[str, Any]:
    os.environ["PYTHONPATH"] = "/root:/root/src"

    from src.kg.client import Neo4jClient

    with Neo4jClient(_settings_from_env()) as client:
        client.verify_connectivity()
        counts = client.run_query(_counts_query())[0]

    return {"status": "ok", "counts": counts}

@app.function(
    image=image,
    gpu="A10G",
    secrets=[neo4j_secret, openai_secret, hf_secret],
    volumes={
        "/data/weights": vol_weights,
        "/data/dataset": vol_data,
    },
    timeout=1800,
)
def run_e2e(
    image_path: str = "",
    image_base64: str = "",
    image_filename: str = "upload.png",
    question: str = "What abnormalities are seen?",
    study_id: str = "modal_e2e_study",
    dicom_id: str = "modal_e2e_image",
    subject_id: str = "",
    patient_id: str = "",
    user_id: str = "",
    view: str = "",
    threshold: float = 0.5,
    anatomy_threshold: float = 0.5,
    min_confidence: float = 0.25,
    limit: int = 5,
    weights_path: str = "/data/weights/medvqa_vision_best.pth",
    sam_checkpoint: str = "/data/weights/sam3",
    encoder_backend: str = "transformers",
    model_id: str = "medvqa_vision_best",
    use_global: bool = True,
    global_checkpoint_path: str = "/data/weights/rad_dino_linear_adapter_best.pth",
    global_threshold: float = 0.5,
    global_model_id: str = "rad_dino_global",
    use_llm: bool = True,
    use_llm_router: bool = False,
) -> Dict[str, Any]:
    return _run_e2e_impl(
        image_path=image_path,
        image_base64=image_base64,
        image_filename=image_filename,
        question=question,
        study_id=study_id,
        dicom_id=dicom_id,
        subject_id=subject_id,
        patient_id=patient_id,
        user_id=user_id,
        view=view,
        threshold=threshold,
        anatomy_threshold=anatomy_threshold,
        min_confidence=min_confidence,
        limit=limit,
        weights_path=weights_path,
        sam_checkpoint=sam_checkpoint,
        encoder_backend=encoder_backend,
        model_id=model_id,
        use_global=use_global,
        global_checkpoint_path=global_checkpoint_path,
        global_threshold=global_threshold,
        global_model_id=global_model_id,
        use_llm=use_llm,
        use_llm_router=use_llm_router,
    )

def _build_answer_generator(use_llm: bool):
    if not use_llm or not os.getenv("VQA_LLM_MODEL"):
        return None

    from src.models.language.llm_answer_generator import OpenAICompatibleAnswerGenerator

    return OpenAICompatibleAnswerGenerator.from_env()

def _run_e2e_impl(
    image_path: str = "",
    image_base64: str = "",
    image_filename: str = "upload.png",
    question: str = "What abnormalities are seen?",
    study_id: str = "modal_e2e_study",
    dicom_id: str = "modal_e2e_image",
    subject_id: str = "",
    patient_id: str = "",
    user_id: str = "",
    view: str = "",
    threshold: float = 0.5,
    anatomy_threshold: float = 0.5,
    min_confidence: float = 0.25,
    limit: int = 5,
    weights_path: str = "/data/weights/medvqa_vision_best.pth",
    sam_checkpoint: str = "/data/weights/sam3",
    encoder_backend: str = "transformers",
    model_id: str = "medvqa_vision_best",
    use_global: bool = True,
    global_checkpoint_path: str = "/data/weights/rad_dino_linear_adapter_best.pth",
    global_threshold: float = 0.5,
    global_model_id: str = "rad_dino_global",
    use_llm: bool = True,
    use_llm_router: bool = False,
) -> Dict[str, Any]:
    os.environ["PYTHONPATH"] = "/root:/root/src"

    from src.kg.client import Neo4jClient
    from src.kg.ingest_dynamic import ingest_dynamic_entities
    from src.models.vqa_model import KnowledgeGraphVQAModel, RuleBasedQuestionParser

    resolved_image_path = _materialize_image_input(
        image_path=image_path,
        image_base64=image_base64,
        image_filename=image_filename,
    )
    predictor = _get_medsam_predictor(
        threshold=threshold,
        anatomy_threshold=anatomy_threshold,
        weights_path=weights_path,
        sam_checkpoint=sam_checkpoint,
        encoder_backend=encoder_backend,
        model_id=model_id,
    )
    resolved_patient_id = str(patient_id or user_id or subject_id or "")
    rows = predictor.predict(
        image_path=resolved_image_path,
        study_id=study_id,
        image_element_id=dicom_id,
        subject_id=resolved_patient_id,
        patient_id=resolved_patient_id,
        user_id=str(user_id or resolved_patient_id),
        view=view,
    )
    global_predictions = []
    global_rows = []
    if use_global:
        global_predictor = _get_rad_dino_predictor(
            checkpoint_path=global_checkpoint_path,
            threshold=global_threshold,
            model_id=global_model_id,
        )
        global_result = global_predictor.predict(
            image_path=resolved_image_path,
            study_id=study_id,
            image_element_id=dicom_id,
            subject_id=resolved_patient_id,
            patient_id=resolved_patient_id,
            user_id=str(user_id or resolved_patient_id),
            view=view,
        )
        global_predictions = global_result.get("predictions", [])
        global_rows = global_result.get("rows", [])
    hybrid_rows = [*global_rows, *rows]

    class DummyNeo4jClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def verify_connectivity(self):
            print("⚠️ Warning: Running in offline mock database mode (verify_connectivity skipped).")
        def run_query(self, query, parameters=None):
            print(f"⚠️ Warning: Mock query run: {query}")
            return []
        def execute_write(self, query, parameters=None):
            return []

    try:
        client_connection = Neo4jClient(_settings_from_env())
        client_connection.verify_connectivity()
    except Exception as conn_err:
        print(f"⚠️ Warning: Neo4j Connection failed. Falling back to DummyNeo4jClient. Error: {conn_err}")
        client_connection = DummyNeo4jClient()

    with client_connection as client:
        try:
            ingested_rows = ingest_dynamic_entities(client, hybrid_rows)
        except Exception as e:
            print(f"⚠️ Warning: Neo4j Ingestion failed (Read-only database fallback). Error: {e}")
            ingested_rows = []

        if use_llm_router:
            from src.models.vqa_model import OpenAIQuestionParser

            parser = OpenAIQuestionParser.from_env()
        else:
            parser = RuleBasedQuestionParser(
                diseases_csv_path="/root/data/label/normalized_diseases.csv",
                anatomy_csv_path="/root/data/label/normalized_anatomy.csv",
            )
        model = KnowledgeGraphVQAModel(
            kg_client=client,
            parser=parser,
            answer_generator=_build_answer_generator(use_llm=use_llm),
            min_confidence=min_confidence,
            limit=limit,
            require_static_backbone=True,
        )
        answer = model.answer(
            question=question,
            study_id=study_id,
            image_element_id=dicom_id,
            image_path=resolved_image_path,
            patient_id=resolved_patient_id,
            user_id=str(user_id or resolved_patient_id),
            hybrid_rows=hybrid_rows,
        )
        try:
            counts_res = client.run_query(_counts_query())
            counts = counts_res[0] if counts_res else {"patient_count": 0, "image_count": 0}
        except Exception:
            counts = {"patient_count": 0, "image_count": 0}

    return {
        "status": "ok",
        "image_path": resolved_image_path,
        "question": question,
        "study_id": study_id,
        "dicom_id": dicom_id,
        "patient_id": resolved_patient_id,
        "user_id": str(user_id or resolved_patient_id),
        "vision_row_count": len(rows),
        "global_row_count": len(global_rows),
        "hybrid_row_count": len(hybrid_rows),
        "global_predictions": global_predictions,
        "ingested_rows": ingested_rows,
        "answer": answer,
        "counts": counts,
    }

def _answer_from_payload(payload: E2EAnswerRequest) -> Dict[str, Any]:
    request_data = payload.model_dump()
    return _run_e2e_impl(
        image_path=str(request_data.get("image_path") or ""),
        image_base64=str(request_data.get("image_base64") or ""),
        image_filename=str(request_data.get("image_filename") or "upload.png"),
        question=str(request_data.get("question") or "What abnormalities are seen?"),
        study_id=str(request_data.get("study_id") or "modal_e2e_study"),
        dicom_id=str(request_data.get("dicom_id") or "modal_e2e_image"),
        subject_id=str(request_data.get("subject_id") or ""),
        patient_id=str(request_data.get("patient_id") or ""),
        user_id=str(request_data.get("user_id") or ""),
        view=str(request_data.get("view") or ""),
        threshold=float(request_data.get("threshold") or 0.5),
        anatomy_threshold=float(request_data.get("anatomy_threshold") or 0.5),
        min_confidence=float(request_data.get("min_confidence") or 0.25),
        limit=int(request_data.get("limit") or 5),
        use_global=bool(request_data.get("use_global", True)),
        global_threshold=float(request_data.get("global_threshold") or 0.5),
        use_llm=bool(request_data.get("use_llm", True)),
        use_llm_router=bool(request_data.get("use_llm_router", False)),
    )

def _patients_by_disease_from_payload(payload: PatientsByDiseaseRequest) -> Dict[str, Any]:
    os.environ["PYTHONPATH"] = "/root:/root/src"

    from src.kg.client import Neo4jClient
    from src.kg.queries import retrieve_patients_by_diseases_on_date

    request_data = payload.model_dump()
    with Neo4jClient(_settings_from_env()) as client:
        client.verify_connectivity()
        return retrieve_patients_by_diseases_on_date(
            client=client,
            diseases=request_data.get("diseases") or [],
            ingest_date=str(request_data.get("ingest_date") or "") or None,
            min_confidence=float(request_data.get("min_confidence") or 0.25),
            limit=int(request_data.get("limit") or 50),
        )

def _natural_language_kg_query_from_payload(payload: NaturalLanguageKGQueryRequest) -> Dict[str, Any]:
    os.environ["PYTHONPATH"] = "/root:/root/src"

    from src.kg.client import Neo4jClient
    from src.kg.queries import execute_routed_kg_query
    from src.models.language.llm_answer_generator import OpenAICompatibleQuestionRouter

    request_data = payload.model_dump()
    router = OpenAICompatibleQuestionRouter.from_env()
    route = router.route(
        question=str(request_data.get("question") or ""),
        current_date=str(request_data.get("current_date") or "") or None,
    )
    with Neo4jClient(_settings_from_env()) as client:
        client.verify_connectivity()
        return execute_routed_kg_query(
            client=client,
            route=route,
            min_confidence=float(request_data.get("min_confidence") or 0.25),
            limit=int(request_data.get("limit") or 50),
        )

@app.function(
    image=image,
    gpu="A10G",
    secrets=[neo4j_secret, openai_secret, hf_secret],
    volumes={
        "/data/weights": vol_weights,
        "/data/dataset": vol_data,
    },
    timeout=1800,
)
@modal.asgi_app(label="answer")
def answer_app():
    from fastapi import FastAPI

    api = FastAPI(title="MedVQA E2E")

    @api.post("/")
    async def answer(payload: E2EAnswerRequest) -> Dict[str, Any]:
        return _answer_from_payload(payload)

    @api.post("/patients/by-disease")
    async def patients_by_disease(payload: PatientsByDiseaseRequest) -> Dict[str, Any]:
        return _patients_by_disease_from_payload(payload)

    @api.post("/query")
    async def natural_language_query(payload: NaturalLanguageKGQueryRequest) -> Dict[str, Any]:
        return _natural_language_kg_query_from_payload(payload)

    @api.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    return api

@app.local_entrypoint()
def main(
    image_path: str = "/data/dataset/mimic-cxr-kaggle/images/00000218-9fb20d4e-86045713-8013e08b-0d5bebba.jpg",
    question: str = "What abnormalities are seen?",
    study_id: str = "modal_e2e_study",
    dicom_id: str = "modal_e2e_image",
    patient_id: str = "",
    check_only: bool = False,
    use_llm: bool = True,
    use_llm_router: bool = False,
    use_global: bool = True,
    global_threshold: float = 0.5,
):
    if check_only:
        result = aura_counts.remote()
    else:
        result = run_e2e.remote(
            image_path=image_path,
            question=question,
            study_id=study_id,
            dicom_id=dicom_id,
            patient_id=patient_id,
            use_global=use_global,
            global_threshold=global_threshold,
            use_llm=use_llm,
            use_llm_router=use_llm_router,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
