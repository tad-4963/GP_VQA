import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vqa_model import KnowledgeGraphVQAModel, OpenAIQuestionParser, RuleBasedQuestionParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline KG-grounded VQA demo")
    parser.add_argument("--question", required=True)
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--dicom-id", default=None, help="ImageElement.element_id / dicom_id")
    parser.add_argument("--patient-id", default=None, help="Patient/user id to attach to the image")
    parser.add_argument("--user-id", default=None, help="Alias for --patient-id")
    parser.add_argument("--image-path", default=None, help="Stored for output only in this baseline")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="MedVQA2026!")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--diseases-csv", default="data/label/normalized_diseases.csv")
    parser.add_argument("--anatomy-csv", default="data/label/normalized_anatomy.csv")
    parser.add_argument(
        "--question-parser",
        choices=["rule", "openai-router"],
        default="rule",
        help="Use OpenAI router instead of the local rule parser for image VQA intent/entity parsing.",
    )
    parser.add_argument("--no-static-backbone-check", action="store_true")
    parser.add_argument("--run-vision", action="store_true", help="Predict and ingest dynamic findings from --image-path before KG retrieval")
    parser.add_argument("--vision-weights", default="weights/medvqa_vision_best.pth")
    parser.add_argument("--sam-checkpoint", default="/data/weights/sam3")
    parser.add_argument("--vision-threshold", type=float, default=0.5)
    parser.add_argument("--anatomy-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--encoder-backend", default="transformers")
    parser.add_argument("--model-id", default="medvqa_vision_best")
    parser.add_argument(
        "--llm-provider",
        choices=["none", "openai-compatible"],
        default="none",
        help="Optional final answer generator. openai-compatible reads VQA_LLM_MODEL and OPENAI_API_KEY from env.",
    )
    return parser.parse_args()


def main() -> int:
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    args = parse_args()

    try:
        from src.kg.client import Neo4jClient, Neo4jSettings
    except ModuleNotFoundError as exc:
        if exc.name == "neo4j":
            raise SystemExit(
                "Missing dependency: neo4j. Run the demo with the repo virtualenv, "
                "for example: venv/bin/python scripts/run_vqa_demo.py ..."
            ) from exc
        raise

    settings = Neo4jSettings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )
    if args.question_parser == "openai-router":
        parser = OpenAIQuestionParser.from_env()
    else:
        parser = RuleBasedQuestionParser(
            diseases_csv_path=args.diseases_csv,
            anatomy_csv_path=args.anatomy_csv,
        )
    vision_predictor = None
    if args.run_vision:
        if not args.image_path:
            raise SystemExit("--run-vision requires --image-path")
        from src.models.vision.dynamic_predictor import MedVQAVisionDynamicPredictor

        try:
            vision_predictor = MedVQAVisionDynamicPredictor(
                weights_path=args.vision_weights,
                sam_checkpoint=args.sam_checkpoint,
                threshold=args.vision_threshold,
                anatomy_threshold=args.anatomy_threshold,
                device=args.device,
                encoder_backend=args.encoder_backend,
                model_id=args.model_id,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(f"Vision initialization failed: {exc}") from exc

    answer_generator = None
    if args.llm_provider == "openai-compatible":
        from src.models.language.llm_answer_generator import OpenAICompatibleAnswerGenerator

        answer_generator = OpenAICompatibleAnswerGenerator.from_env()

    with Neo4jClient(settings) as client:
        client.verify_connectivity()
        model = KnowledgeGraphVQAModel(
            kg_client=client,
            parser=parser,
            answer_generator=answer_generator,
            min_confidence=args.min_confidence,
            limit=args.limit,
            require_static_backbone=not args.no_static_backbone_check,
            vision_predictor=vision_predictor,
        )
        result = model.answer(
            question=args.question,
            study_id=args.study_id,
            image_element_id=args.dicom_id,
            image_path=args.image_path,
            patient_id=args.patient_id or args.user_id,
            user_id=args.user_id or args.patient_id,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
