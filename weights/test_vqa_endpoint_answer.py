import argparse
import base64
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


DEFAULT_ENDPOINT_URL = "https://dungkidx01--answer.modal.run/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the deployed MedVQA endpoint and print only the answer."
    )
    parser.add_argument(
        "image",
        help=(
            "Local image path, Modal image path, or HTTP(S) image URL. "
            "Local/HTTP images are uploaded as base64; Modal paths are sent as image_path."
        ),
    )
    parser.add_argument(
        "--question",
        default="What abnormalities are seen?",
        help="Question to ask about the image.",
    )
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT_URL)
    parser.add_argument("--study-id", default="")
    parser.add_argument("--dicom-id", default="")
    parser.add_argument("--patient-id", default="", help="Optional patient/user id to attach to the uploaded image.")
    parser.add_argument("--user-id", default="", help="Alias for patient id; useful for app/user-facing payloads.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--anatomy-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--show-explanation",
        action="store_true",
        help="Print the explanation after the answer.",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict:
    image = args.image
    study_id = args.study_id or f"test_study_{int(time.time())}"
    dicom_id = args.dicom_id or f"test_image_{int(time.time())}"

    payload = {
        "question": args.question,
        "study_id": study_id,
        "dicom_id": dicom_id,
        "patient_id": args.patient_id or args.user_id,
        "user_id": args.user_id or args.patient_id,
        "use_llm": not args.no_llm,
        "threshold": args.threshold,
        "anatomy_threshold": args.anatomy_threshold,
        "min_confidence": args.min_confidence,
        "limit": args.limit,
    }

    local_path = Path(image)
    if local_path.exists():
        payload["image_base64"] = base64.b64encode(local_path.read_bytes()).decode("ascii")
        payload["image_filename"] = local_path.name
        return payload

    if image.startswith(("http://", "https://")):
        content, filename = download_image(image)
        payload["image_base64"] = base64.b64encode(content).decode("ascii")
        payload["image_filename"] = filename
        return payload

    payload["image_path"] = image
    return payload


def download_image(url: str) -> tuple[bytes, str]:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    parsed = urlparse(url)
    filename = Path(parsed.path).name or "image.jpg"
    return response.content, filename


def extract_answer(response_json: dict) -> str:
    answer = response_json.get("answer", {})
    if isinstance(answer, dict):
        return str(answer.get("answer", ""))
    return str(answer or "")


def extract_explanation(response_json: dict) -> str:
    answer = response_json.get("answer", {})
    if isinstance(answer, dict):
        return str(answer.get("explanation", ""))
    return ""


def main() -> int:
    args = parse_args()
    payload = build_payload(args)

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(args.endpoint_url, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = extract_answer(data)
        explanation = extract_explanation(data)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(answer)
    if args.show_explanation and explanation:
        print(explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
