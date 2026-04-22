import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.language.clinical_ner import ClinicalEntityExtractor


INTENT_MAP = {
    "presence": 0,
    "abnormality": 1,
    "location": 2,
    "view": 3,
    "type": 4,
    "level": 5,
}

NER_TAG_MAP = {
    "O": 0,
    "B-DISEASE": 1,
    "I-DISEASE": 2,
    "B-ANATOMY": 3,
    "I-ANATOMY": 4,
}


def resolve_csv_path(csv_path: str) -> Path:
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return path

    candidates = [
        ROOT_DIR / "data/medical_cxr/medical-cxr-vqa-questions-final.csv",
        ROOT_DIR / "data/medical_cxr/medical-cxr-vqa-questions_final.csv",
        ROOT_DIR / "data/medical_cxr/medical-cxr-vqa-questions.csv",
        ROOT_DIR / "data/medical_cxr/filtered_medical-cxr-vqa-questions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Khong tim thay CSV dau vao. Hay truyen --csv-path hoac them file vao data/medical_cxr/."
    )


def _token_inside_entity(tok_start: int, tok_end: int, ent_start: int, ent_end: int) -> bool:
    return tok_start >= ent_start and tok_end <= ent_end


def build_bio_tags(
    offset_mapping: List[Tuple[int, int]],
    entities: List[Dict[str, object]],
) -> List[str]:
    tags = ["O"] * len(offset_mapping)

    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue

        for ent in entities:
            ent_start = int(ent["start"])
            ent_end = int(ent["end"])
            ent_label = str(ent["label"])
            if ent_label not in ("DISEASE", "ANATOMY"):
                continue

            if _token_inside_entity(tok_start, tok_end, ent_start, ent_end):
                prev = tags[i - 1] if i > 0 else "O"
                if prev.endswith(ent_label):
                    tags[i] = f"I-{ent_label}"
                else:
                    tags[i] = f"B-{ent_label}"
                break

    return tags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BIO NER dataset from normalized disease/anatomy labels")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/medical_cxr/medical-cxr-vqa-questions-final.csv",
        help="Input questions CSV path",
    )
    parser.add_argument(
        "--diseases-csv",
        type=str,
        default="data/label/normalized_diseases.csv",
        help="Normalized diseases CSV",
    )
    parser.add_argument(
        "--anatomy-csv",
        type=str,
        default="data/label/normalized_anatomy.csv",
        help="Normalized anatomy CSV",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="albert-base-v2",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/medical_cxr/vqa_bio_dataset_final.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row cap for quick debugging (0 = all rows)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv_path)
    diseases_csv = (ROOT_DIR / args.diseases_csv).resolve() if not Path(args.diseases_csv).is_absolute() else Path(args.diseases_csv)
    anatomy_csv = (ROOT_DIR / args.anatomy_csv).resolve() if not Path(args.anatomy_csv).is_absolute() else Path(args.anatomy_csv)

    if not diseases_csv.exists() or not anatomy_csv.exists():
        raise FileNotFoundError(
            f"Khong tim thay normalized labels: diseases={diseases_csv} anatomy={anatomy_csv}"
        )

    print(f"[INFO] CSV input: {csv_path}")
    print(f"[INFO] Disease labels: {diseases_csv}")
    print(f"[INFO] Anatomy labels: {anatomy_csv}")

    df = pd.read_csv(csv_path)
    if "question" not in df.columns:
        raise ValueError("CSV phai co cot 'question'")

    if args.limit > 0:
        df = df.head(args.limit)

    extractor = ClinicalEntityExtractor(str(diseases_csv), str(anatomy_csv))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    output_path = (ROOT_DIR / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    matched_rows = 0

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=total_rows, desc="Generating BIO"):
            question = str(row.get("question", "")).strip()
            if not question:
                question = "unknown"

            entities = extractor.extract(question)
            grouped = extractor.extract_grouped(question)

            encoded = tokenizer(
                question,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_offsets_mapping=True,
            )

            offsets: List[Tuple[int, int]] = [tuple(pair) for pair in encoded["offset_mapping"]]
            bio_tags = build_bio_tags(offsets, entities)
            bio_tag_ids = [NER_TAG_MAP.get(tag, 0) for tag in bio_tags]

            if grouped["DISEASE"] or grouped["ANATOMY"]:
                matched_rows += 1

            q_type = str(row.get("question_type", "presence")).strip().lower()
            intent_label = INTENT_MAP.get(q_type, 0)

            record = {
                "subject_id": str(row.get("subject_id", "")),
                "study_id": str(row.get("study_id", "")),
                "dicom_id": str(row.get("dicom_id", "")),
                "question": question,
                "question_type": q_type,
                "intent_label": intent_label,
                "answer": str(row.get("answer", "")),
                "split": str(row.get("split", "")),
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "ner_tags": bio_tags,
                "ner_tag_ids": bio_tag_ids,
                "entities": entities,
                "entities_grouped": grouped,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    coverage = matched_rows / max(total_rows, 1)
    meta = {
        "input_csv": str(csv_path),
        "output_jsonl": str(output_path),
        "rows": total_rows,
        "rows_with_entity": matched_rows,
        "coverage": round(coverage, 6),
        "max_length": args.max_length,
        "tokenizer": args.tokenizer,
        "ner_tag_map": NER_TAG_MAP,
        "intent_map": INTENT_MAP,
    }

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    print("[DONE] Generated BIO dataset")
    print(json.dumps(meta, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
