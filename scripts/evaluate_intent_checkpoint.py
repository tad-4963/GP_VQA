import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.evaluate import evaluate_prediction_rows, write_evaluation_report
from src.engine.train_intent import IntentNerJsonlDataset
from src.models.language.albert_bilstm import MultiTaskALBERTBiLSTM

NER_TAGS = {
    0: "O",
    1: "B-DISEASE",
    2: "I-DISEASE",
    3: "B-ANATOMY",
    4: "I-ANATOMY",
}

INTENT_ID_TO_NAME = {
    0: "presence",
    1: "abnormality",
    2: "location",
    3: "view",
    4: "type",
    5: "level",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ALBERT-BiLSTM intent checkpoint on BIO JSONL split")
    parser.add_argument("--checkpoint", default="weights/best_intent_model.pt")
    parser.add_argument("--bio-jsonl", default="data/medical_cxr/vqa_bio_dataset_questions_final.jsonl")
    parser.add_argument("--output-dir", default="outputs/evaluation/language_checkpoint_val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all validation samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _infer_head_dims(state_dict):
    num_intents = int(state_dict["intent_classifier.2.weight"].shape[0])
    num_ner_tags = int(state_dict["ner_classifier.2.weight"].shape[0])
    return num_intents, num_ner_tags


def _load_val_subset(dataset, val_size, seed):
    labels = np.array([int(record.get("intent_label", 0)) for record in dataset.records], dtype=np.int64)
    indices = np.arange(len(dataset))
    _, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )
    return val_idx.tolist()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    ckpt_args = checkpoint.get("args", {})
    val_size = float(ckpt_args.get("val_size", 0.1))
    seed = int(ckpt_args.get("seed", 42))
    max_length = int(ckpt_args.get("max_length", 64))
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    dataset = IntentNerJsonlDataset(args.bio_jsonl)
    val_indices = _load_val_subset(dataset, val_size=val_size, seed=seed)
    if args.max_samples > 0:
        val_indices = val_indices[: min(args.max_samples, len(val_indices))]
    val_subset = Subset(dataset, val_indices)

    dataloader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    num_intents, num_ner_tags = _infer_head_dims(state_dict)
    model = MultiTaskALBERTBiLSTM(num_intents=num_intents, num_ner_tags=num_ner_tags).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rows = []
    dataset_records = dataset.records
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            intent_logits, ner_logits = model(input_ids=input_ids, attention_mask=attention_mask, return_ner=True)

            intent_pred = torch.argmax(intent_logits, dim=-1).detach().cpu().tolist()
            intent_true = batch["intent_label"].detach().cpu().tolist()
            ner_pred = torch.argmax(ner_logits, dim=-1).detach().cpu().tolist()
            ner_true = batch["ner_tag_ids"].detach().cpu().tolist()
            masks = attention_mask.detach().cpu().tolist()

            batch_size = len(intent_pred)
            for batch_index in range(batch_size):
                dataset_index = val_indices[offset + batch_index]
                record = dataset_records[dataset_index]
                active_len = int(sum(masks[batch_index]))

                rows.append(
                    {
                        "subject_id": record.get("subject_id"),
                        "study_id": record.get("study_id"),
                        "dicom_id": record.get("dicom_id"),
                        "question": record.get("question"),
                        "split": record.get("split"),
                        "intent_true": record.get("question_type"),
                        "intent_pred": INTENT_ID_TO_NAME.get(intent_pred[batch_index], f"intent_{intent_pred[batch_index]}"),
                        "answer_true": record.get("answer"),
                        "entities_true": record.get("entities_grouped"),
                        "ner_tag_ids_true": ner_true[batch_index][:active_len],
                        "ner_tag_ids_pred": ner_pred[batch_index][:active_len],
                        "ner_tags_true": [NER_TAGS.get(tag, f"TAG_{tag}") for tag in ner_true[batch_index][:active_len]],
                        "ner_tags_pred": [NER_TAGS.get(tag, f"TAG_{tag}") for tag in ner_pred[batch_index][:active_len]],
                        "tokens": tokenizer.convert_ids_to_tokens(input_ids[batch_index].detach().cpu().tolist())[:active_len],
                    }
                )
            offset += batch_size

    predictions_path = output_dir / "intent_checkpoint_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = evaluate_prediction_rows(rows)
    paths = write_evaluation_report(report, output_dir, stem="evaluation_report")
    summary = {
        "sample_count": len(rows),
        "intent_accuracy": report.get("sections", {}).get("intent", {}).get("intent_accuracy"),
        "ner_token_accuracy": report.get("sections", {}).get("ner_token", {}).get("token_accuracy"),
        "ner_token_f1": report.get("sections", {}).get("ner_token", {}).get("token_f1"),
    }
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "predictions": str(predictions_path),
                "reports": {key: str(value) for key, value in paths.items()},
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
