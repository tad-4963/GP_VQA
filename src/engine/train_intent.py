import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.datasets.cxr_vqa_dataset import MedicalCXRVQADataset
from src.models.language.albert_bilstm import MultiTaskALBERTBiLSTM


class IntentNerJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        if not self.records:
            raise ValueError(f"No records found in BIO jsonl: {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "intent_label": torch.tensor(int(row.get("intent_label", 0)), dtype=torch.long),
            "ner_tag_ids": torch.tensor(row.get("ner_tag_ids", []), dtype=torch.long),
            "dicom_id": str(row.get("dicom_id", "unknown")),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: MedicalCXRVQADataset, val_size: float, seed: int):
    labels = dataset.df["question_type"].map(dataset.intent_map).fillna(0).astype(int).to_numpy()
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    return train_subset, val_subset, labels[train_idx]


def build_class_weights(train_labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=device)


def resolve_csv_path(csv_path: str) -> Path:
    direct = Path(csv_path) if csv_path else None
    if direct and direct.exists():
        return direct

    candidates = [
        ROOT_DIR / "data/medical_cxr/medical-cxr-vqa-questions_final.csv",
        ROOT_DIR / "data/medical_cxr/filtered_medical-cxr-vqa-questions.csv",
        ROOT_DIR / "data/medical_cxr/medical-cxr-vqa-questions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find intent CSV. Pass --csv-path or place the file under data/medical_cxr/."
    )


def resolve_bio_jsonl_path(bio_jsonl_path: str) -> Path:
    direct = Path(bio_jsonl_path) if bio_jsonl_path else None
    if direct and direct.exists():
        return direct

    candidates = [
        ROOT_DIR / "data/medical_cxr/vqa_bio_dataset_questions_final.jsonl",
        ROOT_DIR / "data/medical_cxr/vqa_bio_dataset_final.jsonl",
        ROOT_DIR / "data/vqa_bio_dataset_questions_final.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find BIO jsonl. Pass --bio-jsonl-path or generate BIO dataset first."
    )


def run_epoch(
    model,
    dataloader,
    criterion_intent,
    criterion_ner,
    optimizer,
    scheduler,
    device,
    train_ner: bool,
    alpha_ner: float,
    train_mode: bool,
    epoch: int,
    total_epochs: int,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_ner_loss = 0.0
    total_ner_tokens = 0
    total_ner_correct = 0

    loop_name = "Training" if train_mode else "Validating"

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        pbar = tqdm(
            dataloader,
            desc=f"{loop_name} Epoch {epoch}/{total_epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=False,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_intent = batch["intent_label"].to(device)

            if train_ner:
                logits_intent, logits_ner = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_ner=True,
                )
            else:
                logits_intent = model(input_ids=input_ids, attention_mask=attention_mask, return_ner=False)
                logits_ner = None

            loss_intent = criterion_intent(logits_intent, labels_intent)
            if train_ner:
                labels_ner = batch["ner_tag_ids"].to(device)
                token_loss = F.cross_entropy(
                    logits_ner.view(-1, logits_ner.size(-1)),
                    labels_ner.view(-1),
                    reduction="none",
                )
                token_mask = attention_mask.view(-1).float()
                loss_ner = (token_loss * token_mask).sum() / token_mask.sum().clamp_min(1.0)
                loss = loss_intent + alpha_ner * loss_ner
            else:
                loss_ner = None
                loss = loss_intent

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            preds_intent = torch.argmax(logits_intent, dim=-1)
            total_correct += (preds_intent == labels_intent).sum().item()
            total_loss += loss.item() * labels_intent.size(0)
            total_samples += labels_intent.size(0)

            if train_ner:
                token_preds = torch.argmax(logits_ner, dim=-1)
                token_mask = attention_mask.bool()
                token_correct = ((token_preds == labels_ner) & token_mask).sum().item()
                token_count = token_mask.sum().item()
                total_ner_correct += token_correct
                total_ner_tokens += token_count
                total_ner_loss += loss_ner.item() * labels_intent.size(0)

            avg_loss_so_far = total_loss / max(total_samples, 1)
            avg_acc_so_far = total_correct / max(total_samples, 1)
            if train_ner:
                avg_ner_loss_so_far = total_ner_loss / max(total_samples, 1)
                avg_ner_acc_so_far = total_ner_correct / max(total_ner_tokens, 1)
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss_so_far:.4f}",
                        "acc": f"{avg_acc_so_far:.4f}",
                        "ner_loss": f"{avg_ner_loss_so_far:.4f}",
                        "ner_acc": f"{avg_ner_acc_so_far:.4f}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss_so_far:.4f}",
                        "acc": f"{avg_acc_so_far:.4f}",
                    }
                )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    avg_ner_loss = total_ner_loss / max(total_samples, 1) if train_ner else 0.0
    avg_ner_acc = total_ner_correct / max(total_ner_tokens, 1) if train_ner else 0.0
    return avg_loss, avg_acc, avg_ner_loss, avg_ner_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train ALBERT+BiLSTM for question intent classification")
    parser.add_argument("--csv-path", type=str, default="data/medical_cxr/medical-cxr-vqa-questions_final.csv")
    parser.add_argument("--output-dir", type=str, default="weights/intent_classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ner", action="store_true")
    parser.add_argument("--bio-jsonl-path", type=str, default="data/medical_cxr/vqa_bio_dataset_questions_final.jsonl")
    parser.add_argument("--alpha-ner", type=float, default=0.5)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    resolved_csv: Optional[Path] = None
    resolved_bio: Optional[Path] = None

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_ner:
        resolved_bio = resolve_bio_jsonl_path(args.bio_jsonl_path)
        dataset = IntentNerJsonlDataset(str(resolved_bio))
        labels = np.array([int(r.get("intent_label", 0)) for r in dataset.records], dtype=np.int64)
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=labels,
        )
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())
        train_labels = labels[train_idx]
        max_ner_tag_id = 0
        for record in dataset.records:
            tag_ids = record.get("ner_tag_ids", [0])
            if isinstance(tag_ids, list) and tag_ids:
                max_ner_tag_id = max(max_ner_tag_id, int(max(tag_ids)))
        num_ner_tags = max_ner_tag_id + 1
    else:
        resolved_csv = resolve_csv_path(args.csv_path)
        dataset = MedicalCXRVQADataset(csv_file_path=str(resolved_csv), max_length=args.max_length)
        train_ds, val_ds, train_labels = split_dataset(dataset, val_size=args.val_size, seed=args.seed)
        num_ner_tags = 10

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.train_ner:
        num_intents = int(np.max(train_labels)) + 1
    else:
        num_intents = len({v for v in dataset.intent_map.values()})
    if hasattr(dataset, "intent_map"):
        intent_map_to_save = dataset.intent_map
    else:
        intent_map_to_save = {f"intent_{idx}": idx for idx in range(num_intents)}
    model = MultiTaskALBERTBiLSTM(num_intents=num_intents, num_ner_tags=num_ner_tags).to(device)

    if args.no_class_weights:
        class_weights = None
    else:
        class_weights = build_class_weights(train_labels, num_classes=num_intents, device=device)

    criterion_intent = nn.CrossEntropyLoss(weight=class_weights)
    criterion_ner = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_val_acc = 0.0
    best_model_path = output_dir / "best_intent_model.pt"

    print(f"Dataset size: {len(dataset)}")
    if resolved_csv is not None:
        print(f"CSV source: {resolved_csv}")
    if resolved_bio is not None:
        print(f"BIO source: {resolved_bio}")
    print(f"Train NER: {args.train_ner} | alpha_ner={args.alpha_ner}")
    print(f"Train/Val split: {len(train_ds)}/{len(val_ds)}")
    print(f"Train steps/epoch: {len(train_loader)} | Val steps/epoch: {len(val_loader)}")
    print("Label counts (cleaned):")
    if hasattr(dataset, "get_label_counts"):
        print(dataset.get_label_counts().to_string())
    else:
        label_counts = np.bincount(train_labels, minlength=num_intents)
        for cls_idx, count in enumerate(label_counts.tolist()):
            print(f"intent_{cls_idx}: {count}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)
        train_loss, train_acc, train_ner_loss, train_ner_acc = run_epoch(
            model,
            train_loader,
            criterion_intent,
            criterion_ner,
            optimizer,
            scheduler,
            device,
            train_ner=args.train_ner,
            alpha_ner=args.alpha_ner,
            train_mode=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_acc, val_ner_loss, val_ner_acc = run_epoch(
            model,
            val_loader,
            criterion_intent,
            criterion_ner,
            optimizer,
            scheduler,
            device,
            train_ner=args.train_ner,
            alpha_ner=args.alpha_ner,
            train_mode=False,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        if args.train_ner:
            row["train_ner_loss"] = round(train_ner_loss, 6)
            row["train_ner_acc"] = round(train_ner_acc, 6)
            row["val_ner_loss"] = round(val_ner_loss, 6)
            row["val_ner_acc"] = round(val_ner_acc, 6)
        history.append(row)
        if args.train_ner:
            print(
                f"KET QUA EPOCH {epoch}: "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train NER Loss: {train_ner_loss:.4f} | Train NER Acc: {train_ner_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val NER Loss: {val_ner_loss:.4f} | Val NER Acc: {val_ner_acc:.4f}",
                flush=True,
            )
        else:
            print(
                f"KET QUA EPOCH {epoch}: "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
                flush=True,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "intent_map": intent_map_to_save,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"Saved best checkpoint -> {best_model_path}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Training log saved -> {metrics_path}")


if __name__ == "__main__":
    main()
