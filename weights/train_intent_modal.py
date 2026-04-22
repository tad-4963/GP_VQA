import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("med-vqa-intent-training")

vol_weights = modal.Volume.from_name("med-vqa-weights")
vol_data = modal.Volume.from_name("med-vqa-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy<2",
        "tqdm",
    )
    .add_local_dir("/home/laptopdev/GP_VQA/src", remote_path="/root/src")
)


def resolve_csv_path(csv_path: str) -> str:
    direct = Path(csv_path) if csv_path else None
    if direct and direct.exists():
        return str(direct)

    candidates = [
        "/data/dataset/medical_cxr/medical-cxr-vqa-questions_final.csv",
        "/data/dataset/medical_cxr/filtered_medical-cxr-vqa-questions.csv",
        "/data/dataset/medical_cxr/medical-cxr-vqa-questions.csv",
        "/data/dataset/medical-cxr-vqa-questions_final.csv",
        "/data/dataset/filtered_medical-cxr-vqa-questions.csv",
        "/data/dataset/medical-cxr-vqa-questions.csv",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Could not find CSV on /data/dataset. "
        "Pass --csv-path to a valid file path inside mounted volume."
    )


def resolve_bio_jsonl_path(bio_jsonl_path: str) -> str:
    direct = Path(bio_jsonl_path) if bio_jsonl_path else None
    if direct and direct.exists():
        return str(direct)

    candidates = [
        "/data/dataset/medical_cxr/vqa_bio_dataset_questions_final.jsonl",
        "/data/dataset/vqa_bio_dataset_questions_final.jsonl",
        "/data/dataset/medical_cxr/vqa_bio_dataset_final.jsonl",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Could not find BIO JSONL on /data/dataset. "
        "Pass --bio-jsonl-path to a valid file path inside mounted volume."
    )


def stream_subprocess(command: list[str], env: dict[str, str], log_file_path: Path) -> int:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to file: {log_file_path}")
    print(f"[INFO] Command: {' '.join(shlex.quote(part) for part in command)}")

    with log_file_path.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=env,
        )

        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            text_chunk = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text_chunk)
            sys.stdout.flush()
            # Replace carriage returns so the persisted log file remains readable.
            log_f.write(text_chunk.replace("\r", "\n"))
            log_f.flush()

        return process.wait()


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data/weights": vol_weights,
        "/data/dataset": vol_data,
    },
    timeout=86400,
)
def train_intent_modal(
    csv_path: str = "",
    bio_jsonl_path: str = "",
    epochs: int = 4,
    batch_size: int = 128,
    max_length: int = 64,
    lr: float = 2e-5,
    val_size: float = 0.1,
    warmup_ratio: float = 0.1,
    num_workers: int = 8,
    seed: int = 42,
    train_ner: bool = True,
    alpha_ner: float = 0.5,
    no_class_weights: bool = False,
    device: str = "cuda",
):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resolved_csv = resolve_csv_path(csv_path)
    resolved_bio_jsonl = resolve_bio_jsonl_path(bio_jsonl_path) if train_ner else ""

    output_root = Path("/data/weights/intent_classifier")
    run_output_dir = output_root / f"run_{run_id}"
    logs_dir = output_root / "logs"
    log_file_path = logs_dir / f"train_intent_{run_id}.log"
    meta_file_path = logs_dir / f"train_intent_{run_id}.meta.json"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-u",
        "/root/src/engine/train_intent.py",
        "--csv-path",
        resolved_csv,
        "--output-dir",
        str(run_output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--lr",
        str(lr),
        "--val-size",
        str(val_size),
        "--warmup-ratio",
        str(warmup_ratio),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if train_ner:
        cmd.extend([
            "--train-ner",
            "--bio-jsonl-path",
            resolved_bio_jsonl,
            "--alpha-ner",
            str(alpha_ner),
        ])
    if no_class_weights:
        cmd.append("--no-class-weights")

    run_meta = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": resolved_csv,
        "output_dir": str(run_output_dir),
        "log_file": str(log_file_path),
        "args": {
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "lr": lr,
            "val_size": val_size,
            "warmup_ratio": warmup_ratio,
            "num_workers": num_workers,
            "seed": seed,
            "train_ner": train_ner,
            "bio_jsonl_path": resolved_bio_jsonl,
            "alpha_ner": alpha_ner,
            "no_class_weights": no_class_weights,
            "device": device,
        },
    }
    meta_file_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TQDM_DISABLE", "0")

    print("=" * 80)
    print(f"[INFO] Start Modal intent training | run_id={run_id}")
    print(f"[INFO] CSV: {resolved_csv}")
    print(f"[INFO] Output dir: {run_output_dir}")
    print("=" * 80)

    return_code = stream_subprocess(cmd, env=env, log_file_path=log_file_path)
    if return_code != 0:
        raise RuntimeError(
            f"Intent training failed with return code {return_code}. "
            f"Check log at {log_file_path}."
        )

    metrics_path = run_output_dir / "metrics.json"
    best_model_path = run_output_dir / "best_intent_model.pt"
    ended_at = datetime.now(timezone.utc).isoformat()

    if metrics_path.exists():
        print(f"[INFO] Metrics file ready: {metrics_path}")
    else:
        print(f"[WARN] Metrics file missing: {metrics_path}")

    if best_model_path.exists():
        print(f"[INFO] Checkpoint ready: {best_model_path}")
    else:
        print(f"[WARN] Checkpoint missing: {best_model_path}")

    run_meta["ended_at_utc"] = ended_at
    run_meta["status"] = "success"
    run_meta["metrics_path"] = str(metrics_path)
    run_meta["best_model_path"] = str(best_model_path)
    meta_file_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")

    print("[INFO] Intent training completed successfully.")
    return {
        "run_id": run_id,
        "csv_path": resolved_csv,
        "output_dir": str(run_output_dir),
        "best_model_path": str(best_model_path),
        "metrics_path": str(metrics_path),
        "log_file": str(log_file_path),
        "meta_file": str(meta_file_path),
    }


@app.local_entrypoint()
def main(
    csv_path: str = "",
    bio_jsonl_path: str = "",
    epochs: int = 4,
    batch_size: int = 128,
    max_length: int = 64,
    lr: float = 2e-5,
    val_size: float = 0.1,
    warmup_ratio: float = 0.1,
    num_workers: int = 8,
    seed: int = 42,
    train_ner: int = 1,
    alpha_ner: float = 0.5,
    no_class_weights: int = 0,
    device: str = "cuda",
):
    result = train_intent_modal.remote(
        csv_path=csv_path,
        bio_jsonl_path=bio_jsonl_path,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        lr=lr,
        val_size=val_size,
        warmup_ratio=warmup_ratio,
        num_workers=num_workers,
        seed=seed,
        train_ner=bool(train_ner),
        alpha_ner=alpha_ner,
        no_class_weights=bool(no_class_weights),
        device=device,
    )
    print("[DONE]", json.dumps(result, ensure_ascii=True, indent=2))
