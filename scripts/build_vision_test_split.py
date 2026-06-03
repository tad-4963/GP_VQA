from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd


DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a subject-disjoint vision holdout CSV from images that exist on Modal "
            "but are not used in the current train/validate CSV."
        )
    )
    parser.add_argument(
        "--final-csv",
        default="data/medical_cxr/mimic_all_final.csv",
        help="Current train/validate CSV to exclude from the holdout.",
    )
    parser.add_argument(
        "--existing-csv",
        default="data/medical_cxr/mimic_all_existing_images.csv",
        help="CSV containing all image rows known to exist on Modal.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/medical_cxr/mimic_vision_test_subject_disjoint.csv",
        help="Output holdout CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path. Defaults to <output-csv>.summary.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional deterministic sample size. Use 0 to keep all eligible rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-subject-overlap",
        action="store_true",
        help="Only exclude DICOM overlap. By default, subjects from final CSV are also excluded.",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        help="Split value to write into the output CSV.",
    )
    return parser.parse_args(argv)


def _label_summary(df: pd.DataFrame, labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for label in labels:
        if label not in df.columns:
            continue
        values = pd.to_numeric(df[label], errors="coerce")
        summary[label] = {
            "non_null": int(values.notna().sum()),
            "positive": int((values == 1).sum()),
            "negative": int((values == 0).sum()),
            "uncertain": int((values == -1).sum()),
            "missing": int(values.isna().sum()),
        }
    return summary


def build_holdout(args: argparse.Namespace) -> Dict[str, Any]:
    final_path = Path(args.final_csv)
    existing_path = Path(args.existing_csv)
    output_path = Path(args.output_csv)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )

    final_df = pd.read_csv(final_path)
    existing_df = pd.read_csv(existing_path)

    required = {"subject_id", "dicom_id", "split"}
    for name, df in (("final", final_df), ("existing", existing_df)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} CSV is missing required columns: {missing}")

    final_dicoms = set(final_df["dicom_id"].astype(str))
    final_subjects = set(final_df["subject_id"].astype(str))

    candidate_mask = ~existing_df["dicom_id"].astype(str).isin(final_dicoms)
    if not args.allow_subject_overlap:
        candidate_mask &= ~existing_df["subject_id"].astype(str).isin(final_subjects)

    holdout = existing_df.loc[candidate_mask].copy()
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(holdout):
        holdout = holdout.sample(n=args.max_samples, random_state=args.seed).sort_index()

    holdout["split"] = str(args.split_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(output_path, index=False)

    labels = [label for label in DEFAULT_LABELS if label in existing_df.columns]
    summary: Dict[str, Any] = {
        "final_csv": str(final_path),
        "existing_csv": str(existing_path),
        "output_csv": str(output_path),
        "subject_disjoint": not bool(args.allow_subject_overlap),
        "split_name": str(args.split_name),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "final_rows": int(len(final_df)),
        "existing_rows": int(len(existing_df)),
        "excluded_dicom_overlap_rows": int(existing_df["dicom_id"].astype(str).isin(final_dicoms).sum()),
        "candidate_rows": int(len(holdout)),
        "candidate_subjects": int(holdout["subject_id"].nunique()),
        "candidate_studies": int(holdout["study_id"].nunique()) if "study_id" in holdout.columns else None,
        "candidate_dicoms": int(holdout["dicom_id"].nunique()),
        "rows_with_any_label": int(holdout[labels].notna().any(axis=1).sum()) if labels else 0,
        "rows_with_all_labels": int(holdout[labels].notna().all(axis=1).sum()) if labels else 0,
        "label_summary": _label_summary(holdout, labels),
        "notes": [
            "This is a project holdout built from existing images, not an official MIMIC-CXR test split.",
            "The source CSV has sparse CheXpert labels; use uncertain_policy=ignore for the main clinical metric.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_holdout(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
