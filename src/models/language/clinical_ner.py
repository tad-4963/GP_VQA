import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

class ClinicalEntityExtractor:
    INVALID_VALUES = {"", "unknown", "nan", "null", "none"}

    def __init__(self, diseases_csv_path: str, anatomy_csv_path: str):
        self.rules: Dict[str, Dict[str, Set[str]]] = {
            "DISEASE": {},
            "ANATOMY": {},
        }
        self._compiled_rules: List[Tuple[str, str, str, re.Pattern]] = []

        self._load_rules_from_normalized_csv(
            csv_path=diseases_csv_path,
            mapped_col="Mapped_Disease",
            label="DISEASE",
        )
        self._load_rules_from_normalized_csv(
            csv_path=anatomy_csv_path,
            mapped_col="Mapped_Anatomy",
            label="ANATOMY",
        )
        self._compile_patterns()

        disease_terms = len(self.rules["DISEASE"])
        anatomy_terms = len(self.rules["ANATOMY"])
        print(
            "✅ Da nap bo tu dien chuan hoa: "
            f"DISEASE terms={disease_terms} | ANATOMY terms={anatomy_terms}"
        )

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    def _split_mapped_values(self, mapped_value: str) -> List[str]:
        items = [self._normalize_text(part) for part in str(mapped_value).split("|")]
        return [x for x in items if x not in self.INVALID_VALUES]

    def _add_rule(self, term: str, canonical_labels: List[str], label: str) -> None:
        term_norm = self._normalize_text(term)
        if not term_norm or term_norm in self.INVALID_VALUES:
            return

        bucket = self.rules[label].setdefault(term_norm, set())
        for canonical in canonical_labels:
            canonical_norm = self._normalize_text(canonical)
            if canonical_norm and canonical_norm not in self.INVALID_VALUES:
                bucket.add(canonical_norm)

    def _load_rules_from_normalized_csv(self, csv_path: str, mapped_col: str, label: str) -> None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Raw_Term" not in (reader.fieldnames or []) or mapped_col not in (reader.fieldnames or []):
                raise ValueError(f"CSV {path} can cot Raw_Term va {mapped_col}")

            for row in reader:
                raw_term = row.get("Raw_Term", "")
                mapped_value = row.get(mapped_col, "")
                canonical_labels = self._split_mapped_values(mapped_value)
                if not canonical_labels:
                    continue

                self._add_rule(raw_term, canonical_labels, label)
                for canonical in canonical_labels:
                    self._add_rule(canonical, [canonical], label)

    def _compile_patterns(self) -> None:
        compiled = []
        for label in ("DISEASE", "ANATOMY"):
            for term, canonical_set in self.rules[label].items():
                if len(term) < 2:
                    continue
                pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
                canonical_sorted = sorted(canonical_set)
                for canonical in canonical_sorted:
                    compiled.append((label, term, canonical, pattern))

        # Longest-first để ưu tiên phrase dài khi có overlap.
        compiled.sort(key=lambda item: len(item[1]), reverse=True)
        self._compiled_rules = compiled

    def _is_overlap(self, span: Tuple[int, int], occupied: List[Tuple[int, int]]) -> bool:
        for s, e in occupied:
            if span[0] < e and s < span[1]:
                return True
        return False

    def extract(self, text: str) -> List[Dict[str, object]]:
        text_norm = self._normalize_text(text)
        entities: List[Dict[str, object]] = []
        occupied_spans: List[Tuple[int, int]] = []

        for label, term, canonical, pattern in self._compiled_rules:
            for match in pattern.finditer(text_norm):
                span = (match.start(), match.end())
                if self._is_overlap(span, occupied_spans):
                    continue
                occupied_spans.append(span)
                entities.append(
                    {
                        "entity": text_norm[span[0]:span[1]],
                        "label": label,
                        "canonical": canonical,
                        "start": span[0],
                        "end": span[1],
                    }
                )

        entities.sort(key=lambda x: (int(x["start"]), int(x["end"])))
        return entities

    def extract_grouped(self, text: str) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {"DISEASE": [], "ANATOMY": []}
        seen = {"DISEASE": set(), "ANATOMY": set()}

        for ent in self.extract(text):
            label = str(ent["label"])
            canonical = str(ent["canonical"])
            if canonical not in seen[label]:
                seen[label].add(canonical)
                grouped[label].append(canonical)
        return grouped

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, object]]]:
        return [self.extract(text) for text in texts]

if __name__ == "__main__":
    disease_csv = "data/label/normalized_diseases.csv"
    anatomy_csv = "data/label/normalized_anatomy.csv"

    extractor = ClinicalEntityExtractor(
        diseases_csv_path=disease_csv,
        anatomy_csv_path=anatomy_csv,
    )

    test_text = "What abnormalities are seen in this image? Possible pneumonia in the left lung base with pleural effusion."
    print(f"\nCau hoi test: '{test_text}'")

    print("\n--- KET QUA TRICH XUAT CHI TIET ---")
    extracted_entities = extractor.extract(test_text)
    for ent in extracted_entities:
        print(f"  - {ent['entity']} | {ent['label']} | canonical={ent['canonical']}")

    print("\n--- GROUPED OUTPUT ---")
    print(extractor.extract_grouped(test_text))