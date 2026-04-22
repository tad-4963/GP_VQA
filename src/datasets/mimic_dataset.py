import os
import json
import torch
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MedicalVQADataset(Dataset):
    # Findings with zero support in the current corpus; drop globally to avoid impossible targets.
    DEFAULT_DROPPED_FINDINGS = {
        "normal",
        "blunted",
        "clear",
        "feeding tube",
        "icd",
        "missing part",
        "poorly defined",
        "sternal wire",
        "swan-ganz catheter",
        "tracheostomy tube",
        "unfolding",
        "apical capping",
        "hardware failure",
        "peribronchial cuffing",
        "picc",
        "pigtail catheter",
        "widened",
        "chest tube",
        "drain",
        "obscured",
        "silhouette sign",
        "pacemaker",
        "valve prosthesis",
        "distended",
        "endotracheal tube",
        "enteric tube",
        "crowded",
        "artifact",
        "surgical clip",
        "airspace disease",
        "rotated",
        "cvp line",
        "linear band",
    }

    def __init__(
        self,
        csv_file,
        json_file,
        img_dir,
        norm_disease_csv=None,
        norm_anatomy_csv=None,
        transform=None,
        split=None,
        fail_on_image_error=True,
        skip_on_image_error=False,
        max_image_error_logs=5,
    ):
        """
        Dataset cho Medical VQA.
        Cấu trúc chuẩn: 11 Bệnh lý (Global Labels),
        Disease-entities (Findings), 29 Anatomy-entities (Location).
        Mỗi entity mẫu là disease + anatomy (anatomy có thể không có).
        """
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()

        self.split_col = 'split' if 'split' in self.df.columns else None
        self.split_aliases = {
            'train': {'train'},
            'valid': {'valid', 'val', 'validation', 'validate'},
            'validate': {'valid', 'val', 'validation', 'validate'},
            'test': {'test'},
        }
        if split is not None and self.split_col is not None:
            target_split = str(split).strip().lower()
            accepted_splits = self.split_aliases.get(target_split, {target_split})
            split_series = self.df[self.split_col].astype(str).str.strip().str.lower()
            self.df = self.df[split_series.isin(accepted_splits)].reset_index(drop=True)
        
        if 'filename' in self.df.columns:
            self.fname_col = 'filename'
        elif 'path' in self.df.columns:
            self.fname_col = 'path'
        elif 'dicom_id' in self.df.columns:
            self.fname_col = 'dicom_id'
        else:
            raise ValueError("CSV phải có một trong các cột: filename, path hoặc dicom_id")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
            
        self.json_dict = {str(item['study_id']): item for item in json_list}
        self.img_dir = img_dir
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((1008, 1008)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.fail_on_image_error = fail_on_image_error
        self.skip_on_image_error = skip_on_image_error
        self.max_image_error_logs = max_image_error_logs
        self.image_error_count = 0
        
        # Global disease labels: chỉ dùng schema 14 nhãn CheXpert mới.
        canonical_14 = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        missing_cols = [c for c in canonical_14 if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                "CSV không đúng schema 14 nhãn. Thiếu cột: " + ", ".join(missing_cols)
            )
        self.disease_cols = canonical_14

        # Label space cho thực thể bệnh (findings) và vị trí (anatomy)
        self.raw_anatomy_labels = [
            "right lung", "right upper lung zone", "right mid lung zone", "right lower lung zone", "right apical zone",
            "right hilar structures", "right costophrenic angle", "right hemidiaphragm", "left lung", "left upper lung zone",
            "left mid lung zone", "left lower lung zone", "left apical zone", "left hilar structures", "left costophrenic angle",
            "left hemidiaphragm", "trachea", "spine", "right clavicle", "left clavicle", "aortic arch", "mediastinum",
            "upper mediastinum", "superior vena cava", "cardiac silhouette", "cavoatrial junction", "right atrium", "carina", "abdomen"
        ]
        self.raw_finding_labels = [
            "consolidation", "pleural effusion", "pneumothorax", "atelectasis", "pulmonary edema", "cardiomegaly", "pneumonia",
            "emphysema", "hernia", "mass", "nodule", "lung opacity", "pleural thickening", "calcification", "granuloma", "fracture",
            "pneumomediastinum", "pneumoperitoneum", "subcutaneous emphysema", "hyperaeration", "cyst / cystic", "scarring / fibrotic",
            "linear band", "infiltration", "vascular congestion", "vascular redistribution", "cavitation", "bronchiectasis", "enlarged",
            "tortuous", "elevated", "blunted", "shifted", "prominent", "abnormal", "obscured", "clear", "distended",
            "collapsed", "widened", "crowded", "rotated", "low lung volumes", "unfolding", "engorgement", "eventration", "lucency",
            "peribronchial cuffing", "airspace disease", "interstitial lung disease", "opacification", "silhouette sign", "apical capping",
            "plural abnormality", "scoliosis", "kyphosis", "degenerative change", "osteopenia", "osteophyte", "arthritic change",
            "surgical material", "hardware failure", "missing part", "artifact", "asymmetry", "poorly defined", "endotracheal tube",
            "enteric tube", "cvp line", "picc", "chest tube", "pacemaker", "icd", "hardware", "sternal wire", "surgical clip",
            "valve prosthesis", "tracheostomy tube", "drain", "pigtail catheter", "swan-ganz catheter", "feeding tube"
        ]

        self.raw_finding_labels = [
            label for label in self.raw_finding_labels
            if label not in self.DEFAULT_DROPPED_FINDINGS
        ]

        self.anatomy_to_idx = {label: idx for idx, label in enumerate(self.raw_anatomy_labels)}
        self.finding_to_idx = {label: idx for idx, label in enumerate(self.raw_finding_labels)}

        # Backward-compatible alias cho code cũ.
        self.raw_disease_labels = self.raw_finding_labels
        self.disease_to_idx = self.finding_to_idx

        # Alias nhỏ để gom các biến thể viết khác nhau về đúng label canonical.
        self.disease_alias = {
            "edema": "pulmonary edema",
            "pulmonary oedema": "pulmonary edema",
            "cyst": "cyst / cystic",
            "cystic": "cyst / cystic",
            "scarring": "scarring / fibrotic",
            "fibrotic": "scarring / fibrotic",
            "pleural abnormality": "plural abnormality",
        }

        # Tải 2 từ điển hoàn toàn độc lập mà FPT LLM vừa tạo ra
        self.disease_map = self._load_custom_dict(norm_disease_csv, col_raw="Raw_Term", col_mapped="Mapped_Disease")
        self.anatomy_map = self._load_custom_dict(norm_anatomy_csv, col_raw="Raw_Term", col_mapped="Mapped_Anatomy")

    def _split_compound_terms(self, term):
        """Tách cụm đã normalize kiểu 'a and b' thành các term đơn lẻ."""
        if not term:
            return []
        return [p.strip() for p in str(term).split(" and ") if p and p.strip()]

    def _canonicalize_disease_label(self, term):
        t = str(term).strip().lower()
        return self.disease_alias.get(t, t)

    def _normalize_optional_field(self, value):
        """Chuẩn hóa field tùy chọn (level/type): thiếu dữ liệu -> chuỗi rỗng."""
        if value is None:
            return ""
        text = str(value).strip().lower()
        if text in {"", "nan", "none", "null", "unknown"}:
            return ""
        return text

    def _dedupe_preserve_order(self, terms):
        seen = set()
        output = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                output.append(term)
        return output

    def _load_custom_dict(self, csv_path, col_raw, col_mapped):
        norm_dict = {}
        if not csv_path or not os.path.exists(csv_path):
            return norm_dict
            
        df_norm = pd.read_csv(csv_path)
        if col_raw in df_norm.columns and col_mapped in df_norm.columns:
            for _, row in df_norm.iterrows():
                raw_term = str(row[col_raw]).strip().lower()
                mapped_term = str(row[col_mapped]).strip().lower()
                
                if mapped_term in ['unknown', 'nan', '', 'null']:
                    norm_dict[raw_term] = "DROP"
                else:
                    # Giữ đầy đủ cụm đôi/đa cụm: a|b -> a and b
                    mapped_parts = [p.strip() for p in mapped_term.split('|')]
                    mapped_parts = [p for p in mapped_parts if p and p not in ['unknown', 'nan', 'null']]
                    norm_dict[raw_term] = " and ".join(mapped_parts) if mapped_parts else "DROP"
        return norm_dict

    def _normalize_term(self, term, term_type="disease"):
        """Chuẩn hóa dựa trên finding/anatomy label space hiện tại."""
        term_lower = str(term).strip().lower()
        if term_type == "disease":
            return self.disease_map.get(term_lower, term_lower)
        elif term_type == "anatomy":
            return self.anatomy_map.get(term_lower, term_lower)
        return term_lower

    def _extract_text_prompts(self, json_data):
        prompts = []
        entity_items = []
        finding_hits = torch.zeros(len(self.raw_finding_labels), dtype=torch.float32)
        anatomy_hits = torch.zeros(len(self.raw_anatomy_labels), dtype=torch.float32)
        merged_entities = {}
        if isinstance(json_data.get('entity'), dict):
            merged_entities.update(json_data['entity'])
        if isinstance(json_data.get('opacity_vs_clear'), dict):
            # Chuẩn hóa opacity_vs_clear về cùng cấu trúc như entity để xử lý thống nhất.
            merged_entities.update(json_data['opacity_vs_clear'])

        for disease_raw, details in merged_entities.items():
            norm_disease = self._normalize_term(disease_raw, "disease")

            # CHỐT CHẶN 1: Nếu bệnh lý là rác (DROP) -> Bỏ qua hoàn toàn cụm này
            if norm_disease == "DROP":
                continue

            valid_disease_terms = []
            for disease_term in self._split_compound_terms(norm_disease):
                disease_term = self._canonicalize_disease_label(disease_term)
                if disease_term in self.finding_to_idx:
                    valid_disease_terms.append(disease_term)
                    finding_hits[self.finding_to_idx[disease_term]] = 1.0
            valid_disease_terms = self._dedupe_preserve_order(valid_disease_terms)

            # Nếu disease map ra nhiều term thì chỉ giữ term hợp lệ trong finding labels hiện tại.
            # Nếu không còn term nào hợp lệ thì bỏ hẳn entity này.
            if len(valid_disease_terms) == 0:
                continue

            location_raw = details.get('location', '') if isinstance(details, dict) else ''
            level_raw = details.get('level', '') if isinstance(details, dict) else ''
            type_raw = details.get('type', '') if isinstance(details, dict) else ''
            norm_location = self._normalize_term(location_raw, "anatomy") if location_raw else ""

            # CHỐT CHẶN 2: Nếu vị trí là rác (DROP) -> Xóa vị trí, nhưng vẫn giữ bệnh lý
            if norm_location == "DROP":
                norm_location = ""

            valid_anatomy_terms = []
            for anatomy_term in self._split_compound_terms(norm_location):
                if anatomy_term in self.anatomy_to_idx:
                    valid_anatomy_terms.append(anatomy_term)
                    anatomy_hits[self.anatomy_to_idx[anatomy_term]] = 1.0
            valid_anatomy_terms = self._dedupe_preserve_order(valid_anatomy_terms)

            # Prompt chỉ giữ disease/anatomy đã đi qua normalize + lọc label space.
            parts = [" and ".join(valid_disease_terms)]
            if len(valid_anatomy_terms) > 0:
                parts.append(f"at {' and '.join(valid_anatomy_terms)}")

            entity_text = " ".join(parts).strip()
            prompts.append(entity_text)
            entity_items.append({
                "diseases": valid_disease_terms,
                "anatomy": valid_anatomy_terms,
                "level": self._normalize_optional_field(level_raw),
                "type": self._normalize_optional_field(type_raw),
            })

        prompt_text = ". ".join(prompts) if prompts else "normal chest anatomy"
        return prompt_text, finding_hits, anatomy_hits, entity_items

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = str(row[self.fname_col]).strip()
        if self.fname_col == 'dicom_id' and not img_filename.lower().endswith('.jpg'):
            img_filename = f"{img_filename}.jpg"
        img_basename = os.path.basename(img_filename)

        study_id = re.sub(r'[^0-9]', '', img_filename)

        # Ưu tiên layout phẳng: <img_dir>/<dicom_id>.jpg
        flat_path = os.path.join(self.img_dir, img_basename)
        if os.path.exists(flat_path):
            img_path = flat_path
        elif self.split_col is not None:
            row_split_raw = str(row.get(self.split_col, '')).strip().lower()
            if row_split_raw in self.split_aliases['valid']:
                split_dir = 'valid'
            elif row_split_raw in self.split_aliases['test']:
                split_dir = 'test'
            else:
                split_dir = 'train'

            # Tìm ảnh theo thứ tự ưu tiên để giảm tối đa số sample bị bỏ vì lệch thư mục split.
            fallback_orders = {
                'train': ['train', 'valid', 'test'],
                'valid': ['valid', 'test', 'train'],
                'test': ['test', 'valid', 'train'],
            }
            candidate_dirs = fallback_orders.get(split_dir, [split_dir, 'valid', 'test', 'train'])

            img_path = None
            for cand_dir in candidate_dirs:
                cand_path = os.path.join(self.img_dir, cand_dir, img_basename)
                if os.path.exists(cand_path):
                    img_path = cand_path
                    break

            if img_path is None:
                # Giữ path đầu tiên để thông báo lỗi rõ ràng theo split gốc.
                img_path = os.path.join(self.img_dir, split_dir, img_basename)
        else:
            img_path = os.path.join(self.img_dir, img_basename)
        
        try:
            # Load ảnh X-quang
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            if self.fail_on_image_error:
                raise RuntimeError(f"Không thể đọc ảnh tại '{img_path}' (study_id={study_id})") from e

            # Giới hạn số lần log để tránh spam terminal khi dữ liệu có nhiều ảnh lỗi.
            if self.image_error_count < self.max_image_error_logs:
                print(f"[WARN] Ảnh lỗi, thay bằng tensor đen: {img_path} | lỗi: {type(e).__name__}")
            self.image_error_count += 1

            # Tạo ảnh đen dự phòng và áp cùng pipeline transform để giữ shape đồng nhất trong batch.
            fallback_pil = Image.new('RGB', (1008, 1008), color=(0, 0, 0))
            if self.transform:
                image = self.transform(fallback_pil)
            else:
                image = torch.zeros((3, 1008, 1008), dtype=torch.float32)

        # Trích xuất vector global labels theo schema 14 nhãn
        labels_values = pd.to_numeric(row[self.disease_cols], errors='coerce').fillna(0.0).values.astype('float32')
        global_labels = torch.tensor(labels_values, dtype=torch.float32)

        # Trích xuất Local Prompts (Findings & Anatomy)
        local_prompt_str = "normal chest anatomy"
        entity_items = []
        finding_entity_labels = torch.zeros(len(self.raw_finding_labels), dtype=torch.float32)
        anatomy_entity_labels = torch.zeros(len(self.raw_anatomy_labels), dtype=torch.float32)
        if study_id in self.json_dict:
            local_prompt_str, finding_entity_labels, anatomy_entity_labels, entity_items = self._extract_text_prompts(self.json_dict[study_id])

        # Vector entity tổng hợp: [finding entities | 29 anatomy entities]
        entity_labels = torch.cat([finding_entity_labels, anatomy_entity_labels], dim=0)

        return {
            'image': image,
            'global_labels': global_labels,   # Kích thước [11]
            'local_prompts': local_prompt_str,
            'entity_multihot_labels': entity_labels,          # Key rõ nghĩa cho train đa nhãn entity
            'entity_labels': entity_labels,                    # Kích thước [finding + anatomy]
            'entity_finding_labels': finding_entity_labels,   # Kích thước [finding]
            'entity_anatomy_labels': anatomy_entity_labels,   # Kích thước [29]
            'entity_items': entity_items,
            # Backward-compatible alias cho pipeline cũ.
            'entity_disease_labels': finding_entity_labels,
            'study_id': study_id
        }
    
# === ĐOẠN CODE TEST NHANH  ===
if __name__ == "__main__":
    print("Đang khởi tạo MedicalVQADataset để test...")
    
    TEST_CSV = "data/mimic_cxr_balanced.csv"         # File CSV đã lọc chứa filename và nhãn global
    TEST_JSON = "data/medical_cxr/filtered_all_diseases.json"       # File JSON chứa thông tin entity thô
    TEST_IMG_DIR = "data/"       # Thư mục chứa ảnh X-quang
    DISEASE_CSV = "data/label/normalized_diseases.csv"
    ANATOMY_CSV = "data/label/normalized_anatomy.csv"
    TEST_SPLIT = "train"
    NUM_DEBUG_SAMPLES = 8

    def decode_positive_labels(binary_tensor, label_names):
        idxs = torch.where(binary_tensor > 0.5)[0].tolist()
        return [label_names[i] for i in idxs]
    
    try:
        # Khởi tạo dataset
        test_dataset = MedicalVQADataset(
            csv_file=TEST_CSV,
            json_file=TEST_JSON,
            img_dir=TEST_IMG_DIR,
            norm_disease_csv=DISEASE_CSV,
            norm_anatomy_csv=ANATOMY_CSV,
            split=TEST_SPLIT,
        )
        
        print(f"Khởi tạo thành công! split={TEST_SPLIT} | Tổng số mẫu: {len(test_dataset)}")
        print(f"Số nhãn global: {len(test_dataset.disease_cols)}")
        print(f"Số finding-entity labels: {len(test_dataset.raw_finding_labels)}")
        print(f"Số anatomy-entity labels: {len(test_dataset.raw_anatomy_labels)}")
        print("=" * 80)

        sample_count = min(NUM_DEBUG_SAMPLES, len(test_dataset))
        mismatch_count = 0
        empty_anatomy_count = 0
        total_pos_entity = 0

        for i in range(sample_count):
            sample = test_dataset[i]
            entity_from_parts = torch.cat([
                sample['entity_finding_labels'],
                sample['entity_anatomy_labels']
            ], dim=0)

            is_match = torch.equal(entity_from_parts, sample['entity_multihot_labels'])
            if not is_match:
                mismatch_count += 1

            pos_finding_entities = decode_positive_labels(sample['entity_finding_labels'], test_dataset.raw_finding_labels)
            pos_anatomy_entities = decode_positive_labels(sample['entity_anatomy_labels'], test_dataset.raw_anatomy_labels)
            sample_entities = sample.get('entity_items', [])

            if len(pos_anatomy_entities) == 0:
                empty_anatomy_count += 1

            total_pos_entity += int(sample['entity_multihot_labels'].sum().item())

            print(f"Mẫu #{i+1} - Study ID: {sample['study_id']}")
            print(f"Kích thước ảnh (Tensor): {sample['image'].shape}")
            print(
                f"Shape labels -> global:{tuple(sample['global_labels'].shape)} | "
                f"entity:{tuple(sample['entity_multihot_labels'].shape)} | "
                f"entity_finding:{tuple(sample['entity_finding_labels'].shape)} | "
                f"entity_anatomy:{tuple(sample['entity_anatomy_labels'].shape)}"
            )
            print(f"Check concat entity_labels: {'OK' if is_match else 'MISMATCH'}")
            print(
                f"Positive counts -> finding_entity:{len(pos_finding_entities)} | "
                f"anatomy_entity:{len(pos_anatomy_entities)}"
            )
            if sample_entities:
                print("Entities (structured):")
                for e_idx, entity in enumerate(sample_entities, start=1):
                    print(f"  {e_idx}. diseases={entity.get('diseases', [])} | anatomy={entity.get('anatomy', [])} | level='{entity.get('level', '')}' | type='{entity.get('type', '')}'")
            else:
                print("Entities (structured): []")
            print(f"Local Prompt: {sample['local_prompts']}")
            print("-" * 80)

        avg_pos_entity = total_pos_entity / max(sample_count, 1)
        print("\nTỔNG KẾT TEST:")
        print(f"- Số mẫu kiểm tra: {sample_count}")
        print(f"- Mismatch concat entity_labels: {mismatch_count}")
        print(f"- Mẫu không có anatomy entity: {empty_anatomy_count}")
        print(f"- Trung bình số entity dương/mẫu: {avg_pos_entity:.2f}")
            
    except Exception as e:
        print(f"Lỗi khi khởi tạo hoặc chạy test: {e}")