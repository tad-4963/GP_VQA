import os
import re
import sys
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Sam3Model
from peft import LoraConfig, get_peft_model

class MedSAM_VisionEncoder(nn.Module):
    def __init__(
        self,
        sam3_base_checkpoint="/data/weights/sam3",       # Đường dẫn Base Model trên Modal
        lora_weights_path="/data/weights/medsam3_v1_lora", # Đường dẫn LoRA trên Modal
        embed_dim=256,
        freeze_encoder=False,
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        text_max_length=48,
        use_text_guidance=True,
        num_text_attention_heads=8,
        num_global_disease_labels=14,
        num_entity_finding_labels=51,
        num_entity_anatomy_labels=29,
        # Deprecated alias, giữ để không vỡ code cũ khi truyền keyword cũ.
        num_entity_disease_labels=None,
        require_attention_lora=False,
        encoder_backend="transformers",
        medsam3_repo_path="/root/external/MedSAM3",
    ):
        super().__init__()
        self.use_text_guidance = use_text_guidance
        self.text_max_length = text_max_length
        self.text_model_name = text_model_name
        self.num_text_attention_heads = num_text_attention_heads
        self.freeze_encoder = freeze_encoder
        self.require_attention_lora = require_attention_lora
        self.encoder_backend = str(encoder_backend).strip().lower()
        self.medsam3_repo_path = medsam3_repo_path
        self.num_global_disease_labels = num_global_disease_labels

        if num_entity_disease_labels is not None:
            num_entity_finding_labels = num_entity_disease_labels

        self.num_entity_finding_labels = num_entity_finding_labels
        self.num_entity_anatomy_labels = num_entity_anatomy_labels
        self.num_entity_labels = num_entity_finding_labels + num_entity_anatomy_labels

        # Backward-compatible alias.
        self.num_entity_disease_labels = self.num_entity_finding_labels
        
        # ==========================================
        # 1. LOAD BASE MODEL (SAM3) & LORA
        # ==========================================
        if self.encoder_backend == "medsam3":
            print(f"Đang tải MedSAM3 vision trunk từ checkpoint/source: {sam3_base_checkpoint}...")
        else:
            print(f"Đang tải SAM Base Model (transformers) từ: {sam3_base_checkpoint}...")
        self.image_encoder_mode = "transformers"

        if self.encoder_backend == "medsam3":
            self.image_encoder = self._load_medsam3_trunk(sam3_base_checkpoint)
            self.image_encoder_mode = "medsam3"
        else:
            # Tải toàn bộ cấu trúc SAM
            base_sam = Sam3Model.from_pretrained(sam3_base_checkpoint)
            # Chúng ta CHỈ LẤY phần Image Encoder (bỏ Prompt Encoder và Mask Decoder để nhẹ máy)
            self.image_encoder = base_sam.vision_encoder
        
        # Nạp "Khối óc Y khoa" (LoRA)
        if lora_weights_path:
            pt_path = os.path.join(lora_weights_path, "best_lora_weights.pt")
            if self.image_encoder_mode == "medsam3":
                self.image_encoder = self._apply_native_medsam3_lora(self.image_encoder, pt_path)
            else:
                self.image_encoder = self._apply_smart_matching_lora(self.image_encoder, pt_path)

        # Đóng băng trọng số nếu chỉ muốn train các lớp Linear/Graph phía sau
        if freeze_encoder:
            print("❄️ Đã đóng băng trọng số Vision Encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            # Nếu KHÔNG đóng băng, ta chỉ nên cho phép module LoRA được cập nhật gradient
            if hasattr(self.image_encoder, "print_trainable_parameters"):
                self.image_encoder.print_trainable_parameters()

        # ==========================================
        # 2. CÁC LỚP PROJECTOR
        # ==========================================
        sam_hidden_dim = 1024 # Giữ cố định như cấu hình cũ để tương thích checkpoint hiện tại
        
        self.global_proj = nn.Sequential(
            # Bỏ AdaptiveAvgPool2d và Flatten đi
            nn.Linear(sam_hidden_dim, embed_dim), 
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        self.local_proj = nn.Sequential(
            nn.Linear(sam_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Head đa nhãn bệnh toàn cục (schema 14 nhãn).
        self.classifier = nn.Linear(embed_dim, self.num_global_disease_labels)

        # Head đa nhãn entity: [finding entities | anatomy entities].
        self.entity_classifier = nn.Linear(embed_dim, self.num_entity_labels)

        # 3. Nhánh Text-Guided
        self.tokenizer = None
        self.text_encoder = None
        self.text_proj = None
        self.text_to_patch_attn = None
        if self.use_text_guidance:
            self._init_text_modules()

    def _load_medsam3_trunk(self, sam3_base_checkpoint):
        repo_path = os.path.abspath(self.medsam3_repo_path)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"Không tìm thấy MedSAM3 repo tại '{repo_path}'. "
                "Hãy mount thư mục external/MedSAM3 vào runtime trước khi bật encoder_backend='medsam3'."
            )

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from sam3.model_builder import build_sam3_image_model

        bpe_path = os.path.join(repo_path, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.isfile(bpe_path):
            raise FileNotFoundError(
                f"Không tìm thấy BPE vocab của MedSAM3 tại '{bpe_path}'."
            )

        checkpoint_path = None
        load_from_hf = True
        if os.path.isfile(sam3_base_checkpoint):
            checkpoint_path = sam3_base_checkpoint
            load_from_hf = False
        elif os.path.isdir(sam3_base_checkpoint):
            local_ckpt = os.path.join(sam3_base_checkpoint, "sam3.pt")
            if os.path.isfile(local_ckpt):
                checkpoint_path = local_ckpt
                load_from_hf = False

        sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            eval_mode=False,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_hf,
            compile=False,
        )

        # Giữ reference model đầy đủ để tránh bị giải phóng ngoài ý muốn.
        self._sam3_full_model = sam3_model

        candidate_paths = [
            ("backbone", "vision_backbone", "trunk"),
            ("backbone", "visual", "trunk"),
            ("vision_backbone", "trunk"),
        ]
        for path in candidate_paths:
            cur = sam3_model
            ok = True
            for part in path:
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok and isinstance(cur, nn.Module):
                print(f"Dùng MedSAM3 vision trunk từ path: model.{'.'.join(path)}")
                return cur

        # Fallback cuối: quét named_modules để tìm module có hậu tố 'trunk'.
        for name, module in sam3_model.named_modules():
            if name.endswith("trunk") and isinstance(module, nn.Module):
                print(f"Dùng MedSAM3 vision trunk từ named_modules fallback: {name}")
                return module

        raise RuntimeError(
            "Không tìm thấy vision trunk trong MedSAM3 model "
            "(đã thử: backbone.vision_backbone.trunk, backbone.visual.trunk, vision_backbone.trunk)."
        )

    def _apply_native_medsam3_lora(self, trunk_encoder, pt_path):
        if not os.path.exists(pt_path):
            print(f"CẢNH BÁO: Không tìm thấy {pt_path}. Sẽ sử dụng Base weights.")
            return trunk_encoder

        repo_path = os.path.abspath(self.medsam3_repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        src_path = os.path.join(repo_path, "src")
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from lora.lora_utils import LoRAConfig as NativeLoRAConfig
            from lora.lora_utils import inject_lora_into_model, load_lora_state_dict, get_lora_state_dict
        except ModuleNotFoundError:
            # Fallback cho một số layout repo MedSAM3 cũ.
            from sam3_lora.lora.lora_utils import LoRAConfig as NativeLoRAConfig
            from sam3_lora.lora.lora_utils import (
                inject_lora_into_model,
                load_lora_state_dict,
                get_lora_state_dict,
            )

        print(f"Khởi động Native MedSAM3 LoRA loader. Đang đọc trọng số từ: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            print(f"Lỗi đọc file .pt: {e}")
            return trunk_encoder

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        if not isinstance(state_dict, dict):
            print("CẢNH BÁO: Checkpoint LoRA không ở dạng dict. Sẽ sử dụng Base weights.")
            return trunk_encoder

        # Suy rank từ tensor lora_A đầu tiên thuộc vision trunk.
        sample_tensor = None
        for k, v in state_dict.items():
            if not torch.is_tensor(v) or "lora_A" not in k:
                continue
            clean_key = k[7:] if k.startswith("module.") else k
            if "trunk." in clean_key:
                sample_tensor = v
                break
        if sample_tensor is None:
            print("Không tìm thấy tensor lora_A thuộc trunk trong checkpoint. Hủy nạp LoRA.")
            return trunk_encoder

        r = min(sample_tensor.shape)
        print(f"Dò tìm thành công (native): Rank (r) = {r}")

        native_cfg = NativeLoRAConfig(
            rank=r,
            alpha=r,
            dropout=0.0,
            target_modules=["qkv", "proj", "fc1", "fc2"],
        )
        lora_trunk = inject_lora_into_model(trunk_encoder, native_cfg, verbose=False)

        # Convert key checkpoint về namespace của trunk model (blocks.*...lora.lora_A/B)
        converted = {}
        trunk_prefix_re = re.compile(r"(?:^|\.)trunk\.")
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            clean_key = k[7:] if k.startswith("module.") else k
            if "lora_" not in clean_key:
                continue

            m = trunk_prefix_re.search(clean_key)
            if not m:
                continue
            local_key = clean_key[m.end():]
            converted[local_key] = v

        expected_state = get_lora_state_dict(lora_trunk)
        expected_keys = set(expected_state.keys())
        provided_keys = set(converted.keys())

        # Chỉ nạp key vừa tồn tại trong model vừa khớp shape để tránh crash runtime.
        filtered = {}
        shape_mismatch_count = 0
        transposed_count = 0
        for k in expected_keys & provided_keys:
            src_t = converted[k]
            dst_t = expected_state[k]
            if tuple(src_t.shape) == tuple(dst_t.shape):
                filtered[k] = src_t
            elif src_t.ndim == 2 and tuple(src_t.t().shape) == tuple(dst_t.shape):
                filtered[k] = src_t.t().contiguous()
                transposed_count += 1
            else:
                shape_mismatch_count += 1

        matched_keys = set(filtered.keys())
        load_lora_state_dict(lora_trunk, filtered)

        print(
            "✅ Hoàn tất Native LoRA load:",
            f"matched={len(matched_keys)} / expected={len(expected_keys)} / provided={len(provided_keys)}"
        )
        if transposed_count > 0:
            print(f"ℹ️ Đã auto-transpose {transposed_count} key LoRA để khớp shape model đích.")
        if shape_mismatch_count > 0:
            print(f"⚠️ Bỏ qua {shape_mismatch_count} key LoRA do lệch shape với model đích.")

        # Log coverage theo module chính để dễ theo dõi.
        def _count_mod(keys, mod):
            return sum(1 for kk in keys if f".{mod}." in kk)

        for mod_name in ["qkv", "proj", "fc1", "fc2"]:
            exp_n = _count_mod(expected_keys, mod_name)
            mat_n = _count_mod(matched_keys, mod_name)
            if exp_n > 0:
                print(f" - {mod_name}: matched {mat_n}/{exp_n}")

        attn_expected = _count_mod(expected_keys, "qkv") + _count_mod(expected_keys, "proj")
        attn_matched = _count_mod(matched_keys, "qkv") + _count_mod(matched_keys, "proj")
        if attn_expected > 0 and attn_matched == 0:
            print("⚠️ CẢNH BÁO: Native loader không map được attention LoRA (qkv/proj).")
            if self.require_attention_lora:
                print("⚠️ require_attention_lora=True -> Bỏ nạp LoRA và quay về base encoder.")
                return trunk_encoder

        return lora_trunk

    # ==========================================
    # HÀM BỔ TRỢ: SMART MATCHING LORA ALGORITHM
    # ==========================================
    def _apply_smart_matching_lora(self, base_encoder, pt_path):
        if not os.path.exists(pt_path):
            print(f"CẢNH BÁO: Không tìm thấy {pt_path}. Sẽ sử dụng Base weights.")
            return base_encoder

        print(f"Khởi động Smart Matching. Đang đọc trọng số từ: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            print(f"Lỗi đọc file .pt: {e}")
            return base_encoder

        # Hỗ trợ checkpoint bọc key phổ biến.
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        if not isinstance(state_dict, dict):
            print("CẢNH BÁO: Checkpoint LoRA không ở dạng dict. Sẽ sử dụng Base weights.")
            return base_encoder

        # 1. NỘI SUY RANK (r) TỰ ĐỘNG
        try:
            sample_key = next(
                k for k, v in state_dict.items()
                if "lora_A" in k and torch.is_tensor(v)
            )
            r = min(state_dict[sample_key].shape)
            print(f"Dò tìm thành công: Rank (r) = {r}")
        except StopIteration:
            print("Không tìm thấy tensor lora_A nào trong checkpoint. Hủy nạp LoRA.")
            return base_encoder

        # 2. XÂY DỰNG BỘ KHUNG PEFT
        config = LoraConfig(
            r=r,
            lora_alpha=r, 
            target_modules=["qkv", "proj", "fc1", "fc2"],
            bias="none"
        )
        peft_encoder = get_peft_model(base_encoder, config)
        peft_dict = dict(peft_encoder.named_parameters())
        
        # 3. THUẬT TOÁN KHỚP NỐI THÔNG MINH
        mapped_state_dict = {}
        matched_count = 0
        skipped_count = 0
        ckpt_module_counter = Counter()
        matched_module_counter = Counter()

        for ckpt_key, weight_tensor in state_dict.items():
            if not torch.is_tensor(weight_tensor):
                skipped_count += 1
                continue

            clean_key = ckpt_key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            # Bước lọc 1: Regex bắt đúng 3 thông tin quan trọng
            match = re.search(r'blocks\.(\d+).*?(qkv|proj|fc1|fc2).*?(lora_[AB])', clean_key)
            if not match:
                skipped_count += 1
                continue # Bỏ qua ngay lập tức các key rác
                
            block_idx, target_module, lora_type = match.groups()
            ckpt_module_counter[target_module] += 1
            
            # Bước lọc 2: Tìm "ổ cắm" tương ứng bên cấu trúc chuẩn của Hugging Face
            peft_match_keys = [
                k for k in peft_dict.keys()
                if re.search(rf"(layers|blocks)\.{block_idx}\.", k) and target_module in k and lora_type in k
            ]
            
            # Bước lọc 3: Khớp nối và cân chỉnh kích thước
            if len(peft_match_keys) == 1:
                target_peft_key = peft_match_keys[0]
                target_peft_weight = peft_dict[target_peft_key]
                
                # Ép xoay chiều ma trận nếu kích thước không khớp
                if weight_tensor.shape != target_peft_weight.shape:
                    if weight_tensor.ndim == 2 and weight_tensor.t().shape == target_peft_weight.shape:
                        weight_tensor = weight_tensor.t()
                    else:
                        print(f"Kích thước vẫn lệch cho {clean_key}. LoRA: {weight_tensor.shape}, Model: {target_peft_weight.shape}")
                        skipped_count += 1
                        continue # Bỏ qua chứ tuyệt đối không để crash!
                        
                mapped_state_dict[target_peft_key] = weight_tensor
                matched_count += 1
                matched_module_counter[target_module] += 1
            else:
                skipped_count += 1

        # 4. BƠM TRỌNG SỐ VÀO MÔ HÌNH
        peft_encoder.load_state_dict(mapped_state_dict, strict=False)
        if matched_count == 0:
            print("⚠️ CẢNH BÁO: Không ghép được key LoRA nào. Mô hình gần như chạy base weights.")
        print(f"✅ Hoàn tất Smart Matching: Đã ghép {matched_count} khối LoRA (Bỏ qua {skipped_count} keys không tương thích).")

        print("LoRA coverage theo module checkpoint:")
        for mod_name in ["qkv", "proj", "fc1", "fc2"]:
            ckpt_n = ckpt_module_counter.get(mod_name, 0)
            matched_n = matched_module_counter.get(mod_name, 0)
            if ckpt_n > 0:
                print(f" - {mod_name}: matched {matched_n}/{ckpt_n}")

        attn_ckpt = ckpt_module_counter.get("qkv", 0) + ckpt_module_counter.get("proj", 0)
        attn_matched = matched_module_counter.get("qkv", 0) + matched_module_counter.get("proj", 0)
        if attn_ckpt > 0 and attn_matched == 0:
            print("⚠️ CẢNH BÁO: Checkpoint có LoRA attention (qkv/proj) nhưng model hiện tại không map được.")
            print("⚠️ Khả năng cao checkpoint LoRA không tương thích hoàn toàn với backbone Sam3Model hiện tại.")
            if self.require_attention_lora:
                print("⚠️ require_attention_lora=True -> Bỏ nạp LoRA và quay về base encoder.")
                return base_encoder
        
        return peft_encoder

    def _init_text_modules(self):
        if not self.use_text_guidance:
            return
        if self.text_encoder is not None and self.tokenizer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)

        if self.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.local_proj[0].out_features),
            nn.LayerNorm(self.local_proj[0].out_features)
        )
        self.text_to_patch_attn = nn.MultiheadAttention(
            embed_dim=self.local_proj[0].out_features,
            num_heads=self.num_text_attention_heads,
            batch_first=True
        )

        # Text modules are initialized lazily, so align them with the model device.
        model_device = next(self.image_encoder.parameters()).device
        self.text_encoder.to(model_device)
        self.text_proj.to(model_device)
        self.text_to_patch_attn.to(model_device)

    def _normalize_local_prompts(self, local_prompts, batch_size):
        """Chuẩn hóa local_prompts về dạng List[List[str]] để tương thích nhiều pipeline."""
        if local_prompts is None:
            return None
        if not isinstance(local_prompts, (list, tuple)):
            raise ValueError("local_prompts phải là list/tuple theo batch.")
        if len(local_prompts) != batch_size:
            raise ValueError(
                f"Số phần tử local_prompts ({len(local_prompts)}) phải bằng batch size ({batch_size})."
            )

        normalized = []
        for sample_prompts in local_prompts:
            # Case 1: dataset trả về chuỗi đơn, ví dụ "edema at left lung. pleural effusion"
            if isinstance(sample_prompts, str):
                parts = [p.strip() for p in sample_prompts.split(".") if p.strip()]
                normalized.append(parts if parts else ["normal chest anatomy"])
                continue

            # Case 2: đã đúng dạng list[str]
            if isinstance(sample_prompts, (list, tuple)):
                clean_prompts = [p.strip() for p in sample_prompts if isinstance(p, str) and p.strip()]
                normalized.append(clean_prompts if clean_prompts else ["normal chest anatomy"])
                continue

            # Fallback
            normalized.append(["normal chest anatomy"])

        return normalized

    def _mean_pool_text(self, token_embeddings, attention_mask):
        """Mean pooling có mask để thu embedding prompt từ token embeddings."""
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked = token_embeddings * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def _encode_prompt_batch(self, local_prompts, device):
        """
        local_prompts: List[List[str]] với batch-size B.
        Trả về:
            prompt_embeds: (B, P_max, embed_dim)
            prompt_mask: (B, P_max) kiểu bool, True = prompt hợp lệ.
        """
        batch_size = len(local_prompts)
        max_prompts = max(len(p) for p in local_prompts)

        padded_prompts = []
        prompt_mask = torch.zeros(batch_size, max_prompts, dtype=torch.bool, device=device)

        for b_idx, prompts in enumerate(local_prompts):
            clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
            if len(clean_prompts) == 0:
                clean_prompts = ["normal chest anatomy"]

            cur_len = len(clean_prompts)
            prompt_mask[b_idx, :cur_len] = True

            while len(clean_prompts) < max_prompts:
                clean_prompts.append(clean_prompts[-1])
            padded_prompts.extend(clean_prompts)

        tokenized = self.tokenizer(
            padded_prompts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        )
        text_device = next(self.text_encoder.parameters()).device
        tokenized = {k: v.to(text_device) for k, v in tokenized.items()}

        text_outputs = self.text_encoder(**tokenized)
        pooled = self._mean_pool_text(text_outputs.last_hidden_state, tokenized["attention_mask"])
        pooled = self.text_proj(pooled)

        prompt_embeds = pooled.view(batch_size, max_prompts, -1)
        return prompt_embeds, prompt_mask

    def forward(self, images, local_prompts=None, extract_local=True):
        import torch.nn.functional as F
        
        # 1. Ép kích thước ảnh về chuẩn 1008x1008 của SAM3
        # images = F.interpolate(images, size=(1008, 1008), mode="bilinear", align_corners=False)
        
        # 2. Trích xuất đặc trưng
        if self.image_encoder_mode == "medsam3":
            feat_list = self.image_encoder(images)
            if not isinstance(feat_list, (list, tuple)) or len(feat_list) == 0:
                raise RuntimeError("MedSAM3 trunk không trả về feature list hợp lệ.")
            feat_map = feat_list[-1]  # [B, C, H, W]
            image_embeddings = feat_map.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
        else:
            encoder_outputs = self.image_encoder(images)
            image_embeddings = encoder_outputs.last_hidden_state
        
        # 3. Tính Global Feature (Bằng cách lấy trung bình tất cả các token)
        # Lấy trung bình theo chiều dim=1 (chiều Sequence_Length)
        pooled_feat = image_embeddings.mean(dim=1) 
        global_feat = self.global_proj(pooled_feat) # Shape: [B, embed_dim]
        
        # Đưa qua classifier để ra kết quả phân loại đa nhãn bệnh (Logits)
        global_logits = self.classifier(global_feat) # Shape: [B, 11]

        # Đầu ra đa nhãn entity trên cùng không gian đặc trưng global.
        entity_logits = self.entity_classifier(global_feat) # Shape: [B, num_entity_labels]
        entity_finding_logits = entity_logits[:, :self.num_entity_finding_labels]
        entity_anatomy_logits = entity_logits[:, self.num_entity_finding_labels:]
        
        local_feats = None
        prompt_embeddings = None
        concept_features = None
        prompt_mask = None

        if extract_local:
            # 4. Tính Local Feature
            # Vì ảnh đã là chuỗi token [B, 5184, 768], ta đưa thẳng vào local_proj luôn!
            # Không cần reshape hay permute lằng nhằng như code cũ nữa.
            local_feats = self.local_proj(image_embeddings) # Shape: [B, 5184, embed_dim]

            # 5. Text-guided Entity Extraction (Giữ nguyên)
            if self.use_text_guidance and local_prompts is not None:
                self._init_text_modules()
                local_prompts = self._normalize_local_prompts(local_prompts, images.size(0))

                prompt_embeddings, prompt_mask = self._encode_prompt_batch(
                    local_prompts=local_prompts,
                    device=images.device,
                )

                # Query = text concepts, Key/Value = image patch features
                concept_features, _ = self.text_to_patch_attn(
                    query=prompt_embeddings,
                    key=local_feats,
                    value=local_feats,
                )

                # Loại bỏ embedding của prompt padding để tránh nhiễu downstream
                concept_features = concept_features * prompt_mask.unsqueeze(-1).type_as(concept_features)

        return {
            "image_embeddings": image_embeddings, # Giữ lại nếu cần cho Prompt Encoder của SAM
            "global_features": global_feat,       # Đưa vào classifier 14 nhãn
            "global_logits": global_logits,
            "entity_logits": entity_logits,                       # (B, num_entity_labels)
            "entity_finding_logits": entity_finding_logits,       # (B, num_entity_finding_labels)
            "entity_anatomy_logits": entity_anatomy_logits,       # (B, 29)
            # Backward-compatible alias.
            "entity_disease_logits": entity_finding_logits,
            "local_features": local_feats,        # (B, 4096, embed_dim) -> Đưa vào GNN
            "prompt_embeddings": prompt_embeddings, # (B, P_max, embed_dim) từ text prompts
            "concept_features": concept_features,   # (B, P_max, embed_dim) text-guided visual entities
            "prompt_mask": prompt_mask             # (B, P_max) True với prompt hợp lệ
        }

# --- TEST THỬ MÔ HÌNH ---
# if __name__ == "__main__":
#     dummy_images = torch.randn(2, 3, 1024, 1024) # Batch = 2
#     model = MedSAM_VisionEncoder(embed_dim=512)
#     outputs = model(dummy_images)
#     
#     print("Global Features:", outputs["global_features"].shape) # (2, 512)
#     print("Local Features:", outputs["local_features"].shape)   # (2, 4096, 512)