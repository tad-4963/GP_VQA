import modal
import os
import itertools
from collections import deque
# ==========================================
# 1. CẤU HÌNH MODAL APP & THƯ VIỆN
# ==========================================
app = modal.App("med-vqa-training")

vol_weights = modal.Volume.from_name("med-vqa-weights")
vol_data = modal.Volume.from_name("med-vqa-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0", # Nâng cấp PyTorch lên bản hỗ trợ Numpy 2.x
        "torchvision>=0.19.0", # Nâng cấp theo PyTorch
        "transformers",
        "pandas",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "peft",
        "wandb",
        "timm",
        "iopath",
        "einops",
        "decord",
        "hydra-core",
        "omegaconf",
        "submitit",
        "open-clip-torch",
        "ftfy",
        "regex",
        "psutil",
        "torchmetrics",
        "opencv-python-headless",
        "scipy",
        "matplotlib",
        "scikit-image",
        "pycocotools",
        "numpy<2" # (Tùy chọn) Chốt cứng Numpy ở bản 1.x cho chắc ăn
    )
    .add_local_dir("/home/laptopdev/GP_VQA/src", remote_path="/root/src")
    .add_local_dir("/home/laptopdev/GP_VQA/external/MedSAM3", remote_path="/root/external/MedSAM3")
)

# ==========================================
# 2. HÀM LOSS
# ==========================================
def soft_clip_contrastive_loss(query_features, key_features, query_labels, key_labels=None):
    import torch
    import torch.nn.functional as F

    if key_labels is None:
        key_labels = query_labels
    else:
        key_labels = key_labels.to(device=query_labels.device, dtype=query_labels.dtype)

    logits = query_features @ key_features.t()
    similarity = query_labels @ key_labels.t()

    if logits.shape[0] == logits.shape[1]:
        similarity = similarity + torch.eye(
            logits.shape[0],
            device=query_labels.device,
            dtype=similarity.dtype,
        )

    target_probs = similarity / similarity.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probs = F.log_softmax(logits, dim=-1)

    return -(target_probs * log_probs).sum(dim=-1).mean()

# ==========================================
# 3. HÀM HUẤN LUYỆN CHÍNH
# ==========================================
@app.function(
    image=image,
    # gpu="a100-80gb",
    gpu="h200",
    volumes={
        "/data/weights": vol_weights, 
        "/data/dataset": vol_data
    },
    secrets=[modal.Secret.from_name("wandb-secret")], 
    timeout=86400,
)
def train_model(
    use_lora: bool = True,
    require_attention_lora: bool = False,
    encoder_backend: str = "transformers",
    debug_mode: bool = False,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import roc_auc_score, f1_score
    import numpy as np
    import pandas as pd
    import warnings
    import wandb 
    from tqdm import tqdm  # Import thanh tiến trình
    warnings.filterwarnings('ignore')

    def safe_macro_auc(y_true, y_prob):
        """Macro AUC chỉ trên các nhãn có cả positive và negative để tránh NaN."""
        auc_values = []
        for idx in range(y_true.shape[1]):
            y_col = y_true[:, idx]
            if np.unique(y_col).size < 2:
                continue
            try:
                auc_val = roc_auc_score(y_col, y_prob[:, idx])
            except ValueError:
                continue
            if np.isfinite(auc_val):
                auc_values.append(float(auc_val))

        if not auc_values:
            return 0.0, 0
        return float(np.mean(auc_values)), len(auc_values)
    
    from src.datasets.mimic_dataset import MedicalVQADataset, medical_vqa_collate_fn
    from src.models.vision.medsam_encoder import MedSAM_VisionEncoder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Khởi động huấn luyện trên: {device} - {torch.cuda.get_device_name(0)}")

    # Tối ưu hóa GPU PyTorch 2.x
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # ================= CẤU HÌNH =================
    image_size = 1008  # Giữ nguyên 1008 cho SAM3
    train_bs = 4
    val_bs = 4
    num_workers = 8
    accumulation_steps = 32
    contrastive_bank_size = 1024
    epochs = 5
    alpha_contrastive = 0.5 
    alpha_entity = 0.5
    use_lora = bool(use_lora)
    DEBUG_MODE = bool(debug_mode)
    if DEBUG_MODE:
        epochs = 1
    # ============================================

    wandb.init(
        project="Med-VQA-Vision", 
        name=f"H200_MedSAM3_Training_{'with_lora' if use_lora else 'no_lora'}",
        config={
            "epochs": epochs,
            "batch_size": train_bs,
            "accumulation_steps": accumulation_steps,
            "learning_rate_encoder": 1e-5,
            "learning_rate_classifier": 1e-3,
            "image_size": image_size,
            "alpha_entity": alpha_entity,
            "use_lora": use_lora,
            "debug_mode": DEBUG_MODE
        }
    )

    print("Đang tải dữ liệu...")
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    csv_file = os.environ.get("MEDSAM_CSV_FILE", "/data/dataset/mimic_all_final.csv")
    json_candidates = [
        os.environ.get("MEDSAM_JSON_FILE", "").strip(),
        "/data/dataset//all_diseases_final.json",
    ]
    json_file = next((p for p in json_candidates if p and os.path.exists(p)), None)
    if json_file is None:
        raise FileNotFoundError("Không tìm thấy JSON entities trên volume data.")

    img_root = os.environ.get("MEDSAM_IMG_ROOT", "/data/dataset/mimic-cxr-kaggle/images")

    csv_df = pd.read_csv(csv_file)
    if "split" not in csv_df.columns:
        raise ValueError("CSV thiếu cột 'split'. Hãy thêm split train/valid/test vào file mimic_cxr_balanced.csv")

    split_norm = (
        csv_df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"val": "valid", "validation": "valid", "validate": "valid"})
    )
    split_counts = split_norm.value_counts()
    split_pct = (split_counts / max(len(csv_df), 1) * 100).round(2)
    print("Phân bố split trong CSV (normalized):")
    for split_name in ["train", "valid", "test"]:
        c = int(split_counts.get(split_name, 0))
        p = float(split_pct.get(split_name, 0.0))
        print(f" - {split_name}: {c} ({p:.2f}%)")

    train_dataset = MedicalVQADataset(
        csv_file=csv_file,
        json_file=json_file,
        img_dir=img_root,
        norm_disease_csv="/data/dataset/label/normalized_diseases.csv",
        norm_anatomy_csv="/data/dataset/label/normalized_anatomy.csv",
        transform=transform,
        split="train",
    )
    val_dataset = MedicalVQADataset(
        csv_file=csv_file,
        json_file=json_file,
        img_dir=img_root,
        norm_disease_csv="/data/dataset/label/normalized_diseases.csv",
        norm_anatomy_csv="/data/dataset/label/normalized_anatomy.csv",
        transform=transform,
        split="validate",
    )

    print(f"Số mẫu theo split (dataset loader): train={len(train_dataset)} | valid={len(val_dataset)}")
    if len(val_dataset) < 500 and not DEBUG_MODE:
        print("⚠️ CẢNH BÁO: valid set đang khá nhỏ (<500). Metric có thể dao động mạnh.")

    # CẮT DATASET NẾU BẬT DEBUG MODE
    if DEBUG_MODE:
        print("🚧 ĐANG CHẠY Ở CHẾ ĐỘ DEBUG: Cắt train/valid để test luồng...")
        train_subset_size = min(400, len(train_dataset))
        val_subset_size = min(200, len(val_dataset))

        train_indices = torch.randperm(len(train_dataset))[:train_subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset))[:val_subset_size].tolist()

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True,
        collate_fn=medical_vqa_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        collate_fn=medical_vqa_collate_fn,
    )
    
    print("Khởi tạo MedSAM Vision Language Model...")
    lora_path = "/data/weights/medsam3_v1_lora" if use_lora else None
    print(f"LoRA mode: {'ON' if use_lora else 'OFF'} | lora_weights_path={lora_path}")
    print(f"Require attention LoRA compatibility: {require_attention_lora}")
    print(f"Encoder backend: {encoder_backend}")
    entity_meta_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    num_global_labels = len(entity_meta_dataset.disease_cols)
    num_entity_finding_labels = len(entity_meta_dataset.raw_finding_labels)
    num_entity_anatomy_labels = len(entity_meta_dataset.raw_anatomy_labels)
    print(
        f"Label dims from dataset: global={num_global_labels}, findings={num_entity_finding_labels}, anatomy={num_entity_anatomy_labels}"
    )
    model = MedSAM_VisionEncoder(
        sam3_base_checkpoint="/data/weights/sam3", 
        lora_weights_path=lora_path,
        embed_dim=256,
        freeze_encoder=False,
        use_text_guidance=True,
        num_global_disease_labels=num_global_labels,
        num_entity_finding_labels=num_entity_finding_labels,
        num_entity_anatomy_labels=num_entity_anatomy_labels,
        require_attention_lora=require_attention_lora,
        encoder_backend=encoder_backend,
        medsam3_repo_path="/root/external/MedSAM3",
    ).to(device)
    
    # Bật Gradient Checkpointing để tiết kiệm VRAM
    # if hasattr(model.image_encoder, "gradient_checkpointing_enable"):
    #     model.image_encoder.gradient_checkpointing_enable()
    
    # Tính lại pos_weight theo tập train hiện tại (sau cân bằng/lọc split)
    base_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    label_cols = base_dataset.disease_cols
    if hasattr(train_dataset, "indices") and hasattr(base_dataset, "df"):
        train_labels_df = base_dataset.df.iloc[train_dataset.indices][label_cols].fillna(0.0).astype(float)
    else:
        train_labels_df = base_dataset.df[label_cols].fillna(0.0).astype(float)
    train_labels_df = (train_labels_df > 0).astype(float)
    pos_counts = train_labels_df.sum(axis=0).to_numpy(dtype=np.float32)
    total_samples = float(len(train_labels_df))
    neg_counts = np.maximum(total_samples - pos_counts, 0.0)
    pos_counts = np.maximum(pos_counts, 1.0)
    dynamic_pos_weight = neg_counts / pos_counts

    pos_weight_tensor = torch.tensor(dynamic_pos_weight, dtype=torch.float32, device=device)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=8.0)
    print("Pos weight (dynamic, clamped):", [round(float(v), 3) for v in pos_weight_tensor.tolist()])
    criterion_global = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Tính pos_weight cho nhánh finding+anatomy (số nhãn động theo dataset) từ tập train hiện tại.
    entity_dataset = base_dataset
    if hasattr(train_dataset, "indices"):
        entity_indices = train_dataset.indices
        entity_rows = entity_dataset.df.iloc[entity_indices].reset_index(drop=True)
    else:
        entity_rows = entity_dataset.df.reset_index(drop=True)

    num_entity_labels = len(entity_dataset.raw_finding_labels) + len(entity_dataset.raw_anatomy_labels)
    entity_pos_counts = np.zeros(num_entity_labels, dtype=np.float32)
    for _, entity_row in entity_rows.iterrows():
        row_study_id = entity_dataset.resolve_study_id(entity_row)
        if row_study_id in entity_dataset.json_dict:
            _, finding_hits, anatomy_hits, _ = entity_dataset._extract_text_prompts(entity_dataset.json_dict[row_study_id])
            entity_vec = torch.cat([finding_hits, anatomy_hits], dim=0).numpy().astype(np.float32)
            entity_pos_counts += entity_vec

    entity_total = float(len(entity_rows))
    entity_neg_counts = np.maximum(entity_total - entity_pos_counts, 0.0)
    entity_pos_counts = np.maximum(entity_pos_counts, 1.0)
    entity_pos_weight_np = entity_neg_counts / entity_pos_counts
    entity_pos_weight_tensor = torch.tensor(entity_pos_weight_np, dtype=torch.float32, device=device)
    entity_pos_weight_tensor = torch.clamp(entity_pos_weight_tensor, min=1.0, max=20.0)
    print(
        "Entity pos_weight (dynamic, clamped) stats:",
        {
            "min": round(float(entity_pos_weight_tensor.min().item()), 3),
            "max": round(float(entity_pos_weight_tensor.max().item()), 3),
            "mean": round(float(entity_pos_weight_tensor.mean().item()), 3),
        }
    )
    criterion_entity = nn.BCEWithLogitsLoss(pos_weight=entity_pos_weight_tensor)
    
    encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    head_params = list(itertools.chain(model.global_proj.parameters(), model.classifier.parameters()))
    if hasattr(model, "entity_classifier"):
        head_params.extend(list(model.entity_classifier.parameters()))

    # Contrastive branch gồm local/text projector + cross-attention cần được tối ưu, nếu không loss contrastive sẽ rất nhiễu.
    contrastive_branch_params = list(model.local_proj.parameters())
    if getattr(model, "text_proj", None) is not None:
        contrastive_branch_params.extend(list(model.text_proj.parameters()))
    if getattr(model, "text_to_patch_attn", None) is not None:
        contrastive_branch_params.extend(list(model.text_to_patch_attn.parameters()))

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3},
        {'params': contrastive_branch_params, 'lr': 5e-4}
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Sẵn sàng huấn luyện! Batch = {train_bs} | Accumulation = {accumulation_steps}")
    best_val_loss = float('inf') 

    # Memory bank giúp contrastive loss có thêm negatives dù micro-batch nhỏ.
    img_feat_bank = deque()
    text_feat_bank = deque()
    label_bank = deque()
    bank_sample_count = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        img_feat_bank.clear()
        text_feat_bank.clear()
        label_bank.clear()
        bank_sample_count = 0
        
        # ==========================================
        # TRAINING LOOP
        # ==========================================
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        # Bọc DataLoader vào tqdm để hiển thị thanh tiến trình
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = (batch['global_labels'] > 0).to(device=device, dtype=torch.float32)
            finding_multihot_labels = batch.get('entity_multihot_labels', batch.get('entity_labels')).to(device)
            text_prompts = batch['local_prompts']

            # ÉP XUNG BFLOAT16 (Tăng tốc x2, Giảm RAM x2)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images, local_prompts=text_prompts)
                
                # Tính BCE Loss dựa trên số nhãn global hiện tại
                loss_bce = criterion_global(outputs["global_logits"], labels)
                loss_entity = criterion_entity(outputs["entity_logits"], finding_multihot_labels)
                
                # Tính Contrastive Loss dựa trên đặc trưng 256 chiều
                if outputs["concept_features"] is not None:
                    text_global = outputs["concept_features"].mean(dim=1) 
                else:
                    text_global = outputs["global_features"]

                img_features = torch.nn.functional.normalize(outputs["global_features"].float(), dim=-1)
                text_features = torch.nn.functional.normalize(text_global.float(), dim=-1)
                labels_float = labels.float()

                if bank_sample_count > 0:
                    bank_img = torch.cat(list(img_feat_bank), dim=0).to(device)
                    bank_txt = torch.cat(list(text_feat_bank), dim=0).to(device)
                    bank_lbl = torch.cat(list(label_bank), dim=0).to(device)

                    key_txt = torch.cat([text_features, bank_txt], dim=0)
                    key_img = torch.cat([img_features, bank_img], dim=0)
                    key_lbl = torch.cat([labels_float, bank_lbl], dim=0)
                else:
                    key_txt = text_features
                    key_img = img_features
                    key_lbl = labels_float

                loss_i2t = soft_clip_contrastive_loss(img_features, key_txt, labels_float, key_lbl)
                loss_t2i = soft_clip_contrastive_loss(text_features, key_img, labels_float, key_lbl)
                loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                
                # Tổng hợp Loss
                loss = loss_bce + alpha_entity * loss_entity + alpha_contrastive * loss_contrastive
                loss = loss / accumulation_steps 
                
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps
            
            # Cập nhật số Loss liên tục lên thanh tiến trình
            pbar.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})
            
            # Log lên WandB mỗi 20 batch
            if i % 20 == 0:
                wandb.log({
                    "batch_train_loss": loss.item() * accumulation_steps,
                    "batch_loss_bce": loss_bce.item(),
                    "batch_loss_entity": loss_entity.item(),
                    "batch_loss_contrastive": loss_contrastive.item(),
                })

            # Cập nhật memory bank sau khi đã tính loss để tránh tự so khớp vòng lặp hiện tại.
            img_feat_bank.append(img_features.detach().cpu())
            text_feat_bank.append(text_features.detach().cpu())
            label_bank.append(labels_float.detach().cpu())
            bank_sample_count += img_features.shape[0]

            while bank_sample_count > contrastive_bank_size:
                removed_img = img_feat_bank.popleft()
                text_feat_bank.popleft()
                label_bank.popleft()
                bank_sample_count -= removed_img.shape[0]
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        model.eval()
        total_val_loss = 0.0
        all_probs, all_labels = [], []
        all_finding_probs, all_finding_labels = [], []
        
        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(device)
                labels = (batch['global_labels'] > 0).to(device=device, dtype=torch.float32)
                finding_multihot_labels = batch.get('entity_multihot_labels', batch.get('entity_labels')).to(device)
                text_prompts = batch['local_prompts']

                # Bật Autocast lúc test để chạy nhanh hơn
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(images, local_prompts=text_prompts)
                    loss_bce = criterion_global(outputs["global_logits"], labels)
                    loss_entity = criterion_entity(outputs["entity_logits"], finding_multihot_labels)
                    
                    if outputs["concept_features"] is not None:
                        text_global = outputs["concept_features"].mean(dim=1) 
                    else:
                        text_global = outputs["global_features"]

                    img_features = torch.nn.functional.normalize(outputs["global_features"].float(), dim=-1)
                    text_features = torch.nn.functional.normalize(text_global.float(), dim=-1)
                    labels_float = labels.float()

                    loss_i2t = soft_clip_contrastive_loss(img_features, text_features, labels_float)
                    loss_t2i = soft_clip_contrastive_loss(text_features, img_features, labels_float)
                    loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                    loss = loss_bce + alpha_entity * loss_entity + alpha_contrastive * loss_contrastive
                    
                total_val_loss += loss.item()
                
                # Tính xác suất từ Logits
                probs = torch.sigmoid(outputs["global_logits"])
                finding_probs = torch.sigmoid(outputs["entity_logits"])
                
                # Ép kiểu về float32 để Numpy (scikit-learn) không bị hoảng loạn
                all_probs.append(probs.detach().cpu().to(torch.float32).numpy())
                all_labels.append(labels.detach().cpu().to(torch.float32).numpy())
                all_finding_probs.append(finding_probs.detach().cpu().to(torch.float32).numpy())
                all_finding_labels.append(finding_multihot_labels.detach().cpu().to(torch.float32).numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        all_probs_np = np.concatenate(all_probs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        all_finding_probs_np = np.concatenate(all_finding_probs, axis=0)
        all_finding_labels_np = np.concatenate(all_finding_labels, axis=0)
        
        val_auc, valid_auc_classes = safe_macro_auc(all_labels_np, all_probs_np)
        val_entity_auc, valid_entity_auc_classes = safe_macro_auc(all_finding_labels_np, all_finding_probs_np)
            
        preds_np = (all_probs_np > 0.5).astype(int)
        finding_preds_np = (all_finding_probs_np > 0.5).astype(int)
        positive_rate = float(preds_np.mean())
        finding_positive_rate = float(finding_preds_np.mean())
        val_f1 = f1_score(all_labels_np, preds_np, average='macro', zero_division=0)
        val_entity_f1 = f1_score(all_finding_labels_np, finding_preds_np, average='macro', zero_division=0)
        
        print(f"\nKẾT QUẢ EPOCH {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        print(f"AUC valid classes (global): {valid_auc_classes}/{num_global_labels}")
        print(f"Entity Val AUC: {val_entity_auc:.4f} | Entity Val F1: {val_entity_f1:.4f}")
        print(f"AUC valid classes (entity): {valid_entity_auc_classes}/{all_finding_labels_np.shape[1]}")
        print(f"Positive prediction rate (@0.5): {positive_rate:.4f}")
        print(f"Finding positive prediction rate (@0.5): {finding_positive_rate:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_entity_auc": val_entity_auc,
            "val_entity_f1": val_entity_f1,
            "val_positive_rate": positive_rate,
            "val_finding_positive_rate": finding_positive_rate,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = "/data/weights/medvqa_vision_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Đã lưu Best Model tại Epoch {epoch+1} (Val Loss: {best_val_loss:.4f})")

        scheduler.step()

    torch.save(model.state_dict(), "/data/weights/medvqa_vision_last.pth")
    print("\n✅ Tiến trình huấn luyện hoàn tất!")
    wandb.finish()

@app.local_entrypoint()
def main():
    print("Chuẩn bị...")
    use_lora = os.environ.get("USE_LORA", "1") == "1"
    require_attention_lora = os.environ.get("REQUIRE_ATTENTION_LORA", "0") == "1"
    encoder_backend = os.environ.get("ENCODER_BACKEND", "transformers").strip().lower()
    debug_mode = os.environ.get("DEBUG_MODE", "0") == "1"
    print(f"USE_LORA from local env: {use_lora}")
    print(f"REQUIRE_ATTENTION_LORA from local env: {require_attention_lora}")
    print(f"ENCODER_BACKEND from local env: {encoder_backend}")
    print(f"DEBUG_MODE from local env: {debug_mode}")
    train_model.remote(
        use_lora=use_lora,
        require_attention_lora=require_attention_lora,
        encoder_backend=encoder_backend,
        debug_mode=debug_mode,
    )
