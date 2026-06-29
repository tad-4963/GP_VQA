import modal
import os
import itertools
import random
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
        "huggingface_hub",
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


def asymmetric_loss_with_logits(
    logits,
    targets,
    pos_weight=None,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
):
    import torch

    targets = targets.to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    pos_prob = probs
    neg_prob = 1.0 - probs
    if clip and clip > 0.0:
        neg_prob = (neg_prob + float(clip)).clamp(max=1.0)

    pos_loss = targets * torch.log(pos_prob.clamp_min(eps))
    neg_loss = (1.0 - targets) * torch.log(neg_prob.clamp_min(eps))

    if gamma_pos > 0.0 or gamma_neg > 0.0:
        pt = pos_prob * targets + neg_prob * (1.0 - targets)
        gamma = float(gamma_pos) * targets + float(gamma_neg) * (1.0 - targets)
        focal_weight = (1.0 - pt).clamp_min(0.0).pow(gamma)
        pos_loss = pos_loss * focal_weight
        neg_loss = neg_loss * focal_weight

    loss = pos_loss + neg_loss
    if pos_weight is not None:
        loss = torch.where(targets > 0.5, loss * pos_weight.view(1, -1).to(device=logits.device, dtype=logits.dtype), loss)
    return -loss

def pairwise_ranking_loss_with_logits(
    logits,
    targets,
    label_mask=None,
    label_weight=None,
):
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    targets = targets.float()
    if label_mask is None:
        label_mask = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
    else:
        label_mask = label_mask.float()

    losses = []
    weights = []
    for class_idx in range(targets.shape[1]):
        valid = label_mask[:, class_idx] > 0.5
        pos_logits = logits[(targets[:, class_idx] > 0.5) & valid, class_idx]
        neg_logits = logits[(targets[:, class_idx] <= 0.5) & valid, class_idx]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue

        class_loss = F.softplus(-(pos_logits[:, None] - neg_logits[None, :])).mean()
        if label_weight is None:
            class_weight = logits.new_tensor(1.0)
        else:
            class_weight = label_weight[class_idx].to(device=logits.device, dtype=logits.dtype)
        losses.append(class_loss * class_weight)
        weights.append(class_weight)

    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1.0)

# ==========================================
# 3. HÀM HUẤN LUYỆN CHÍNH
# ==========================================
@app.function(
    image=image,
    # gpu="a100-80gb",
    gpu="h100",
    volumes={
        "/data/weights": vol_weights, 
        "/data/dataset": vol_data
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("my-huggingface-secret"),
    ],
    timeout=86400,
)
def train_model(
    use_lora: bool = True,
    require_attention_lora: bool = False,
    encoder_backend: str = "transformers",
    debug_mode: bool = False,
    debug_epochs: int = 0,
    global_only: int = -1,
    alpha_global_arg: float = -1.0,
    alpha_entity_arg: float = -1.0,
    alpha_contrastive_arg: float = -1.0,
    uncertain_policy_arg: str = "",
    exclude_no_finding: int = -1,
    init_checkpoint_arg: str = "",
    use_text_guidance: int = -1,
    use_local_entity_head: int = -1,
    local_entity_merge_arg: str = "",
    entity_pooling_arg: str = "",
    global_pooling_arg: str = "",
    global_head_arg: str = "",
    global_head_dropout_arg: float = -1.0,
    encoder_lr_arg: float = -1.0,
    head_lr_arg: float = -1.0,
    contrastive_lr_arg: float = -1.0,
    accumulation_steps_arg: int = 0,
    global_loss_arg: str = "",
    asl_gamma_pos_arg: float = -1.0,
    asl_gamma_neg_arg: float = -1.0,
    asl_clip_arg: float = -1.0,
    asl_use_pos_weight_arg: int = -1,
    global_rank_loss_weight_arg: float = -1.0,
    hard_global_labels_arg: str = "",
    hard_global_loss_boost_arg: float = -1.0,
    global_sampler_arg: str = "",
    global_sampler_boost_arg: float = -1.0,
    teacher_checkpoint_arg: str = "",
    distill_weight_arg: float = -1.0,
    distill_temperature_arg: float = -1.0,
    distill_mask_arg: str = "",
    output_prefix_arg: str = "",
    entity_pos_weight_max_arg: float = -1.0,
    seed_arg: int = 0,
    debug_train_size: int = 0,
    debug_val_size: int = 0,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    import numpy as np
    import pandas as pd
    import warnings
    import wandb 
    from tqdm import tqdm  # Import thanh tiến trình
    warnings.filterwarnings('ignore')

    def safe_macro_auc(y_true, y_prob, y_mask=None):
        """Macro AUC chỉ trên các nhãn có cả positive và negative để tránh NaN."""
        auc_values = []
        if y_mask is None:
            y_mask = np.ones_like(y_true, dtype=np.float32)
        for idx in range(y_true.shape[1]):
            valid = y_mask[:, idx] > 0.5
            if valid.sum() < 2:
                continue
            y_col = y_true[valid, idx]
            if np.unique(y_col).size < 2:
                continue
            try:
                auc_val = roc_auc_score(y_col, y_prob[valid, idx])
            except ValueError:
                continue
            if np.isfinite(auc_val):
                auc_values.append(float(auc_val))

        if not auc_values:
            return 0.0, 0
        return float(np.mean(auc_values)), len(auc_values)

    def safe_macro_ap(y_true, y_prob, y_mask=None):
        """Macro average precision trên các nhãn có ít nhất một positive hợp lệ."""
        ap_values = []
        for class_idx in range(y_true.shape[1]):
            if y_mask is None:
                valid = np.ones(y_true.shape[0], dtype=bool)
            else:
                valid = y_mask[:, class_idx] > 0.5
            if valid.sum() == 0:
                continue
            yt = y_true[valid, class_idx]
            yp = y_prob[valid, class_idx]
            if len(np.unique(yt)) < 2:
                continue
            ap_values.append(average_precision_score(yt, yp))
        if not ap_values:
            return 0.0, 0
        return float(np.mean(ap_values)), len(ap_values)

    def safe_macro_f1(y_true, y_prob, y_mask=None, threshold=0.5):
        if y_mask is None:
            y_mask = np.ones_like(y_true, dtype=np.float32)
        pred = (y_prob >= threshold).astype(np.float32)
        f1_values = []
        for idx in range(y_true.shape[1]):
            valid = y_mask[:, idx] > 0.5
            if valid.sum() < 2:
                continue
            yt = y_true[valid, idx]
            yp = pred[valid, idx]
            tp = float(((yt == 1.0) & (yp == 1.0)).sum())
            fp = float(((yt == 0.0) & (yp == 1.0)).sum())
            fn = float(((yt == 1.0) & (yp == 0.0)).sum())
            denom = 2.0 * tp + fp + fn
            f1_values.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1_values)) if f1_values else 0.0, len(f1_values)

    def safe_sklearn_macro_f1(y_true, y_pred):
        if y_true.shape[1] == 0:
            return 0.0
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    from src.datasets.mimic_dataset import MedicalVQADataset, medical_vqa_collate_fn
    from src.models.vision.medsam_encoder import MedSAM_VisionEncoder
    from src.models.vision.simple_encoder import RadDinoAdapter
    
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
    accumulation_steps = int(accumulation_steps_arg or os.environ.get("MEDSAM_ACCUMULATION_STEPS", "32"))
    contrastive_bank_size = 1024
    epochs = 5
    global_only_mode = (
        os.environ.get("MEDSAM_GLOBAL_ONLY", "0").strip() == "1"
        if global_only < 0
        else bool(global_only)
    )
    alpha_global = (
        float(os.environ.get("MEDSAM_ALPHA_GLOBAL", "0.0"))
        if alpha_global_arg < 0.0
        else float(alpha_global_arg)
    )
    alpha_contrastive = (
        float(os.environ.get("MEDSAM_ALPHA_CONTRASTIVE", "0.0"))
        if alpha_contrastive_arg < 0.0
        else float(alpha_contrastive_arg)
    )
    alpha_entity = (
        float(os.environ.get("MEDSAM_ALPHA_ENTITY", "0.0" if global_only_mode else "1.0"))
        if alpha_entity_arg < 0.0
        else float(alpha_entity_arg)
    )
    uncertain_policy = (uncertain_policy_arg or os.environ.get("MEDSAM_UNCERTAIN_POLICY", "to_zero")).strip().lower()
    exclude_no_finding = (
        os.environ.get("MEDSAM_EXCLUDE_NO_FINDING", "1").strip() == "1"
        if exclude_no_finding < 0
        else bool(exclude_no_finding)
    )
    init_checkpoint = (init_checkpoint_arg or os.environ.get("MEDSAM_INIT_CHECKPOINT", "")).strip()
    use_text_guidance = (
        os.environ.get("MEDSAM_USE_TEXT_GUIDANCE", "1" if alpha_contrastive > 0.0 else "0").strip() == "1"
        if use_text_guidance < 0
        else bool(use_text_guidance)
    )
    default_use_local_entity_head = "1" if (not global_only_mode and alpha_entity > 0.0) else "0"
    use_local_entity_head = (
        os.environ.get("MEDSAM_USE_LOCAL_ENTITY_HEAD", default_use_local_entity_head).strip() == "1"
        if use_local_entity_head < 0
        else bool(use_local_entity_head)
    )
    local_entity_merge = (local_entity_merge_arg or os.environ.get("MEDSAM_LOCAL_ENTITY_MERGE", "anatomy")).strip().lower()
    if local_entity_merge not in {"all", "anatomy"}:
        raise ValueError("MEDSAM_LOCAL_ENTITY_MERGE phải là 'all' hoặc 'anatomy'.")
    entity_pooling = (entity_pooling_arg or os.environ.get("MEDSAM_ENTITY_POOLING", "global")).strip().lower()
    if entity_pooling not in {"global", "meanmax"}:
        raise ValueError("MEDSAM_ENTITY_POOLING phải là 'global' hoặc 'meanmax'.")
    global_pooling = (global_pooling_arg or os.environ.get("MEDSAM_GLOBAL_POOLING", "mean")).strip().lower()
    if global_pooling not in {"mean", "max", "meanmax", "attn", "attn_meanmax"}:
        raise ValueError("MEDSAM_GLOBAL_POOLING phải là 'mean', 'max', 'meanmax', 'attn', hoặc 'attn_meanmax'.")
    global_head = (global_head_arg or os.environ.get("MEDSAM_GLOBAL_HEAD", "linear")).strip().lower()
    if global_head not in {"linear", "mlp"}:
        raise ValueError("MEDSAM_GLOBAL_HEAD phải là 'linear' hoặc 'mlp'.")
    global_head_dropout = (
        float(os.environ.get("MEDSAM_GLOBAL_HEAD_DROPOUT", "0.1"))
        if global_head_dropout_arg < 0.0
        else float(global_head_dropout_arg)
    )
    encoder_lr = (
        float(os.environ.get("MEDSAM_ENCODER_LR", "1e-5"))
        if encoder_lr_arg < 0.0
        else float(encoder_lr_arg)
    )
    head_lr = (
        float(os.environ.get("MEDSAM_HEAD_LR", "1e-3"))
        if head_lr_arg < 0.0
        else float(head_lr_arg)
    )
    contrastive_lr = (
        float(os.environ.get("MEDSAM_CONTRASTIVE_LR", "5e-4"))
        if contrastive_lr_arg < 0.0
        else float(contrastive_lr_arg)
    )
    global_loss_type = (global_loss_arg or os.environ.get("MEDSAM_GLOBAL_LOSS", "bce")).strip().lower()
    if global_loss_type not in {"bce", "asl"}:
        raise ValueError("MEDSAM_GLOBAL_LOSS phải là 'bce' hoặc 'asl'.")
    asl_gamma_pos = (
        float(os.environ.get("MEDSAM_ASL_GAMMA_POS", "1.0"))
        if asl_gamma_pos_arg < 0.0
        else float(asl_gamma_pos_arg)
    )
    asl_gamma_neg = (
        float(os.environ.get("MEDSAM_ASL_GAMMA_NEG", "4.0"))
        if asl_gamma_neg_arg < 0.0
        else float(asl_gamma_neg_arg)
    )
    asl_clip = (
        float(os.environ.get("MEDSAM_ASL_CLIP", "0.05"))
        if asl_clip_arg < 0.0
        else float(asl_clip_arg)
    )
    asl_use_pos_weight = (
        os.environ.get("MEDSAM_ASL_USE_POS_WEIGHT", "0").strip() == "1"
        if asl_use_pos_weight_arg < 0
        else bool(asl_use_pos_weight_arg)
    )
    global_rank_loss_weight = (
        float(os.environ.get("MEDSAM_GLOBAL_RANK_LOSS_WEIGHT", "0.0"))
        if global_rank_loss_weight_arg < 0.0
        else float(global_rank_loss_weight_arg)
    )
    hard_global_labels = [
        label.strip()
        for label in (hard_global_labels_arg or os.environ.get("MEDSAM_HARD_GLOBAL_LABELS", "")).split(",")
        if label.strip()
    ]
    hard_global_loss_boost = (
        float(os.environ.get("MEDSAM_HARD_GLOBAL_LOSS_BOOST", "1.0"))
        if hard_global_loss_boost_arg < 0.0
        else float(hard_global_loss_boost_arg)
    )
    global_sampler = (global_sampler_arg or os.environ.get("MEDSAM_GLOBAL_SAMPLER", "none")).strip().lower()
    if global_sampler not in {"none", "hard"}:
        raise ValueError("MEDSAM_GLOBAL_SAMPLER phải là 'none' hoặc 'hard'.")
    global_sampler_boost = (
        float(os.environ.get("MEDSAM_GLOBAL_SAMPLER_BOOST", "1.0"))
        if global_sampler_boost_arg < 0.0
        else float(global_sampler_boost_arg)
    )
    teacher_checkpoint = (
        teacher_checkpoint_arg
        or os.environ.get("MEDSAM_TEACHER_CHECKPOINT", "")
    ).strip()
    distill_weight = (
        float(os.environ.get("MEDSAM_DISTILL_WEIGHT", "0.0"))
        if distill_weight_arg < 0.0
        else float(distill_weight_arg)
    )
    if distill_weight > 0.0 and not teacher_checkpoint:
        teacher_checkpoint = "/data/weights/rad_dino_linear_adapter_best.pth"
    distill_temperature = (
        float(os.environ.get("MEDSAM_DISTILL_TEMPERATURE", "1.0"))
        if distill_temperature_arg < 0.0
        else float(distill_temperature_arg)
    )
    distill_temperature = max(float(distill_temperature), 1e-3)
    distill_mask = (distill_mask_arg or os.environ.get("MEDSAM_DISTILL_MASK", "valid")).strip().lower()
    if distill_mask not in {"valid", "all"}:
        raise ValueError("MEDSAM_DISTILL_MASK phải là 'valid' hoặc 'all'.")
    output_prefix = (
        output_prefix_arg
        or os.environ.get("MEDSAM_OUTPUT_PREFIX", "/data/weights/medvqa_vision")
    ).strip()
    if output_prefix.endswith(".pth"):
        output_prefix = output_prefix[:-4]
    entity_pos_weight_max = (
        float(os.environ.get("MEDSAM_ENTITY_POS_WEIGHT_MAX", "10.0"))
        if entity_pos_weight_max_arg < 0.0
        else float(entity_pos_weight_max_arg)
    )
    seed = int(seed_arg or os.environ.get("MEDSAM_SEED", "42"))
    use_lora = bool(use_lora)
    DEBUG_MODE = bool(debug_mode)
    if DEBUG_MODE:
        epochs = min(epochs, int(debug_epochs or os.environ.get("MEDSAM_DEBUG_EPOCHS", "3")))
    if alpha_entity <= 0.0:
        print(
            "⚠️ MEDSAM_ALPHA_ENTITY <= 0: entity head sẽ không nhận gradient từ entity loss. "
            "Đặt MEDSAM_GLOBAL_ONLY=0 và MEDSAM_ALPHA_ENTITY>0 để train entity."
        )
    if not global_only_mode and alpha_entity <= 0.0:
        raise ValueError("MEDSAM_GLOBAL_ONLY=0 nhưng MEDSAM_ALPHA_ENTITY<=0, entity head sẽ không học.")
    # ============================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb.init(
        project="Med-VQA-Vision", 
        name=f"H200_MedSAM3_Training_{'with_lora' if use_lora else 'no_lora'}",
        config={
            "epochs": epochs,
            "batch_size": train_bs,
            "accumulation_steps": accumulation_steps,
            "learning_rate_encoder": encoder_lr,
            "learning_rate_classifier": head_lr,
            "learning_rate_contrastive": contrastive_lr,
            "image_size": image_size,
            "alpha_global": alpha_global,
            "alpha_entity": alpha_entity,
            "alpha_contrastive": alpha_contrastive,
            "global_loss": global_loss_type,
            "asl_gamma_pos": asl_gamma_pos,
            "asl_gamma_neg": asl_gamma_neg,
            "asl_clip": asl_clip,
            "asl_use_pos_weight": asl_use_pos_weight,
            "global_rank_loss_weight": global_rank_loss_weight,
            "hard_global_labels": hard_global_labels,
            "hard_global_loss_boost": hard_global_loss_boost,
            "global_sampler": global_sampler,
            "global_sampler_boost": global_sampler_boost,
            "teacher_checkpoint": teacher_checkpoint,
            "distill_weight": distill_weight,
            "distill_temperature": distill_temperature,
            "distill_mask": distill_mask,
            "output_prefix": output_prefix,
            "entity_pos_weight_max": entity_pos_weight_max,
            "global_only_mode": global_only_mode,
            "uncertain_policy": uncertain_policy,
            "exclude_no_finding": exclude_no_finding,
            "init_checkpoint": init_checkpoint,
            "use_text_guidance": use_text_guidance,
            "use_local_entity_head": use_local_entity_head,
            "local_entity_merge": local_entity_merge,
            "entity_pooling": entity_pooling,
            "global_pooling": global_pooling,
            "global_head": global_head,
            "global_head_dropout": global_head_dropout,
            "use_lora": use_lora,
            "seed": seed,
            "debug_mode": DEBUG_MODE,
            "debug_epochs": debug_epochs,
        }
    )

    print("Đang tải dữ liệu...")
    import torchvision.transforms as T
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    image_mean_tensor = torch.tensor(image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    image_std_tensor = torch.tensor(image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
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
        transform=train_transform,
        split="train",
        uncertain_policy=uncertain_policy,
        excluded_global_labels=["No Finding"] if exclude_no_finding else None,
    )
    val_dataset = MedicalVQADataset(
        csv_file=csv_file,
        json_file=json_file,
        img_dir=img_root,
        norm_disease_csv="/data/dataset/label/normalized_diseases.csv",
        norm_anatomy_csv="/data/dataset/label/normalized_anatomy.csv",
        transform=val_transform,
        split="validate",
        uncertain_policy=uncertain_policy,
        excluded_global_labels=["No Finding"] if exclude_no_finding else None,
    )

    print(f"Số mẫu theo split (dataset loader): train={len(train_dataset)} | valid={len(val_dataset)}")
    if len(val_dataset) < 500 and not DEBUG_MODE:
        print("⚠️ CẢNH BÁO: valid set đang khá nhỏ (<500). Metric có thể dao động mạnh.")

    # CẮT DATASET NẾU BẬT DEBUG MODE
    if DEBUG_MODE:
        print("🚧 ĐANG CHẠY Ở CHẾ ĐỘ DEBUG: Cắt train/valid để test luồng...")
        train_subset_size = min(
            int(debug_train_size or os.environ.get("MEDSAM_DEBUG_TRAIN_SIZE", "400")),
            len(train_dataset),
        )
        val_subset_size = min(
            int(debug_val_size or os.environ.get("MEDSAM_DEBUG_VAL_SIZE", "200")),
            len(val_dataset),
        )

        train_gen = torch.Generator().manual_seed(seed)
        val_gen = torch.Generator().manual_seed(seed + 1)
        train_indices = torch.randperm(len(train_dataset), generator=train_gen)[:train_subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset), generator=val_gen)[:val_subset_size].tolist()

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True,
        generator=loader_gen,
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
        use_text_guidance=use_text_guidance,
        num_global_disease_labels=num_global_labels,
        num_entity_finding_labels=num_entity_finding_labels,
        num_entity_anatomy_labels=num_entity_anatomy_labels,
        use_local_entity_head=use_local_entity_head,
        local_entity_merge=local_entity_merge,
        entity_pooling=entity_pooling,
        global_pooling=global_pooling,
        global_head=global_head,
        global_head_dropout=global_head_dropout,
        require_attention_lora=require_attention_lora,
        encoder_backend=encoder_backend,
        medsam3_repo_path="/root/external/MedSAM3",
    ).to(device)

    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"MEDSAM_INIT_CHECKPOINT không tồn tại: {init_checkpoint}")
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        init_state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if isinstance(checkpoint.get(key), dict):
                    init_state = checkpoint[key]
                    break
        current_state = model.state_dict()
        compatible_state = {}
        skipped_shape_keys = []
        for key, value in init_state.items():
            if key in current_state and getattr(value, "shape", None) != current_state[key].shape:
                skipped_shape_keys.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            compatible_state[key] = value

        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
        print(
            "Loaded init checkpoint:",
            init_checkpoint,
            f"missing={len(missing_keys)}",
            f"unexpected={len(unexpected_keys)}",
            f"skipped_shape={len(skipped_shape_keys)}",
        )
        if skipped_shape_keys:
            print("Skipped checkpoint tensors with mismatched shape:")
            for key, old_shape, new_shape in skipped_shape_keys[:20]:
                print(f" - {key}: checkpoint={old_shape} model={new_shape}")
            if len(skipped_shape_keys) > 20:
                print(f" - ... {len(skipped_shape_keys) - 20} more")
    
    # Bật Gradient Checkpointing để tiết kiệm VRAM
    # if hasattr(model.image_encoder, "gradient_checkpointing_enable"):
    #     model.image_encoder.gradient_checkpointing_enable()
    
    # Tính lại pos_weight theo tập train hiện tại (sau cân bằng/lọc split)
    base_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    label_cols = base_dataset.disease_cols
    if hasattr(train_dataset, "indices") and hasattr(base_dataset, "df"):
        raw_train_labels_df = base_dataset.df.iloc[train_dataset.indices][label_cols]
    else:
        raw_train_labels_df = base_dataset.df[label_cols]
    train_labels_df = raw_train_labels_df.apply(pd.to_numeric, errors="coerce")
    if uncertain_policy == "ignore":
        train_mask_df = ((train_labels_df.notna()) & (train_labels_df != -1.0)).astype("float32")
        train_labels_df = train_labels_df.where(train_mask_df > 0.5, 0.0).fillna(0.0)
    elif uncertain_policy == "to_nan":
        train_labels_df = train_labels_df.replace(-1.0, np.nan)
        train_mask_df = train_labels_df.notna().astype("float32")
        train_labels_df = train_labels_df.fillna(0.0)
    else:
        train_labels_df = train_labels_df.fillna(0.0).replace(-1.0, 0.0)
        train_mask_df = pd.DataFrame(1.0, index=train_labels_df.index, columns=train_labels_df.columns)
    train_labels_df = (train_labels_df > 0).astype("float32")
    valid_counts = train_mask_df.sum(axis=0).to_numpy(dtype=np.float32)
    pos_counts = (train_labels_df * train_mask_df).sum(axis=0).to_numpy(dtype=np.float32)
    neg_counts = np.maximum(valid_counts - pos_counts, 0.0)
    pos_counts = np.maximum(pos_counts, 1.0)
    dynamic_pos_weight = np.where(valid_counts > 0, neg_counts / pos_counts, 1.0)

    pos_weight_tensor = torch.tensor(dynamic_pos_weight, dtype=torch.float32, device=device)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=8.0)
    print("Pos weight (dynamic, clamped):", [round(float(v), 3) for v in pos_weight_tensor.tolist()])
    criterion_global = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")

    hard_label_indices = [idx for idx, label in enumerate(label_cols) if label in set(hard_global_labels)]
    ignored_hard_labels = sorted(set(hard_global_labels) - set(label_cols))
    if ignored_hard_labels:
        print("Hard global labels ignored because they are not trained labels:", ignored_hard_labels)

    global_label_loss_weight_np = np.ones(len(label_cols), dtype=np.float32)
    if hard_label_indices and hard_global_loss_boost > 1.0:
        global_label_loss_weight_np[hard_label_indices] = float(hard_global_loss_boost)
    global_label_loss_weight_tensor = torch.tensor(
        global_label_loss_weight_np,
        dtype=torch.float32,
        device=device,
    )
    print(
        "Global label loss weights:",
        {
            label: round(float(global_label_loss_weight_np[idx]), 3)
            for idx, label in enumerate(label_cols)
            if global_label_loss_weight_np[idx] != 1.0
        },
    )

    if global_sampler == "hard" and hard_label_indices:
        hard_hits = (train_labels_df.iloc[:, hard_label_indices].to_numpy(dtype=np.float32) > 0.5)
        hard_valid = train_mask_df.iloc[:, hard_label_indices].to_numpy(dtype=np.float32) > 0.5
        hard_hits = hard_hits & hard_valid
        hard_hit_count = hard_hits.sum(axis=1)
        hard_pos_counts = np.maximum(hard_hits.sum(axis=0).astype(np.float64), 1.0)
        hard_label_scale = hard_pos_counts.max() / hard_pos_counts
        hard_label_scale = hard_label_scale / max(float(hard_label_scale.mean()), 1e-6)
        hard_score = hard_hits.astype(np.float64) @ hard_label_scale
        sampler_weights_np = np.ones(len(train_labels_df), dtype=np.float64)
        sampler_weights_np += np.maximum(float(global_sampler_boost) - 1.0, 0.0) * np.minimum(hard_score, 1.0)
        sampler_weights_np += 0.25 * np.maximum(float(global_sampler_boost) - 1.0, 0.0) * np.maximum(hard_score - 1.0, 0.0)
        sampler_weights_np = np.clip(sampler_weights_np, 1.0, max(float(global_sampler_boost) * 2.0, 1.0))
        sampler_generator = torch.Generator().manual_seed(seed + 17)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights_np, dtype=torch.double),
            num_samples=len(sampler_weights_np),
            replacement=True,
            generator=sampler_generator,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_bs, shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True,
            collate_fn=medical_vqa_collate_fn,
        )
        print(
            "Using hard-label global sampler:",
            {
                "labels": [label_cols[idx] for idx in hard_label_indices],
                "label_positive_counts": {
                    label_cols[idx]: int(hard_pos_counts[pos])
                    for pos, idx in enumerate(hard_label_indices)
                },
                "label_sampler_scale": {
                    label_cols[idx]: round(float(hard_label_scale[pos]), 3)
                    for pos, idx in enumerate(hard_label_indices)
                },
                "boost": float(global_sampler_boost),
                "positive_rows": int((hard_hit_count > 0).sum()),
                "total_rows": int(len(train_labels_df)),
                "weight_min": round(float(sampler_weights_np.min()), 3),
                "weight_max": round(float(sampler_weights_np.max()), 3),
                "weight_mean": round(float(sampler_weights_np.mean()), 3),
            },
        )
    elif global_sampler != "none":
        print("Global sampler requested but no matching hard labels were found; using shuffled DataLoader.")

    def global_loss_matrix(logits, targets):
        if global_loss_type == "asl":
            return asymmetric_loss_with_logits(
                logits.float(),
                targets.float(),
                pos_weight=pos_weight_tensor if asl_use_pos_weight else None,
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
            )
        return criterion_global(logits, targets)

    teacher_model = None
    teacher_label_index_tensor = None
    teacher_label_names = []
    if distill_weight > 0.0:
        if not teacher_checkpoint or not os.path.exists(teacher_checkpoint):
            raise FileNotFoundError(f"MEDSAM_TEACHER_CHECKPOINT không tồn tại: {teacher_checkpoint}")
        teacher_ckpt = torch.load(teacher_checkpoint, map_location="cpu")
        teacher_label_names = list(teacher_ckpt.get("disease_cols") or [])
        model_state_dict = teacher_ckpt.get("model_state_dict")
        classifier_state_dict = teacher_ckpt.get("classifier_state_dict")
        if not teacher_label_names or (model_state_dict is None and classifier_state_dict is None):
            raise ValueError(
                "RadDINO teacher checkpoint phải có 'disease_cols' và 'model_state_dict' hoặc 'classifier_state_dict'."
            )
        missing_teacher_labels = [label for label in label_cols if label not in teacher_label_names]
        if missing_teacher_labels:
            raise ValueError(
                "RadDINO teacher thiếu nhãn mà MedSAM3 đang train: "
                f"{missing_teacher_labels}. teacher_labels={teacher_label_names}"
            )

        hf_token = (
            os.environ.get("HF_TOKEN_MEDVQA")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        
        freeze_backbone = teacher_ckpt.get("freeze_backbone", True)
        unfreeze_last_n_blocks = teacher_ckpt.get("unfreeze_last_n_blocks", 0)

        teacher_model = RadDinoAdapter(
            model_id=teacher_ckpt.get("model_id", os.environ.get("RAD_DINO_MODEL_ID", "microsoft/rad-dino")),
            num_classes=len(teacher_label_names),
            hf_token=hf_token,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        ).to(device)
        
        if model_state_dict is not None:
            teacher_model.load_state_dict(model_state_dict)
        else:
            teacher_model.classifier.load_state_dict(classifier_state_dict)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_label_index_tensor = torch.tensor(
            [teacher_label_names.index(label) for label in label_cols],
            dtype=torch.long,
            device=device,
        )
        print(
            "Using RadDINO teacher distillation:",
            {
                "checkpoint": teacher_checkpoint,
                "weight": float(distill_weight),
                "temperature": float(distill_temperature),
                "mask": distill_mask,
                "labels": list(label_cols),
                "teacher_labels": teacher_label_names,
            },
        )

    def distill_loss_from_teacher(student_logits, batch_images, label_mask):
        if teacher_model is None:
            return student_logits.new_tensor(0.0)
        teacher_images = (batch_images.float() * image_std_tensor + image_mean_tensor).clamp(0.0, 1.0)
        with torch.no_grad():
            teacher_outputs = teacher_model(teacher_images)
            teacher_logits = teacher_outputs["global_logits"].float().index_select(1, teacher_label_index_tensor)
            teacher_probs = torch.sigmoid(teacher_logits / float(distill_temperature))

        student_distill_logits = student_logits.float() / float(distill_temperature)
        distill_loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(
            student_distill_logits,
            teacher_probs,
            reduction="none",
        ) * float(distill_temperature ** 2)
        if distill_mask == "all":
            distill_weight_mat = torch.ones_like(label_mask, dtype=distill_loss_mat.dtype)
        else:
            distill_weight_mat = label_mask.to(dtype=distill_loss_mat.dtype)
        return (distill_loss_mat * distill_weight_mat).sum() / distill_weight_mat.sum().clamp_min(1.0)

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
    raw_entity_pos_counts = entity_pos_counts.copy()
    entity_pos_counts = np.maximum(entity_pos_counts, 1.0)
    entity_pos_weight_np = entity_neg_counts / entity_pos_counts
    entity_pos_weight_tensor = torch.tensor(entity_pos_weight_np, dtype=torch.float32, device=device)
    entity_pos_weight_tensor = torch.clamp(entity_pos_weight_tensor, min=1.0, max=entity_pos_weight_max)
    print(
        "Entity pos_weight (dynamic, clamped) stats:",
        {
            "min": round(float(entity_pos_weight_tensor.min().item()), 3),
            "max": round(float(entity_pos_weight_tensor.max().item()), 3),
            "mean": round(float(entity_pos_weight_tensor.mean().item()), 3),
        }
    )
    entity_positive_labels = int((raw_entity_pos_counts > 0).sum())
    print(
        "Entity label support:",
        {
            "labels_with_positive": entity_positive_labels,
            "total_positive_targets": int(raw_entity_pos_counts.sum()),
            "num_entity_labels": int(num_entity_labels),
        },
    )
    if entity_positive_labels == 0:
        raise ValueError("Không có entity label dương nào trong train split. Kiểm tra MEDSAM_JSON_FILE/volume data.")
    criterion_entity = nn.BCEWithLogitsLoss(pos_weight=entity_pos_weight_tensor)
    
    encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    head_params = list(itertools.chain(model.global_proj.parameters(), model.classifier.parameters()))
    if getattr(model, "global_attention_pool", None) is not None:
        head_params.extend(list(model.global_attention_pool.parameters()))
    if getattr(model, "global_pool_logits", None) is not None:
        head_params.append(model.global_pool_logits)
    if getattr(model, "global_head", None) is not None:
        head_params.extend(list(model.global_head.parameters()))
    if getattr(model, "entity_pool_proj", None) is not None:
        head_params.extend(list(model.entity_pool_proj.parameters()))
    if hasattr(model, "entity_classifier"):
        head_params.extend(list(model.entity_classifier.parameters()))
    if getattr(model, "entity_patch_classifier", None) is not None:
        head_params.extend(list(model.entity_patch_classifier.parameters()))

    # Contrastive branch gồm local/text projector + cross-attention cần được tối ưu, nếu không loss contrastive sẽ rất nhiễu.
    contrastive_branch_params = list(model.local_proj.parameters())
    if getattr(model, "text_proj", None) is not None:
        contrastive_branch_params.extend(list(model.text_proj.parameters()))
    if getattr(model, "text_to_patch_attn", None) is not None:
        contrastive_branch_params.extend(list(model.text_to_patch_attn.parameters()))

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': head_params, 'lr': head_lr},
        {'params': contrastive_branch_params, 'lr': contrastive_lr}
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(
        "Sẵn sàng huấn luyện!",
        f"Batch={train_bs}",
        f"Accumulation={accumulation_steps}",
        f"entity_pooling={entity_pooling}",
        f"global_pooling={global_pooling}",
        f"global_head={global_head}",
        f"use_local_entity_head={use_local_entity_head}",
        f"local_entity_merge={local_entity_merge}",
        f"encoder_lr={encoder_lr}",
        f"head_lr={head_lr}",
        f"contrastive_lr={contrastive_lr}",
        f"global_loss={global_loss_type}",
        f"asl=({asl_gamma_pos}, {asl_gamma_neg}, clip={asl_clip}, pos_weight={asl_use_pos_weight})",
        f"rank_loss_weight={global_rank_loss_weight}",
        f"distill_weight={distill_weight}",
        f"distill_teacher={teacher_checkpoint or 'none'}",
    )
    best_val_loss = float('inf') 
    best_val_auc = 0.0
    best_val_map = 0.0
    best_val_entity_auc = 0.0

    def checkpoint_path(kind: str) -> str:
        prefix = output_prefix
        if DEBUG_MODE and not prefix.endswith("_debug"):
            prefix = f"{prefix}_debug"
        return f"{prefix}_{kind}.pth"

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
        total_train_entity_loss = 0.0
        optimizer.zero_grad()
        
        # Bọc DataLoader vào tqdm để hiển thị thanh tiến trình
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=DEBUG_MODE)
        
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = (batch['global_labels'] > 0).to(device=device, dtype=torch.float32)
            label_mask = batch.get('global_label_mask')
            if label_mask is not None:
                label_mask = label_mask.to(device=device, dtype=torch.float32)
            else:
                label_mask = torch.ones_like(labels, device=device)
            finding_multihot_labels = batch.get('entity_multihot_labels', batch.get('entity_labels')).to(device)
            text_prompts = batch['local_prompts']

            # ÉP XUNG BFLOAT16 (Tăng tốc x2, Giảm RAM x2)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    images,
                    local_prompts=text_prompts,
                    extract_local=(alpha_contrastive > 0.0) or use_local_entity_head,
                )
                if outputs["global_logits"].shape != labels.shape:
                    raise RuntimeError(
                        f"Global logits/labels shape mismatch: logits={tuple(outputs['global_logits'].shape)} "
                        f"labels={tuple(labels.shape)}"
                    )
                if outputs["entity_logits"].shape != finding_multihot_labels.shape:
                    raise RuntimeError(
                        f"Entity logits/labels shape mismatch: logits={tuple(outputs['entity_logits'].shape)} "
                        f"labels={tuple(finding_multihot_labels.shape)}"
                    )
                
                # Tính BCE Loss dựa trên số nhãn global hiện tại
                loss_bce = outputs["global_logits"].new_tensor(0.0)
                loss_rank = outputs["global_logits"].new_tensor(0.0)
                loss_distill = outputs["global_logits"].new_tensor(0.0)
                if alpha_global > 0.0:
                    loss_bce_mat = global_loss_matrix(outputs["global_logits"], labels)
                    global_loss_weight = label_mask * global_label_loss_weight_tensor.view(1, -1)
                    loss_bce = (loss_bce_mat * global_loss_weight).sum() / global_loss_weight.sum().clamp_min(1.0)
                    if global_rank_loss_weight > 0.0:
                        loss_rank = pairwise_ranking_loss_with_logits(
                            outputs["global_logits"],
                            labels,
                            label_mask=label_mask,
                            label_weight=global_label_loss_weight_tensor,
                        )
                    else:
                        loss_rank = outputs["global_logits"].new_tensor(0.0)
                    if distill_weight > 0.0:
                        loss_distill = distill_loss_from_teacher(outputs["global_logits"], images, label_mask)
                    else:
                        loss_distill = outputs["global_logits"].new_tensor(0.0)
                    loss_global = loss_bce + global_rank_loss_weight * loss_rank + distill_weight * loss_distill
                else:
                    loss_global = outputs["global_logits"].new_tensor(0.0)

                if alpha_entity > 0.0:
                    loss_entity = criterion_entity(outputs["entity_logits"], finding_multihot_labels)
                else:
                    loss_entity = outputs["global_logits"].new_tensor(0.0)
                
                # Tính Contrastive Loss dựa trên đặc trưng 256 chiều
                if alpha_contrastive > 0.0 and outputs["concept_features"] is not None:
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

                if alpha_contrastive > 0.0:
                    loss_i2t = soft_clip_contrastive_loss(img_features, key_txt, labels_float, key_lbl)
                    loss_t2i = soft_clip_contrastive_loss(text_features, key_img, labels_float, key_lbl)
                    loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                else:
                    loss_contrastive = outputs["global_logits"].new_tensor(0.0)
                
                # Tổng hợp Loss
                loss = alpha_global * loss_global + alpha_entity * loss_entity + alpha_contrastive * loss_contrastive
                loss = loss / accumulation_steps 
                
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps
            total_train_entity_loss += float(loss_entity.detach().item())
            
            # Cập nhật số Loss liên tục lên thanh tiến trình
            pbar.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})
            
            # Log lên WandB mỗi 20 batch
            if i % 20 == 0:
                wandb.log({
                    "batch_train_loss": loss.item() * accumulation_steps,
                    "batch_loss_bce": loss_bce.item(),
                    "batch_loss_rank": loss_rank.item(),
                    "batch_loss_distill": loss_distill.item(),
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
        avg_train_entity_loss = total_train_entity_loss / len(train_loader)
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        model.eval()
        total_val_loss = 0.0
        total_val_distill_loss = 0.0
        all_probs, all_labels, all_masks = [], [], []
        all_finding_probs, all_finding_labels = [], []
        all_finding_logits = []
        
        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", disable=DEBUG_MODE)
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(device)
                labels = (batch['global_labels'] > 0).to(device=device, dtype=torch.float32)
                label_mask = batch.get('global_label_mask')
                if label_mask is not None:
                    label_mask = label_mask.to(device=device, dtype=torch.float32)
                else:
                    label_mask = torch.ones_like(labels, device=device)
                finding_multihot_labels = batch.get('entity_multihot_labels', batch.get('entity_labels')).to(device)
                text_prompts = batch['local_prompts']

                # Bật Autocast lúc test để chạy nhanh hơn
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        images,
                        local_prompts=text_prompts,
                        extract_local=(alpha_contrastive > 0.0) or use_local_entity_head,
                    )
                    if outputs["global_logits"].shape != labels.shape:
                        raise RuntimeError(
                            f"Global logits/labels shape mismatch: logits={tuple(outputs['global_logits'].shape)} "
                            f"labels={tuple(labels.shape)}"
                        )
                    if outputs["entity_logits"].shape != finding_multihot_labels.shape:
                        raise RuntimeError(
                            f"Entity logits/labels shape mismatch: logits={tuple(outputs['entity_logits'].shape)} "
                            f"labels={tuple(finding_multihot_labels.shape)}"
                        )
                    loss_bce = outputs["global_logits"].new_tensor(0.0)
                    loss_rank = outputs["global_logits"].new_tensor(0.0)
                    loss_distill = outputs["global_logits"].new_tensor(0.0)
                    if alpha_global > 0.0:
                        loss_bce_mat = global_loss_matrix(outputs["global_logits"], labels)
                        global_loss_weight = label_mask * global_label_loss_weight_tensor.view(1, -1)
                        loss_bce = (loss_bce_mat * global_loss_weight).sum() / global_loss_weight.sum().clamp_min(1.0)
                        if global_rank_loss_weight > 0.0:
                            loss_rank = pairwise_ranking_loss_with_logits(
                                outputs["global_logits"],
                                labels,
                                label_mask=label_mask,
                                label_weight=global_label_loss_weight_tensor,
                            )
                        else:
                            loss_rank = outputs["global_logits"].new_tensor(0.0)
                        if distill_weight > 0.0:
                            loss_distill = distill_loss_from_teacher(outputs["global_logits"], images, label_mask)
                        else:
                            loss_distill = outputs["global_logits"].new_tensor(0.0)
                        loss_global = loss_bce + global_rank_loss_weight * loss_rank + distill_weight * loss_distill
                    else:
                        loss_global = outputs["global_logits"].new_tensor(0.0)

                    if alpha_entity > 0.0:
                        loss_entity = criterion_entity(outputs["entity_logits"], finding_multihot_labels)
                    else:
                        loss_entity = outputs["global_logits"].new_tensor(0.0)
                    
                    if alpha_contrastive > 0.0 and outputs["concept_features"] is not None:
                        text_global = outputs["concept_features"].mean(dim=1) 
                    else:
                        text_global = outputs["global_features"]

                    img_features = torch.nn.functional.normalize(outputs["global_features"].float(), dim=-1)
                    text_features = torch.nn.functional.normalize(text_global.float(), dim=-1)
                    labels_float = labels.float()

                    if alpha_contrastive > 0.0:
                        loss_i2t = soft_clip_contrastive_loss(img_features, text_features, labels_float)
                        loss_t2i = soft_clip_contrastive_loss(text_features, img_features, labels_float)
                        loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                    else:
                        loss_contrastive = outputs["global_logits"].new_tensor(0.0)
                    loss = alpha_global * loss_global + alpha_entity * loss_entity + alpha_contrastive * loss_contrastive
                    
                total_val_loss += loss.item()
                total_val_distill_loss += float(loss_distill.detach().item())
                
                # Tính xác suất từ Logits
                probs = torch.sigmoid(outputs["global_logits"])
                finding_probs = torch.sigmoid(outputs["entity_logits"])
                
                # Ép kiểu về float32 để Numpy (scikit-learn) không bị hoảng loạn
                all_probs.append(probs.detach().cpu().to(torch.float32).numpy())
                all_labels.append(labels.detach().cpu().to(torch.float32).numpy())
                all_masks.append(label_mask.detach().cpu().to(torch.float32).numpy())
                all_finding_probs.append(finding_probs.detach().cpu().to(torch.float32).numpy())
                all_finding_logits.append(outputs["entity_logits"].detach().cpu().to(torch.float32).numpy())
                all_finding_labels.append(finding_multihot_labels.detach().cpu().to(torch.float32).numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_distill_loss = total_val_distill_loss / len(val_loader)
        all_probs_np = np.concatenate(all_probs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        all_masks_np = np.concatenate(all_masks, axis=0)
        all_finding_probs_np = np.concatenate(all_finding_probs, axis=0)
        all_finding_logits_np = np.concatenate(all_finding_logits, axis=0)
        all_finding_labels_np = np.concatenate(all_finding_labels, axis=0)
        
        val_auc, valid_auc_classes = safe_macro_auc(all_labels_np, all_probs_np, all_masks_np)
        val_map, valid_map_classes = safe_macro_ap(all_labels_np, all_probs_np, all_masks_np)
        val_entity_auc, valid_entity_auc_classes = safe_macro_auc(all_finding_labels_np, all_finding_probs_np)
        entity_finding_true = all_finding_labels_np[:, :num_entity_finding_labels]
        entity_finding_prob = all_finding_probs_np[:, :num_entity_finding_labels]
        entity_anatomy_true = all_finding_labels_np[:, num_entity_finding_labels:]
        entity_anatomy_prob = all_finding_probs_np[:, num_entity_finding_labels:]
        val_entity_finding_auc, valid_entity_finding_auc_classes = safe_macro_auc(
            entity_finding_true,
            entity_finding_prob,
        )
        val_entity_anatomy_auc, valid_entity_anatomy_auc_classes = safe_macro_auc(
            entity_anatomy_true,
            entity_anatomy_prob,
        )
            
        preds_np = (all_probs_np > 0.5).astype(int)
        finding_preds_np = (all_finding_probs_np > 0.5).astype(int)
        positive_rate = float((preds_np * all_masks_np).sum() / max(float(all_masks_np.sum()), 1.0))
        finding_positive_rate = float(finding_preds_np.mean())
        entity_target_positive_rate = float(all_finding_labels_np.mean())
        entity_prob_mean = float(all_finding_probs_np.mean())
        entity_prob_std = float(all_finding_probs_np.std())
        entity_logit_std = float(all_finding_logits_np.std())
        val_f1, valid_f1_classes = safe_macro_f1(all_labels_np, all_probs_np, all_masks_np, threshold=0.5)
        val_entity_f1 = f1_score(all_finding_labels_np, finding_preds_np, average='macro', zero_division=0)
        val_entity_finding_f1 = safe_sklearn_macro_f1(
            entity_finding_true,
            finding_preds_np[:, :num_entity_finding_labels],
        )
        val_entity_anatomy_f1 = safe_sklearn_macro_f1(
            entity_anatomy_true,
            finding_preds_np[:, num_entity_finding_labels:],
        )
        
        print(f"\nKẾT QUẢ EPOCH {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Entity Loss: {avg_train_entity_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val mAP: {val_map:.4f} | Val F1: {val_f1:.4f}")
        print(f"AUC valid classes (global): {valid_auc_classes}/{num_global_labels}")
        print(f"mAP valid classes (global): {valid_map_classes}/{num_global_labels}")
        print(f"F1 valid classes (global): {valid_f1_classes}/{num_global_labels}")
        print(f"Entity Val AUC: {val_entity_auc:.4f} | Entity Val F1: {val_entity_f1:.4f}")
        print(f"AUC valid classes (entity): {valid_entity_auc_classes}/{all_finding_labels_np.shape[1]}")
        print(
            "Entity split metrics:",
            {
                "finding_auc": round(float(val_entity_finding_auc), 4),
                "finding_valid": f"{valid_entity_finding_auc_classes}/{num_entity_finding_labels}",
                "finding_f1": round(float(val_entity_finding_f1), 4),
                "anatomy_auc": round(float(val_entity_anatomy_auc), 4),
                "anatomy_valid": f"{valid_entity_anatomy_auc_classes}/{num_entity_anatomy_labels}",
                "anatomy_f1": round(float(val_entity_anatomy_f1), 4),
            },
        )
        print(
            "Entity stats:",
            {
                "target_positive_rate": round(entity_target_positive_rate, 4),
                "prob_mean": round(entity_prob_mean, 4),
                "prob_std": round(entity_prob_std, 4),
                "logit_std": round(entity_logit_std, 4),
            },
        )
        print(f"Positive prediction rate (@0.5): {positive_rate:.4f}")
        print(f"Finding positive prediction rate (@0.5): {finding_positive_rate:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_entity_loss": avg_train_entity_loss,
            "val_loss": avg_val_loss,
            "val_distill_loss": avg_val_distill_loss,
            "val_auc": val_auc,
            "val_map": val_map,
            "val_f1": val_f1,
            "val_entity_auc": val_entity_auc,
            "val_entity_f1": val_entity_f1,
            "val_entity_finding_auc": val_entity_finding_auc,
            "val_entity_finding_f1": val_entity_finding_f1,
            "val_entity_anatomy_auc": val_entity_anatomy_auc,
            "val_entity_anatomy_f1": val_entity_anatomy_f1,
            "val_positive_rate": positive_rate,
            "val_finding_positive_rate": finding_positive_rate,
            "val_entity_target_positive_rate": entity_target_positive_rate,
            "val_entity_prob_mean": entity_prob_mean,
            "val_entity_prob_std": entity_prob_std,
            "val_entity_logit_std": entity_logit_std,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "disease_cols": list(label_cols),
            "full_canonical_labels": list(getattr(base_dataset, "full_disease_cols", label_cols)),
            "excluded_labels": list(getattr(base_dataset, "excluded_global_labels", [])),
            "entity_finding_labels": list(getattr(base_dataset, "raw_finding_labels", [])),
            "entity_anatomy_labels": list(getattr(base_dataset, "raw_anatomy_labels", [])),
            "derived_label_rules": {
                "No Finding": "1 if all trained pathology/support labels are below threshold, else 0"
            } if exclude_no_finding else {},
            "num_global_labels": int(num_global_labels),
            "num_entity_finding_labels": int(num_entity_finding_labels),
            "num_entity_anatomy_labels": int(num_entity_anatomy_labels),
            "uncertain_policy": uncertain_policy,
            "global_only_mode": global_only_mode,
            "alpha_global": float(alpha_global),
            "alpha_entity": float(alpha_entity),
            "alpha_contrastive": float(alpha_contrastive),
            "global_loss": global_loss_type,
            "asl_gamma_pos": float(asl_gamma_pos),
            "asl_gamma_neg": float(asl_gamma_neg),
            "asl_clip": float(asl_clip),
            "asl_use_pos_weight": bool(asl_use_pos_weight),
            "global_rank_loss_weight": float(global_rank_loss_weight),
            "hard_global_labels": list(hard_global_labels),
            "hard_global_loss_boost": float(hard_global_loss_boost),
            "global_sampler": global_sampler,
            "global_sampler_boost": float(global_sampler_boost),
            "teacher_checkpoint": teacher_checkpoint,
            "distill_weight": float(distill_weight),
            "distill_temperature": float(distill_temperature),
            "distill_mask": distill_mask,
            "teacher_labels": list(teacher_label_names),
            "init_checkpoint": init_checkpoint,
            "encoder_backend": encoder_backend,
            "use_local_entity_head": bool(use_local_entity_head),
            "local_entity_merge": local_entity_merge,
            "entity_pooling": entity_pooling,
            "global_pooling": global_pooling,
            "global_head": global_head,
            "global_head_dropout": float(global_head_dropout),
            "encoder_lr": float(encoder_lr),
            "head_lr": float(head_lr),
            "contrastive_lr": float(contrastive_lr),
            "accumulation_steps": int(accumulation_steps),
            "output_prefix": output_prefix,
            "use_lora": use_lora,
            "val_loss": float(avg_val_loss),
            "val_distill_loss": float(avg_val_distill_loss),
            "val_auc": float(val_auc),
            "val_map": float(val_map),
            "val_f1": float(val_f1),
            "val_entity_auc": float(val_entity_auc),
            "val_entity_f1": float(val_entity_f1),
            "val_entity_finding_auc": float(val_entity_finding_auc),
            "val_entity_finding_f1": float(val_entity_finding_f1),
            "val_entity_anatomy_auc": float(val_entity_anatomy_auc),
            "val_entity_anatomy_f1": float(val_entity_anatomy_f1),
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_path("best")
            torch.save(checkpoint_payload, best_model_path)
            print(f"🌟 Đã lưu Best Model tại Epoch {epoch+1} (Val Loss: {best_val_loss:.4f})")
        if epoch == 0 or val_auc >= best_val_auc:
            best_val_auc = float(val_auc)
            wandb.run.summary["best_val_auc"] = best_val_auc
            torch.save(
                checkpoint_payload,
                checkpoint_path("best_auc"),
            )
            print(f"🌟 Đã lưu Best AUC Model tại Epoch {epoch+1} (Val AUC: {val_auc:.4f})")
        if epoch == 0 or val_map >= best_val_map:
            best_val_map = float(val_map)
            wandb.run.summary["best_val_map"] = best_val_map
            torch.save(
                checkpoint_payload,
                checkpoint_path("best_map"),
            )
            print(f"🌟 Đã lưu Best mAP Model tại Epoch {epoch+1} (Val mAP: {val_map:.4f})")
        if epoch == 0 or val_entity_auc >= best_val_entity_auc:
            best_val_entity_auc = float(val_entity_auc)
            wandb.run.summary["best_val_entity_auc"] = best_val_entity_auc
            torch.save(
                checkpoint_payload,
                checkpoint_path("best_entity_auc"),
            )
            print(f"🌟 Đã lưu Best Entity AUC Model tại Epoch {epoch+1} (Entity Val AUC: {val_entity_auc:.4f})")

        scheduler.step()

    torch.save(
        checkpoint_payload,
        checkpoint_path("last"),
    )
    print("\n✅ Tiến trình huấn luyện hoàn tất!")
    wandb.finish()

@app.local_entrypoint()
def main(
    use_lora: int = -1,
    require_attention_lora: int = -1,
    encoder_backend: str = "",
    debug_mode: int = -1,
    debug_epochs: int = 0,
    global_only: int = -1,
    alpha_global: float = -1.0,
    alpha_entity: float = -1.0,
    alpha_contrastive: float = -1.0,
    uncertain_policy: str = "",
    exclude_no_finding: int = -1,
    init_checkpoint: str = "",
    use_text_guidance: int = -1,
    use_local_entity_head: int = -1,
    local_entity_merge: str = "",
    entity_pooling: str = "",
    global_pooling: str = "",
    global_head: str = "",
    global_head_dropout: float = -1.0,
    encoder_lr: float = -1.0,
    head_lr: float = -1.0,
    contrastive_lr: float = -1.0,
    accumulation_steps: int = 0,
    global_loss: str = "",
    asl_gamma_pos: float = -1.0,
    asl_gamma_neg: float = -1.0,
    asl_clip: float = -1.0,
    asl_use_pos_weight: int = -1,
    global_rank_loss_weight: float = -1.0,
    hard_global_labels: str = "",
    hard_global_loss_boost: float = -1.0,
    global_sampler: str = "",
    global_sampler_boost: float = -1.0,
    teacher_checkpoint: str = "",
    distill_weight: float = -1.0,
    distill_temperature: float = -1.0,
    distill_mask: str = "",
    output_prefix: str = "",
    entity_pos_weight_max: float = -1.0,
    spawn: int = -1,
    seed: int = 0,
    debug_train_size: int = 0,
    debug_val_size: int = 0,
):
    print("Chuẩn bị...")
    use_lora_value = (os.environ.get("USE_LORA", "1") == "1") if use_lora < 0 else bool(use_lora)
    require_attention_lora_value = (
        os.environ.get("REQUIRE_ATTENTION_LORA", "0") == "1"
        if require_attention_lora < 0
        else bool(require_attention_lora)
    )
    encoder_backend_value = (encoder_backend or os.environ.get("ENCODER_BACKEND", "transformers")).strip().lower()
    debug_mode_value = (os.environ.get("DEBUG_MODE", "0") == "1") if debug_mode < 0 else bool(debug_mode)
    debug_epochs_value = int(debug_epochs or os.environ.get("MEDSAM_DEBUG_EPOCHS", "0") or "0")
    global_only_value = (
        os.environ.get("MEDSAM_GLOBAL_ONLY", "0") == "1"
        if global_only < 0
        else bool(global_only)
    )
    alpha_global_value = (
        float(os.environ.get("MEDSAM_ALPHA_GLOBAL", "0.0"))
        if alpha_global < 0.0
        else float(alpha_global)
    )
    alpha_contrastive_value = (
        float(os.environ.get("MEDSAM_ALPHA_CONTRASTIVE", "0.0"))
        if alpha_contrastive < 0.0
        else float(alpha_contrastive)
    )
    alpha_entity_value = (
        float(os.environ.get("MEDSAM_ALPHA_ENTITY", "0.0" if global_only_value else "1.0"))
        if alpha_entity < 0.0
        else float(alpha_entity)
    )
    uncertain_policy_value = (uncertain_policy or os.environ.get("MEDSAM_UNCERTAIN_POLICY", "to_zero")).strip().lower()
    exclude_no_finding_value = (
        os.environ.get("MEDSAM_EXCLUDE_NO_FINDING", "1") == "1"
        if exclude_no_finding < 0
        else bool(exclude_no_finding)
    )
    init_checkpoint_value = (init_checkpoint or os.environ.get("MEDSAM_INIT_CHECKPOINT", "")).strip()
    use_text_guidance_value = (
        os.environ.get("MEDSAM_USE_TEXT_GUIDANCE", "1" if alpha_contrastive_value > 0.0 else "0") == "1"
        if use_text_guidance < 0
        else bool(use_text_guidance)
    )
    default_use_local_entity_head = "1" if (not global_only_value and alpha_entity_value > 0.0) else "0"
    use_local_entity_head_value = (
        os.environ.get("MEDSAM_USE_LOCAL_ENTITY_HEAD", default_use_local_entity_head) == "1"
        if use_local_entity_head < 0
        else bool(use_local_entity_head)
    )
    local_entity_merge_value = (local_entity_merge or os.environ.get("MEDSAM_LOCAL_ENTITY_MERGE", "anatomy")).strip().lower()
    entity_pooling_value = (entity_pooling or os.environ.get("MEDSAM_ENTITY_POOLING", "global")).strip().lower()
    global_pooling_value = (global_pooling or os.environ.get("MEDSAM_GLOBAL_POOLING", "mean")).strip().lower()
    global_head_value = (global_head or os.environ.get("MEDSAM_GLOBAL_HEAD", "linear")).strip().lower()
    global_head_dropout_value = (
        float(os.environ.get("MEDSAM_GLOBAL_HEAD_DROPOUT", "0.1"))
        if global_head_dropout < 0.0
        else float(global_head_dropout)
    )
    encoder_lr_value = (
        float(os.environ.get("MEDSAM_ENCODER_LR", "1e-5"))
        if encoder_lr < 0.0
        else float(encoder_lr)
    )
    head_lr_value = (
        float(os.environ.get("MEDSAM_HEAD_LR", "1e-3"))
        if head_lr < 0.0
        else float(head_lr)
    )
    contrastive_lr_value = (
        float(os.environ.get("MEDSAM_CONTRASTIVE_LR", "5e-4"))
        if contrastive_lr < 0.0
        else float(contrastive_lr)
    )
    accumulation_steps_value = int(accumulation_steps or os.environ.get("MEDSAM_ACCUMULATION_STEPS", "32"))
    global_loss_value = (global_loss or os.environ.get("MEDSAM_GLOBAL_LOSS", "bce")).strip().lower()
    asl_gamma_pos_value = (
        float(os.environ.get("MEDSAM_ASL_GAMMA_POS", "1.0"))
        if asl_gamma_pos < 0.0
        else float(asl_gamma_pos)
    )
    asl_gamma_neg_value = (
        float(os.environ.get("MEDSAM_ASL_GAMMA_NEG", "4.0"))
        if asl_gamma_neg < 0.0
        else float(asl_gamma_neg)
    )
    asl_clip_value = (
        float(os.environ.get("MEDSAM_ASL_CLIP", "0.05"))
        if asl_clip < 0.0
        else float(asl_clip)
    )
    asl_use_pos_weight_value = (
        os.environ.get("MEDSAM_ASL_USE_POS_WEIGHT", "0") == "1"
        if asl_use_pos_weight < 0
        else bool(asl_use_pos_weight)
    )
    global_rank_loss_weight_value = (
        float(os.environ.get("MEDSAM_GLOBAL_RANK_LOSS_WEIGHT", "0.0"))
        if global_rank_loss_weight < 0.0
        else float(global_rank_loss_weight)
    )
    hard_global_labels_value = (
        hard_global_labels or os.environ.get("MEDSAM_HARD_GLOBAL_LABELS", "")
    ).strip()
    hard_global_loss_boost_value = (
        float(os.environ.get("MEDSAM_HARD_GLOBAL_LOSS_BOOST", "1.0"))
        if hard_global_loss_boost < 0.0
        else float(hard_global_loss_boost)
    )
    global_sampler_value = (global_sampler or os.environ.get("MEDSAM_GLOBAL_SAMPLER", "none")).strip().lower()
    global_sampler_boost_value = (
        float(os.environ.get("MEDSAM_GLOBAL_SAMPLER_BOOST", "1.0"))
        if global_sampler_boost < 0.0
        else float(global_sampler_boost)
    )
    teacher_checkpoint_value = (
        teacher_checkpoint or os.environ.get("MEDSAM_TEACHER_CHECKPOINT", "")
    ).strip()
    distill_weight_value = (
        float(os.environ.get("MEDSAM_DISTILL_WEIGHT", "0.0"))
        if distill_weight < 0.0
        else float(distill_weight)
    )
    distill_temperature_value = (
        float(os.environ.get("MEDSAM_DISTILL_TEMPERATURE", "1.0"))
        if distill_temperature < 0.0
        else float(distill_temperature)
    )
    distill_mask_value = (distill_mask or os.environ.get("MEDSAM_DISTILL_MASK", "valid")).strip().lower()
    output_prefix_value = (
        output_prefix or os.environ.get("MEDSAM_OUTPUT_PREFIX", "/data/weights/medvqa_vision")
    ).strip()
    entity_pos_weight_max_value = (
        float(os.environ.get("MEDSAM_ENTITY_POS_WEIGHT_MAX", "10.0"))
        if entity_pos_weight_max < 0.0
        else float(entity_pos_weight_max)
    )
    spawn_value = (
        os.environ.get("MEDSAM_SPAWN", "0") == "1"
        if spawn < 0
        else bool(spawn)
    )
    seed_value = int(seed or os.environ.get("MEDSAM_SEED", "42"))
    debug_train_size_value = int(debug_train_size or os.environ.get("MEDSAM_DEBUG_TRAIN_SIZE", "0") or "0")
    debug_val_size_value = int(debug_val_size or os.environ.get("MEDSAM_DEBUG_VAL_SIZE", "0") or "0")
    print(f"USE_LORA: {use_lora_value}")
    print(f"REQUIRE_ATTENTION_LORA: {require_attention_lora_value}")
    print(f"ENCODER_BACKEND: {encoder_backend_value}")
    print(f"DEBUG_MODE: {debug_mode_value}")
    print(f"MEDSAM_DEBUG_EPOCHS: {debug_epochs_value}")
    print(f"MEDSAM_GLOBAL_ONLY: {global_only_value}")
    print(f"MEDSAM_ALPHA_GLOBAL: {alpha_global_value}")
    print(f"MEDSAM_ALPHA_ENTITY: {alpha_entity_value}")
    print(f"MEDSAM_ALPHA_CONTRASTIVE: {alpha_contrastive_value}")
    print(f"MEDSAM_UNCERTAIN_POLICY: {uncertain_policy_value}")
    print(f"MEDSAM_EXCLUDE_NO_FINDING: {exclude_no_finding_value}")
    print(f"MEDSAM_INIT_CHECKPOINT: {init_checkpoint_value}")
    print(f"MEDSAM_USE_TEXT_GUIDANCE: {use_text_guidance_value}")
    print(f"MEDSAM_USE_LOCAL_ENTITY_HEAD: {use_local_entity_head_value}")
    print(f"MEDSAM_LOCAL_ENTITY_MERGE: {local_entity_merge_value}")
    print(f"MEDSAM_ENTITY_POOLING: {entity_pooling_value}")
    print(f"MEDSAM_GLOBAL_POOLING: {global_pooling_value}")
    print(f"MEDSAM_GLOBAL_HEAD: {global_head_value}")
    print(f"MEDSAM_GLOBAL_HEAD_DROPOUT: {global_head_dropout_value}")
    print(f"MEDSAM_ENCODER_LR: {encoder_lr_value}")
    print(f"MEDSAM_HEAD_LR: {head_lr_value}")
    print(f"MEDSAM_CONTRASTIVE_LR: {contrastive_lr_value}")
    print(f"MEDSAM_ACCUMULATION_STEPS: {accumulation_steps_value}")
    print(f"MEDSAM_GLOBAL_LOSS: {global_loss_value}")
    print(f"MEDSAM_ASL_GAMMA_POS: {asl_gamma_pos_value}")
    print(f"MEDSAM_ASL_GAMMA_NEG: {asl_gamma_neg_value}")
    print(f"MEDSAM_ASL_CLIP: {asl_clip_value}")
    print(f"MEDSAM_ASL_USE_POS_WEIGHT: {asl_use_pos_weight_value}")
    print(f"MEDSAM_GLOBAL_RANK_LOSS_WEIGHT: {global_rank_loss_weight_value}")
    print(f"MEDSAM_HARD_GLOBAL_LABELS: {hard_global_labels_value}")
    print(f"MEDSAM_HARD_GLOBAL_LOSS_BOOST: {hard_global_loss_boost_value}")
    print(f"MEDSAM_GLOBAL_SAMPLER: {global_sampler_value}")
    print(f"MEDSAM_GLOBAL_SAMPLER_BOOST: {global_sampler_boost_value}")
    print(f"MEDSAM_TEACHER_CHECKPOINT: {teacher_checkpoint_value}")
    print(f"MEDSAM_DISTILL_WEIGHT: {distill_weight_value}")
    print(f"MEDSAM_DISTILL_TEMPERATURE: {distill_temperature_value}")
    print(f"MEDSAM_DISTILL_MASK: {distill_mask_value}")
    print(f"MEDSAM_OUTPUT_PREFIX: {output_prefix_value}")
    print(f"MEDSAM_ENTITY_POS_WEIGHT_MAX: {entity_pos_weight_max_value}")
    print(f"MEDSAM_SPAWN: {spawn_value}")
    print(f"MEDSAM_SEED: {seed_value}")
    print(f"MEDSAM_DEBUG_TRAIN_SIZE: {debug_train_size_value}")
    print(f"MEDSAM_DEBUG_VAL_SIZE: {debug_val_size_value}")
    remote_kwargs = {
        "use_lora": use_lora_value,
        "require_attention_lora": require_attention_lora_value,
        "encoder_backend": encoder_backend_value,
        "debug_mode": debug_mode_value,
        "debug_epochs": debug_epochs_value,
        "global_only": int(global_only_value),
        "alpha_global_arg": alpha_global_value,
        "alpha_entity_arg": alpha_entity_value,
        "alpha_contrastive_arg": alpha_contrastive_value,
        "uncertain_policy_arg": uncertain_policy_value,
        "exclude_no_finding": int(exclude_no_finding_value),
        "init_checkpoint_arg": init_checkpoint_value,
        "use_text_guidance": int(use_text_guidance_value),
        "use_local_entity_head": int(use_local_entity_head_value),
        "local_entity_merge_arg": local_entity_merge_value,
        "entity_pooling_arg": entity_pooling_value,
        "global_pooling_arg": global_pooling_value,
        "global_head_arg": global_head_value,
        "global_head_dropout_arg": global_head_dropout_value,
        "encoder_lr_arg": encoder_lr_value,
        "head_lr_arg": head_lr_value,
        "contrastive_lr_arg": contrastive_lr_value,
        "accumulation_steps_arg": accumulation_steps_value,
        "global_loss_arg": global_loss_value,
        "asl_gamma_pos_arg": asl_gamma_pos_value,
        "asl_gamma_neg_arg": asl_gamma_neg_value,
        "asl_clip_arg": asl_clip_value,
        "asl_use_pos_weight_arg": int(asl_use_pos_weight_value),
        "global_rank_loss_weight_arg": global_rank_loss_weight_value,
        "hard_global_labels_arg": hard_global_labels_value,
        "hard_global_loss_boost_arg": hard_global_loss_boost_value,
        "global_sampler_arg": global_sampler_value,
        "global_sampler_boost_arg": global_sampler_boost_value,
        "teacher_checkpoint_arg": teacher_checkpoint_value,
        "distill_weight_arg": distill_weight_value,
        "distill_temperature_arg": distill_temperature_value,
        "distill_mask_arg": distill_mask_value,
        "output_prefix_arg": output_prefix_value,
        "entity_pos_weight_max_arg": entity_pos_weight_max_value,
        "seed_arg": seed_value,
        "debug_train_size": debug_train_size_value,
        "debug_val_size": debug_val_size_value,
    }
    if spawn_value:
        function_call = train_model.spawn(**remote_kwargs)
        print(f"Spawned train_model call: {function_call}")
        print("Local entrypoint exits now; follow with `modal app logs <app-id>` or inspect checkpoints later.")
    else:
        train_model.remote(**remote_kwargs)
