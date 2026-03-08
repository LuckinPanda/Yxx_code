"""
Train Stage-R-pre V6: 改进版训练脚本

改进点（保持Retinex框架和自监督算法）：
1. 使用AdaReNetV2Lite（残差连接 + SE注意力）
2. 添加SSIM损失 - 结构保持
3. 添加多尺度梯度损失 - 多级边缘保持
4. 学习率warmup和cosine衰减
5. 更好的mask策略

自监督框架不变：
- mask-based inpainting supervision
- 50% mixed training (masked/full input)
- P_ref作为监督目标
"""
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from src.data.dataset import LowLightDataset
from src.models.adarenet_v2 import AdaReNetV2Lite
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.losses.structural_losses import SSIMLoss, MultiScaleGradientLoss, color_consistency_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def _eq(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) < tol


def setup_logger(stage: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stage}_{timestamp}.log"

    logger = logging.getLogger(stage)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def validate_constants(cfg: dict) -> None:
    c = cfg["constants"]
    if not _eq(c["eps"], 1e-6):
        raise ValueError("SPEC-fixed constant eps must be 1e-6.")
    tau = c["tau"]
    if tau < 0.01:
        print(f"[WARN] tau={tau} is very small. Recommend tau>=0.05.")
    n = cfg["noise"]
    if not (_eq(n["sigma_min"], 0.01) and _eq(n["sigma_max"], 0.05)):
        raise ValueError("SPEC-fixed noise range must be sigma in [0.01, 0.05]")


def sample_mask(p_ref: torch.Tensor, mask_prob: float) -> torch.Tensor:
    B, _, H, W = p_ref.shape
    M = torch.bernoulli(torch.full((B, 1, H, W), mask_prob, device=p_ref.device))
    return M


def sample_mask_structured(p_ref: torch.Tensor, mask_prob: float, block_size: int = 4) -> torch.Tensor:
    """
    结构化mask - 使用block mask而非pixel-wise mask
    这有助于学习更好的空间一致性
    """
    B, _, H, W = p_ref.shape
    # 生成低分辨率mask
    h_blocks = H // block_size
    w_blocks = W // block_size
    M_low = torch.bernoulli(torch.full((B, 1, h_blocks, w_blocks), mask_prob, device=p_ref.device))
    # 上采样到原始分辨率
    M = F.interpolate(M_low, size=(H, W), mode='nearest')
    return M


def construct_masked_reflectance(p_ref: torch.Tensor, mask_prob: float, sigma_min: float, sigma_max: float, use_structured: bool = True) -> tuple:
    """
    构建masked reflectance用于自监督训练
    """
    if use_structured:
        M = sample_mask_structured(p_ref, mask_prob, block_size=4)
    else:
        M = sample_mask(p_ref, mask_prob)
    
    # Local mean provides reasonable starting point
    local_mean = F.avg_pool2d(p_ref, kernel_size=5, stride=1, padding=2)
    sigma = random.uniform(sigma_min, sigma_max)
    xi = local_mean + torch.randn_like(p_ref) * sigma
    p_tilde = (1 - M) * p_ref + M * xi
    return p_tilde, M


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    """
    带warmup的余弦学习率调度
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性warmup
            return epoch / warmup_epochs
        else:
            # 余弦衰减
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(progress * math.pi))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    logger = setup_logger("stage_R_pre_v6")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_R_pre_v6.yaml")
    args = parser.parse_args()

    logger.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    validate_constants(cfg)
    logger.info("Config validated.")

    set_seed(cfg["seed"])
    logger.info(f"Random seed set to: {cfg['seed']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_cfg = cfg["data"]
    resize = data_cfg["resize"]
    if resize is not None:
        resize = tuple(resize)

    dataset = LowLightDataset(
        mode=data_cfg["mode"],
        source_low_dir=data_cfg["source_low_dir"],
        source_high_dir=data_cfg["source_high_dir"],
        target_low_dir=data_cfg["target_low_dir"],
        resize=resize,
    )
    logger.info(f"Dataset mode: {data_cfg['mode']}, samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # 使用改进的AdaReNetV2Lite
    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    adarenet = AdaReNetV2Lite(base_channels=cfg["model"]["adarenet_channels"])
    
    logger.info(f"AdaReNetV2Lite parameters: {sum(p.numel() for p in adarenet.parameters()):,}")

    illum_adjust_mode = cfg["constants"].get("illum_adjust_mode", "gamma")
    pref_max = cfg["constants"].get("pref_max", 5.0)
    model = RetinexAdaReNet(
        illum,
        adarenet,
        omega=cfg["constants"]["omega"],
        tau=cfg["constants"]["tau"],
        eps=cfg["constants"]["eps"],
        illum_adjust_mode=illum_adjust_mode,
        pref_max=pref_max,
    ).to(device)

    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    logger.info(f"Loading illumination checkpoint: {illum_ckpt}")
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))

    for p in model.illumination.parameters():
        p.requires_grad = False
    model.illumination.eval()
    logger.info("IlluminationNet frozen, AdaReNetV2Lite trainable")

    # Pre-cache illumination outputs
    logger.info("Pre-caching illumination outputs...")
    cached_le = []
    cached_pref = []
    with torch.no_grad():
        cache_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=data_cfg["num_workers"], pin_memory=True,
        )
        for batch in tqdm(cache_loader, desc="Caching illumination", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            l_t, l_e = model.compute_illumination(low)
            p_ref = model.compute_pref(low, l_t)
            cached_le.append(l_e.squeeze(0).cpu())
            cached_pref.append(p_ref.squeeze(0).cpu())

    class CachedReflectanceDataset(torch.utils.data.Dataset):
        def __init__(self, le_list, pref_list):
            self.le = le_list
            self.pref = pref_list
        def __len__(self):
            return len(self.le)
        def __getitem__(self, idx):
            return {"l_e": self.le[idx], "p_ref": self.pref[idx]}

    cached_dataset = CachedReflectanceDataset(cached_le, cached_pref)
    cached_loader = DataLoader(
        cached_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.adarenet.parameters(), 
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4)
    )
    
    # Learning rate scheduler
    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    logger.info(f"Using cosine LR schedule with {warmup_epochs} warmup epochs")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    save_interval = cfg["train"].get("save_interval", 10)
    periodic_dir = save_dir / "stage_R_pre_v6"
    periodic_dir.mkdir(parents=True, exist_ok=True)

    log_interval = cfg["train"]["log_interval"]
    sigma_min    = cfg["noise"]["sigma_min"]
    sigma_max    = cfg["noise"]["sigma_max"]

    # Loss weights
    lambda_delta = cfg["train"].get("lambda_delta", 0.1)
    mask_prob    = cfg["train"].get("mask_prob", 0.15)
    lambda_ssim  = cfg["train"].get("lambda_ssim", 0.3)
    lambda_grad  = cfg["train"].get("lambda_grad", 0.1)
    lambda_color = cfg["train"].get("lambda_color", 0.3)
    use_structured_mask = cfg["train"].get("use_structured_mask", True)

    # 初始化损失函数
    ssim_loss_fn = SSIMLoss()
    grad_loss_fn = MultiScaleGradientLoss(scales=3)

    logger.info("=" * 80)
    logger.info("Starting Stage-R-pre V6 (AdaReNetV2Lite + SSIM + MultiScale Gradient)")
    logger.info(f"Epochs: {epochs}, mask_prob: {mask_prob}, use_structured_mask: {use_structured_mask}")
    logger.info(f"Loss weights: lambda_ssim={lambda_ssim}, lambda_grad={lambda_grad}, lambda_color={lambda_color}, lambda_delta={lambda_delta}")
    logger.info("=" * 80)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_ss = 0.0
        epoch_ssim = 0.0
        epoch_grad = 0.0
        epoch_color = 0.0
        epoch_reg = 0.0
        
        for step, batch in enumerate(tqdm(cached_loader, desc=f"V6 Epoch {epoch}", ncols=80)):
            l_e = batch["l_e"].to(device, non_blocking=True)
            p_ref = batch["p_ref"].to(device, non_blocking=True)

            # Mask-based inpainting supervision
            p_tilde, M = construct_masked_reflectance(
                p_ref, mask_prob, sigma_min, sigma_max, 
                use_structured=use_structured_mask
            )

            # 50% mixed training
            if random.random() < 0.5:
                input_ref = p_tilde
                use_mask_loss = True
            else:
                input_ref = p_ref
                use_mask_loss = False

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                delta = model.adarenet(torch.cat([input_ref, l_e], dim=1))
                r_tilde = input_ref - delta

                # ── 1) Masked inpainting loss (primary) ──
                diff = r_tilde - p_ref
                if use_mask_loss:
                    loss_ss = (M * diff.abs()).sum() / (M.sum() + 1e-6)
                else:
                    loss_ss = diff.abs().mean()

                # ── 2) SSIM loss (structural) ──
                loss_ssim = ssim_loss_fn(r_tilde.clamp(0, 1), p_ref.clamp(0, 1))

                # ── 3) Multi-scale gradient loss ──
                loss_grad = grad_loss_fn(r_tilde, p_ref)

                # ── 4) Color consistency loss ──
                loss_color = color_consistency_loss(r_tilde, p_ref)

                # ── 5) Delta regularization (小一点) ──
                loss_reg = lambda_delta * delta.abs().mean()

                # Total loss
                loss = loss_ss + lambda_ssim * loss_ssim + lambda_grad * loss_grad + lambda_color * loss_color + loss_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability (更强的clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.adarenet.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            epoch_ss    += loss_ss.item()
            epoch_ssim  += loss_ssim.item()
            epoch_grad  += loss_grad.item()
            epoch_color += loss_color.item()
            epoch_reg   += loss_reg.item()

            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[V6][E{epoch}][S{step+1}] loss={loss.item():.4f} "
                    f"ss={loss_ss.item():.4f} ssim={loss_ssim.item():.4f} "
                    f"grad={loss_grad.item():.4f} color={loss_color.item():.4f} "
                    f"reg={loss_reg.item():.4f} lr={lr:.6g}"
                )
                print(msg)
                logger.info(msg)

        scheduler.step()
        
        n = max(1, len(cached_loader))
        avg_loss = epoch_loss / n
        msg = (
            f"[V6][E{epoch}] avg_loss={avg_loss:.4f} "
            f"ss={epoch_ss/n:.4f} ssim={epoch_ssim/n:.4f} "
            f"grad={epoch_grad/n:.4f} color={epoch_color/n:.4f} "
            f"reg={epoch_reg/n:.4f} lr={scheduler.get_last_lr()[0]:.6g}"
        )
        print(msg)
        logger.info(msg)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "stage_R_pre_v6" / "best.pth"
            torch.save(model.adarenet.state_dict(), best_path)
            logger.info(f"Best model saved: {best_path} (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % save_interval == 0:
            periodic_path = periodic_dir / f"epoch_{epoch:04d}.pth"
            torch.save(model.adarenet.state_dict(), periodic_path)
            logger.info(f"Checkpoint saved: {periodic_path}")

    torch.save(model.adarenet.state_dict(), save_path)
    logger.info(f"Final checkpoint saved: {save_path}")
    logger.info("=" * 80)
    logger.info("Stage-R-pre V6 training completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
