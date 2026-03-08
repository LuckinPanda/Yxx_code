"""
Train Stage-R-pre V7: 原始AdaReNet架构 + SSIM损失

只改进损失函数，架构不变：
1. 使用原始AdaReNet架构
2. 添加SSIM损失（轻量权重）
3. 保持原有自监督mask-based inpainting
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
from src.models.adarenet import AdaReNet  # 原始架构
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.losses.structural_losses import SSIMLoss, color_consistency_loss
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
    n = cfg["noise"]
    if not (_eq(n["sigma_min"], 0.01) and _eq(n["sigma_max"], 0.05)):
        raise ValueError("SPEC-fixed noise range must be sigma in [0.01, 0.05]")


def sample_mask(p_ref: torch.Tensor, mask_prob: float) -> torch.Tensor:
    B, _, H, W = p_ref.shape
    M = torch.bernoulli(torch.full((B, 1, H, W), mask_prob, device=p_ref.device))
    return M


def construct_masked_reflectance(p_ref: torch.Tensor, mask_prob: float, sigma_min: float, sigma_max: float) -> tuple:
    M = sample_mask(p_ref, mask_prob)
    local_mean = F.avg_pool2d(p_ref, kernel_size=5, stride=1, padding=2)
    sigma = random.uniform(sigma_min, sigma_max)
    xi = local_mean + torch.randn_like(p_ref) * sigma
    p_tilde = (1 - M) * p_ref + M * xi
    return p_tilde, M


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """原始梯度损失"""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(progress * math.pi))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    logger = setup_logger("stage_R_pre_v7")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_R_pre_v7.yaml")
    args = parser.parse_args()

    logger.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    validate_constants(cfg)
    
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
    logger.info(f"Dataset samples: {len(dataset)}")

    # 原始AdaReNet架构
    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    adarenet = AdaReNet(base_channels=cfg["model"]["adarenet_channels"])
    
    logger.info(f"AdaReNet parameters: {sum(p.numel() for p in adarenet.parameters()):,}")

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
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device, weights_only=True))

    for p in model.illumination.parameters():
        p.requires_grad = False
    model.illumination.eval()
    logger.info("IlluminationNet frozen, AdaReNet trainable")

    # Pre-cache illumination
    logger.info("Pre-caching illumination outputs...")
    cached_le = []
    cached_pref = []
    with torch.no_grad():
        cache_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=data_cfg["num_workers"], pin_memory=True,
        )
        for batch in tqdm(cache_loader, desc="Caching", ncols=80):
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

    optimizer = torch.optim.Adam(model.adarenet.parameters(), lr=cfg["train"]["lr"])
    
    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)
    if warmup_epochs > 0:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    else:
        scheduler = None

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    save_interval = cfg["train"].get("save_interval", 10)
    periodic_dir = save_dir / "stage_R_pre_v7"
    periodic_dir.mkdir(parents=True, exist_ok=True)

    log_interval = cfg["train"]["log_interval"]
    sigma_min = cfg["noise"]["sigma_min"]
    sigma_max = cfg["noise"]["sigma_max"]

    # Loss weights
    lambda_delta = cfg["train"].get("lambda_delta", 0.15)
    mask_prob = cfg["train"].get("mask_prob", 0.2)
    lambda_ssim = cfg["train"].get("lambda_ssim", 0.2)
    lambda_grad = cfg["train"].get("lambda_grad", 0.1)
    lambda_color = cfg["train"].get("lambda_color", 0.5)

    # SSIM损失
    ssim_loss_fn = SSIMLoss()

    logger.info("=" * 80)
    logger.info("Stage-R-pre V7: 原始AdaReNet + SSIM损失")
    logger.info(f"Epochs: {epochs}, mask_prob: {mask_prob}")
    logger.info(f"Weights: ssim={lambda_ssim}, grad={lambda_grad}, color={lambda_color}, delta={lambda_delta}")
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
        
        for step, batch in enumerate(tqdm(cached_loader, desc=f"V7 Epoch {epoch}", ncols=80)):
            l_e = batch["l_e"].to(device, non_blocking=True)
            p_ref = batch["p_ref"].to(device, non_blocking=True)

            p_tilde, M = construct_masked_reflectance(p_ref, mask_prob, sigma_min, sigma_max)

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

                # 1) Masked inpainting loss
                diff = r_tilde - p_ref
                if use_mask_loss:
                    loss_ss = (M * diff.abs()).sum() / (M.sum() + 1e-6)
                else:
                    loss_ss = diff.abs().mean()

                # 2) SSIM loss (新增)
                loss_ssim = ssim_loss_fn(r_tilde.clamp(0, 1), p_ref.clamp(0, 1))

                # 3) Gradient loss
                loss_grad = gradient_loss(r_tilde, p_ref)

                # 4) Color consistency
                loss_color = color_consistency_loss(r_tilde, p_ref)

                # 5) Delta regularization
                loss_reg = lambda_delta * delta.abs().mean()

                loss = loss_ss + lambda_ssim * loss_ssim + lambda_grad * loss_grad + lambda_color * loss_color + loss_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_ss += loss_ss.item()
            epoch_ssim += loss_ssim.item()
            epoch_grad += loss_grad.item()
            epoch_color += loss_color.item()
            epoch_reg += loss_reg.item()

            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[V7][E{epoch}][S{step+1}] loss={loss.item():.4f} "
                    f"ss={loss_ss.item():.4f} ssim={loss_ssim.item():.4f} "
                    f"grad={loss_grad.item():.4f} lr={lr:.6g}"
                )
                print(msg)
                logger.info(msg)

        if scheduler:
            scheduler.step()
        
        n = max(1, len(cached_loader))
        avg_loss = epoch_loss / n
        msg = (
            f"[V7][E{epoch}] avg_loss={avg_loss:.4f} "
            f"ss={epoch_ss/n:.4f} ssim={epoch_ssim/n:.4f} grad={epoch_grad/n:.4f}"
        )
        print(msg)
        logger.info(msg)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = periodic_dir / "best.pth"
            torch.save(model.adarenet.state_dict(), best_path)
            logger.info(f"Best: {best_path} (loss={best_loss:.4f})")

        if epoch % save_interval == 0:
            periodic_path = periodic_dir / f"epoch_{epoch:04d}.pth"
            torch.save(model.adarenet.state_dict(), periodic_path)
            logger.info(f"Checkpoint: {periodic_path}")

    torch.save(model.adarenet.state_dict(), save_path)
    logger.info(f"Final: {save_path}")
    logger.info("Training completed")


if __name__ == "__main__":
    main()
