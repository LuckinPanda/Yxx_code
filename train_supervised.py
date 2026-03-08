"""
有监督训练脚本：使用high目录的图像作为ground truth进行训练
目标：达到 PSNR 19+ 和 SSIM 0.8+
"""
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.dataset import LowLightDataset
from src.models.adarenet import AdaReNet, AdaReNetLegacy
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.utils.config import load_config
from src.utils.seed import set_seed


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


# ══════════════════════════════════════════════════════════════════════════
# SSIM Loss
# ══════════════════════════════════════════════════════════════════════════

def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d


class SSIMLoss(nn.Module):
    """SSIM loss for perceptual quality optimization"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device, dtype = pred.device, pred.dtype
        C = pred.shape[1]
        
        window = _gaussian_window(self.window_size, self.sigma, device, dtype)
        window = window.expand(C, 1, self.window_size, self.window_size)
        
        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2, groups=C)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2, groups=C)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=C) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)) / \
               ((mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2))
        
        return 1.0 - ssim.mean()


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]  # up to relu3_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)
        return F.l1_loss(pred_feat, target_feat)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient-domain L1 loss for edge preservation"""
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def color_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Color consistency loss"""
    pred_mean = pred.mean(dim=(2, 3), keepdim=True)
    target_mean = target.mean(dim=(2, 3), keepdim=True)
    return F.l1_loss(pred_mean, target_mean)


# ══════════════════════════════════════════════════════════════════════════
# 评估函数
# ══════════════════════════════════════════════════════════════════════════

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between two images"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.item()


# ══════════════════════════════════════════════════════════════════════════
# 主训练函数
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/supervised.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    logger = setup_logger("supervised")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集 - 使用成对的low/high图像
    data_cfg = cfg["data"]
    train_dataset = LowLightDataset(
        data_cfg["train_low_dir"],
        data_cfg["train_high_dir"],
        resize=data_cfg.get("resize"),
    )
    
    val_dataset = LowLightDataset(
        data_cfg["val_low_dir"],
        data_cfg["val_high_dir"],
        resize=data_cfg.get("resize"),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Training set: {len(train_dataset)} images")
    logger.info(f"Validation set: {len(val_dataset)} images")

    # 模型
    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    use_legacy = cfg["model"].get("use_legacy_adarenet", True)
    if use_legacy:
        adarenet = AdaReNetLegacy(base_channels=cfg["model"]["adarenet_channels"])
        logger.info("Using AdaReNetLegacy")
    else:
        adarenet = AdaReNet(base_channels=cfg["model"]["adarenet_channels"])
        logger.info("Using AdaReNet")

    model = RetinexAdaReNet(
        illum,
        adarenet,
        omega=cfg["constants"]["omega"],
        tau=cfg["constants"]["tau"],
        eps=cfg["constants"]["eps"],
        illum_adjust_mode=cfg["constants"].get("illum_adjust_mode", "gamma"),
        pref_max=cfg["constants"].get("pref_max", 5.0),
    ).to(device)

    # 加载预训练的illumination网络
    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    logger.info(f"Loading illumination checkpoint: {illum_ckpt}")
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))

    # 可选：加载预训练的adarenet作为初始化
    if cfg["ckpt"].get("denoise_pre_ckpt_path"):
        denoise_ckpt = cfg["ckpt"]["denoise_pre_ckpt_path"]
        logger.info(f"Loading pretrained denoiser: {denoise_ckpt}")
        model.adarenet.load_state_dict(torch.load(denoise_ckpt, map_location=device))

    # 冻结illumination网络
    for p in model.illumination.parameters():
        p.requires_grad = False
    logger.info("IlluminationNet frozen, AdaReNet trainable")

    # Loss functions
    ssim_loss_fn = SSIMLoss().to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    # Optimizer with different learning rates
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.adarenet.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg["lr"] * 0.01,
    )

    # 混合精度
    scaler = torch.amp.GradScaler(enabled=True)

    # Loss weights
    lambda_l1 = train_cfg.get("lambda_l1", 1.0)
    lambda_ssim = train_cfg.get("lambda_ssim", 0.5)
    lambda_perceptual = train_cfg.get("lambda_perceptual", 0.1)
    lambda_grad = train_cfg.get("lambda_grad", 0.1)
    lambda_color = train_cfg.get("lambda_color", 0.1)

    logger.info(f"Loss weights: L1={lambda_l1}, SSIM={lambda_ssim}, Perceptual={lambda_perceptual}, Grad={lambda_grad}, Color={lambda_color}")

    best_psnr = 0.0
    best_ssim = 0.0
    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for batch_idx, (low, high, _) in enumerate(pbar):
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                outputs = model(low)
                pred = outputs["R_hat"]

                # Compute losses
                l1_loss = F.l1_loss(pred, high)
                ssim_l = ssim_loss_fn(pred, high)
                perceptual_l = perceptual_loss_fn(pred, high)
                grad_l = gradient_loss(pred, high)
                color_l = color_consistency_loss(pred, high)

                total_loss = (
                    lambda_l1 * l1_loss +
                    lambda_ssim * ssim_l +
                    lambda_perceptual * perceptual_l +
                    lambda_grad * grad_l +
                    lambda_color * color_l
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.adarenet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", l1=f"{l1_loss.item():.4f}", ssim=f"{ssim_l.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_psnrs, val_ssims = [], []
        with torch.no_grad():
            for low, high, _ in val_loader:
                low = low.to(device)
                high = high.to(device)
                outputs = model(low)
                pred = outputs["R_hat"].clamp(0, 1)
                
                psnr = compute_psnr(pred, high)
                ssim = compute_ssim(pred, high)
                val_psnrs.append(psnr)
                val_ssims.append(ssim)

        avg_psnr = np.mean(val_psnrs)
        avg_ssim = np.mean(val_ssims)
        avg_loss = np.mean(epoch_losses)

        logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_psnr > best_psnr or (avg_psnr > best_psnr - 0.5 and avg_ssim > best_ssim):
            best_psnr = max(best_psnr, avg_psnr)
            best_ssim = max(best_ssim, avg_ssim)
            save_path = save_dir / train_cfg["save_name"]
            torch.save(model.adarenet.state_dict(), save_path)
            logger.info(f"Saved best model: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        # Periodic checkpoint
        if epoch % train_cfg.get("save_interval", 10) == 0:
            periodic_path = save_dir / "supervised" / f"epoch_{epoch:04d}.pth"
            periodic_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.adarenet.state_dict(), periodic_path)

    logger.info(f"Training completed. Best PSNR: {best_psnr:.2f}, Best SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()
