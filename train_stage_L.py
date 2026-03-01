import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LowLightDataset
from src.models.adarenet import AdaReNet
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.utils.config import load_config
from src.utils.seed import set_seed


def _eq(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) < tol


def setup_logger(stage: str) -> logging.Logger:
    """设置日志记录器，同时输出到文件和控制台"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stage}_{timestamp}.log"
    
    logger = logging.getLogger(stage)
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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


# ── Helper losses ──────────────────────────────────────────────────────────

def compute_retinex_pseudo_gt(
    low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Derive illumination pseudo-GT from paired low / high images.

    Under Retinex: I_low = R · L,  I_high ≈ R  (well-lit)
    Therefore L ≈ I_low / I_high  (per-pixel, per-channel).
    We take the *mean across RGB* to get a single-channel map and then
    smooth it with avg_pool to regularise high-frequency noise.

    Returns: [B, 1, H, W] in [0, 1].
    """
    # Per-channel ratio → mean across channels → single-channel map
    ratio = low / (high + eps)                           # [B, 3, H, W]
    l_pseudo = ratio.mean(dim=1, keepdim=True)           # [B, 1, H, W]
    # Mild spatial smoothing to enforce low-frequency structure
    l_pseudo = F.avg_pool2d(l_pseudo, kernel_size=5, stride=1, padding=2)
    return l_pseudo.clamp(0.0, 1.0)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total Variation loss - encourages spatial smoothness of L_T."""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def recon_loss(
    l_t: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Reconstruction consistency: L_T · I_high ≈ I_low.

    If L_T correctly captures the illumination ratio then
    L_T × R (≈ I_high) should reconstruct I_low.
    L_T: [B,1,H,W] broadcasts over the 3 RGB channels of high/low.
    """
    return F.l1_loss(l_t * high, low)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logger("stage_L")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_L.yaml")
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
    logger.info(f"DataLoader: batch_size={cfg['train']['batch_size']}, num_workers={data_cfg['num_workers']}")

    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    adarenet = AdaReNet(base_channels=cfg["model"]["adarenet_channels"])

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

    # Freeze AdaReNet in Stage-L (illumination pretraining)
    logger.info("AdaReNet frozen, IlluminationNet trainable")

    optimizer = torch.optim.Adam(model.illumination.parameters(), lr=cfg["train"]["lr"])
    logger.info(f"Optimizer: Adam, lr={cfg['train']['lr']}")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]

    # Loss weights (from config, with defaults)
    train_cfg = cfg["train"]
    lambda_illum = train_cfg.get("lambda_illum", 1.0)
    lambda_tv    = train_cfg.get("lambda_tv", 0.1)
    lambda_recon = train_cfg.get("lambda_recon", 1.0)
    logger.info(
        f"Training: {epochs} epochs, log_interval={log_interval}, "
        f"lambda_illum={lambda_illum}, lambda_tv={lambda_tv}, lambda_recon={lambda_recon}, "
        f"illum_adjust_mode={illum_adjust_mode}, omega={cfg['constants']['omega']}"
    )

    logger.info("=" * 80)
    logger.info("Starting Stage-L training (Illumination Pretraining)")
    logger.info("Pseudo-GT: Retinex-derived from paired data  L_pseudo = mean(I_low / I_high)")
    logger.info("Losses: L1(L_T, L_pseudo) + TV(L_T) + Recon(L_T * I_high, I_low)")
    logger.info("=" * 80)

    # AMP (mixed precision) for speedup on modern GPUs
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("AMP (mixed precision) enabled")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_illum = 0.0
        epoch_tv = 0.0
        epoch_recon = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Stage-L Epoch {epoch}", ncols=80)):
            low  = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # ── Retinex-derived pseudo illumination GT ──
                # L_pseudo ≈ mean_channel(I_low / I_high), smoothed → [B,1,H,W]
                L_pseudo = compute_retinex_pseudo_gt(low, high)
                
                # Get illumination prediction
                L_T, _ = model.compute_illumination(low)   # [B, 1, H, W]
                
                # ── Losses ──
                # 1) Direct illumination supervision (single-channel)
                loss_illum = F.l1_loss(L_T, L_pseudo)
                
                # 2) TV smoothness on L_T
                loss_tv = tv_loss(L_T)
                
                # 3) Reconstruction consistency: L_T * I_high ≈ I_low
                loss_rec = recon_loss(L_T, low, high)
                
                loss = lambda_illum * loss_illum + lambda_tv * loss_tv + lambda_recon * loss_rec

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss  += loss.item()
            epoch_illum += loss_illum.item()
            epoch_tv    += loss_tv.item()
            epoch_recon += loss_rec.item()
            
            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Stage-L][E{epoch}][S{step+1}] loss={loss.item():.6f} "
                    f"illum={loss_illum.item():.6f} tv={loss_tv.item():.6f} "
                    f"recon={loss_rec.item():.6f} lr={lr:.6g}"
                )
                print(msg)
                logger.info(msg)

        n = max(1, len(loader))
        msg = (
            f"[Stage-L][E{epoch}] avg loss={epoch_loss/n:.6f} "
            f"illum={epoch_illum/n:.6f} tv={epoch_tv/n:.6f} "
            f"recon={epoch_recon/n:.6f}"
        )
        print(msg)
        logger.info(msg)

    torch.save(model.illumination.state_dict(), save_path)
    logger.info(f"Checkpoint saved: {save_path}")
    logger.info("=" * 80)
    logger.info("Stage-L training completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
