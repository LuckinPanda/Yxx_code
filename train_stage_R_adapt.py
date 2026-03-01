import argparse
import logging
import random
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

# Deprecated: Stage-R-adapt is temporarily abandoned.
# This script is retained for reference and should not be used in current runs.


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
    # Note: omega can vary for better illumination enhancement
    # tau is now configurable (recommended 0.05-0.2 for numerical stability)
    c = cfg["constants"]
    if not _eq(c["eps"], 1e-6):
        raise ValueError("SPEC-fixed constant eps must be 1e-6.")
    tau = c["tau"]
    if tau < 0.01:
        print(f"[WARN] tau={tau} is very small. Dark-area P_ref may explode. Recommend tau>=0.05.")
    n = cfg["noise"]
    if not (_eq(n["sigma_min"], 0.01) and _eq(n["sigma_max"], 0.05)):
        raise ValueError("SPEC-fixed noise range must be sigma in [0.01, 0.05]")


def sample_noise(p_ref: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    sigma = random.uniform(sigma_min, sigma_max)
    noise = torch.randn_like(p_ref) * sigma
    return noise


def sample_mask(p_ref: torch.Tensor, mask_prob: float) -> torch.Tensor:
    """Sample binary mask M ~ Bernoulli(mask_prob) for each spatial location.
    
    Args:
        p_ref: [B, C, H, W]
        mask_prob: masking probability (e.g., 0.1-0.3)
    
    Returns:
        M: [B, 1, H, W] binary mask
    """
    B, _, H, W = p_ref.shape
    M = torch.bernoulli(torch.full((B, 1, H, W), mask_prob, device=p_ref.device))
    return M


def construct_masked_reflectance(p_ref: torch.Tensor, mask_prob: float, sigma_min: float, sigma_max: float) -> tuple:
    """Construct masked reflectance for mask-based inpainting supervision.
    
    P_tilde = (1 - M) * P_ref + M * xi
    
    Args:
        p_ref: [B, 3, H, W] reflectance proxy
        mask_prob: masking probability
        sigma_min, sigma_max: noise scale range
    
    Returns:
        p_tilde: [B, 3, H, W] masked reflectance
        M: [B, 1, H, W] binary mask
    """
    M = sample_mask(p_ref, mask_prob)
    xi = sample_noise(p_ref, sigma_min, sigma_max)
    p_tilde = (1 - M) * p_ref + M * xi
    return p_tilde, M


def main() -> None:
    logger = setup_logger("stage_R_adapt")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_R_adapt.yaml")
    args = parser.parse_args()

    logger.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    validate_constants(cfg)
    c = cfg["constants"]
    logger.info(f"Config validated. tau={c['tau']}, eps={c['eps']}, omega={c['omega']}")

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
    logger.info(f"Model created on {device}")

    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    pre_ckpt = cfg["ckpt"]["denoise_pre_ckpt_path"]
    logger.info(f"Loading illumination checkpoint: {illum_ckpt}")
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))
    logger.info(f"Loading pretrained AdaReNet checkpoint: {pre_ckpt}")
    model.adarenet.load_state_dict(torch.load(pre_ckpt, map_location=device))
    logger.info("Checkpoints loaded")

    # Frozen copy for anchor regularization
    anchor_net = AdaReNet(base_channels=cfg["model"]["adarenet_channels"]).to(device)
    anchor_net.load_state_dict(torch.load(pre_ckpt, map_location=device))
    anchor_net.eval()
    for p in anchor_net.parameters():
        p.requires_grad = False
    logger.info("Anchor network created (frozen copy of pretrained AdaReNet)")

    # Freeze illumination in Stage-R-adapt
    for p in model.illumination.parameters():
        p.requires_grad = False
    model.illumination.eval()
    logger.info("IlluminationNet frozen, AdaReNet trainable")

    optimizer = torch.optim.Adam(model.adarenet.parameters(), lr=cfg["train"]["lr"])
    logger.info(f"Optimizer: Adam, lr={cfg['train']['lr']}")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    sigma_min = cfg["noise"]["sigma_min"]
    sigma_max = cfg["noise"]["sigma_max"]
    lambda_anc = cfg["train"]["lambda_anc"]
    lambda_delta = cfg["train"].get("lambda_delta", 0.0)
    mask_prob = cfg["train"].get("mask_prob", 0.2)  # Masking probability (0.1-0.3 recommended)
    logger.info(
        f"Training: {epochs} epochs, log_interval={log_interval}, noise sigma in [{sigma_min}, {sigma_max}], "
        f"mask_prob={mask_prob}, lambda_anc={lambda_anc}, lambda_delta={lambda_delta}"
    )

    model.adarenet.train()
    model.illumination.eval()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_loss_ss = 0.0
        epoch_loss_anc = 0.0
        epoch_loss_reg = 0.0
        
        for step, batch in enumerate(tqdm(loader, desc=f"Stage-R-adapt Epoch {epoch}", ncols=80)):
            low = batch["low"].to(device, non_blocking=True)

            with torch.no_grad():
                l_t, l_e = model.compute_illumination(low)
                p_ref = model.compute_pref(low, l_t)

            # Mask-based inpainting supervision
            p_tilde, M = construct_masked_reflectance(p_ref, mask_prob, sigma_min, sigma_max)
            
            # 🔥 FIX 1: 50% mixed training (full vs masked input)
            # Prevents network from becoming pure inpainting model
            if random.random() < 0.5:
                input_ref = p_tilde  # masked input
            else:
                input_ref = p_ref    # full input (no mask)
            
            # Network predicts residual
            delta = model.adarenet(torch.cat([input_ref, l_e], dim=1))
            
            # Recover reflectance
            r_tilde = input_ref - delta
            
            # 🔥 FIX 2: Normalized masked loss
            # Compute only on masked pixels with proper normalization
            diff = r_tilde - p_ref
            loss_ss = (M * diff.abs()).sum() / (M.sum() + 1e-6)

            # 🔥 FIX 3: Anchor with matched input distribution
            # Teacher uses same input as student (p_tilde, not p_ref)
            with torch.no_grad():
                d_anchor = anchor_net(torch.cat([p_tilde, l_e], dim=1))
            
            # 🔥 FIX 4: Anchor loss only on non-masked regions
            # Prevents conflict between mask-filling and anchoring
            loss_anchor = ((1 - M) * (delta - d_anchor).abs()).sum() / ((1 - M).sum() + 1e-6)

            # 🔥 FIX 5: Disable delta regularization (was too strong)
            loss_reg = lambda_delta * delta.abs().mean()
            loss = loss_ss + lambda_anc * loss_anchor + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_ss += loss_ss.item()
            epoch_loss_anc += loss_anchor.item()
            epoch_loss_reg += loss_reg.item()
            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Stage-R-adapt][E{epoch}][S{step+1}] loss={loss.item():.6f} "
                    f"ss={loss_ss.item():.6f} anc={loss_anchor.item():.6f} reg={loss_reg.item():.6f} lr={lr:.6g}"
                )
                print(msg)
                logger.info(msg)

        avg_loss = epoch_loss / max(1, len(loader))
        avg_loss_ss = epoch_loss_ss / max(1, len(loader))
        avg_loss_anc = epoch_loss_anc / max(1, len(loader))
        avg_loss_reg = epoch_loss_reg / max(1, len(loader))
        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"[Stage-R-adapt][E{epoch}] avg_loss={avg_loss:.6f} ss={avg_loss_ss:.6f} "
            f"anc={avg_loss_anc:.6f} reg={avg_loss_reg:.6f} lr={lr:.6g}"
        )
        print(msg)
        logger.info(msg)

    torch.save(model.adarenet.state_dict(), save_path)
    msg = f"[Stage-R-adapt] saved ckpt: {save_path}"
    print(msg)
    logger.info(msg)


if __name__ == "__main__":
    main()
