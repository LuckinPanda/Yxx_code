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
        print(f"[WARN] tau={tau} is very small. Dark-area P_ref may explode. Recommend tau>=0.05.")
    n = cfg["noise"]
    if not (_eq(n["sigma_min"], 0.01) and _eq(n["sigma_max"], 0.05)):
        raise ValueError("SPEC-fixed noise range must be sigma in [0.01, 0.05]")


# ── Mask / noise helpers ──────────────────────────────────────────────────

def sample_noise(p_ref: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    sigma = random.uniform(sigma_min, sigma_max)
    noise = torch.randn_like(p_ref) * sigma
    return noise


def sample_mask(p_ref: torch.Tensor, mask_prob: float) -> torch.Tensor:
    B, _, H, W = p_ref.shape
    M = torch.bernoulli(torch.full((B, 1, H, W), mask_prob, device=p_ref.device))
    return M


def construct_masked_reflectance(p_ref: torch.Tensor, mask_prob: float, sigma_min: float, sigma_max: float) -> tuple:
    """Construct masked reflectance for mask-based inpainting supervision.

    P_tilde = (1 - M) * P_ref + M * xi

    Instead of pure random noise, xi uses local-mean + noise so that the
    masked regions start closer to the true reflectance.  This helps preserve
    local color statistics and reduces color shift in recovered reflectance.
    """
    M = sample_mask(p_ref, mask_prob)
    # Local mean provides a reasonable starting point for inpainting
    local_mean = F.avg_pool2d(p_ref, kernel_size=5, stride=1, padding=2)
    sigma = random.uniform(sigma_min, sigma_max)
    xi = local_mean + torch.randn_like(p_ref) * sigma
    p_tilde = (1 - M) * p_ref + M * xi
    return p_tilde, M


# ── Additional loss functions ─────────────────────────────────────────────

def tv_loss_on_reflectance(r: torch.Tensor) -> torch.Tensor:
    """Total Variation loss on recovered reflectance R_e.

    Encourages spatial smoothness of the output, which acts as an
    implicit denoising prior.  Unlike gradient_loss(pred, noisy_target),
    this does NOT force the output to replicate noisy gradients.
    """
    diff_h = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().mean()
    diff_w = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def channel_balance_loss(pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Channel balance prior: encourage R/G/B channel means to be similar.

    Under the assumption that average reflectance across a scene is
    roughly achromatic (gray-world prior), this prevents systematic
    color shifts (e.g. green tint) without requiring a clean reference.
    """
    # pred: [B, 3, H, W]
    ch_mean = pred.mean(dim=(2, 3))  # [B, 3]
    ch_mean_avg = ch_mean.mean(dim=1, keepdim=True)  # [B, 1]
    return (ch_mean - ch_mean_avg).abs().mean()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logger("stage_R_pre")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_R_pre.yaml")
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
    pref_max = cfg["constants"].get("pref_max", 3.0)
    gf_radius = cfg["constants"].get("guided_filter_radius", 3)
    gf_eps = cfg["constants"].get("guided_filter_eps", 0.02)
    model = RetinexAdaReNet(
        illum,
        adarenet,
        omega=cfg["constants"]["omega"],
        tau=cfg["constants"]["tau"],
        eps=cfg["constants"]["eps"],
        illum_adjust_mode=illum_adjust_mode,
        pref_max=pref_max,
        guided_filter_radius=gf_radius,
        guided_filter_eps=gf_eps,
    ).to(device)
    logger.info(f"Model created on {device}")

    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    logger.info(f"Loading illumination checkpoint: {illum_ckpt}")
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))
    logger.info("Illumination checkpoint loaded")

    for p in model.illumination.parameters():
        p.requires_grad = False
    model.illumination.eval()
    logger.info("IlluminationNet frozen, AdaReNet trainable")

    # ── Pre-cache frozen illumination outputs ──────────────────────────────
    # Since IlluminationNet is frozen, L_T/L_e/P_ref are constant per image.
    # Pre-computing them once avoids 20 epochs × 122 steps of redundant work.
    logger.info("Pre-caching illumination outputs (L_e, P_ref) for all images...")
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
    logger.info(f"Cached {len(cached_le)} illumination outputs")

    # Build a lightweight dataset of pre-cached tensors for efficient training
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
    logger.info(f"Optimizer: Adam, lr={cfg['train']['lr']}")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    # Periodic checkpoint folder
    save_interval = cfg["train"].get("save_interval", 20)
    periodic_dir = save_dir / "stage_R_pre"
    periodic_dir.mkdir(parents=True, exist_ok=True)

    epochs       = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    sigma_min    = cfg["noise"]["sigma_min"]
    sigma_max    = cfg["noise"]["sigma_max"]

    # Loss weights
    lambda_delta = cfg["train"].get("lambda_delta", 0.05)
    mask_prob    = cfg["train"].get("mask_prob", 0.3)
    lambda_tv    = cfg["train"].get("lambda_tv", 0.05)
    lambda_cb    = cfg["train"].get("lambda_cb", 0.1)

    logger.info(
        f"Training: {epochs} epochs, mask_prob={mask_prob}, sigma=[{sigma_min},{sigma_max}], "
        f"lambda_delta={lambda_delta}, lambda_tv={lambda_tv}, lambda_cb={lambda_cb}"
    )

    logger.info("=" * 80)
    logger.info("Starting Stage-R-pre (Reflectance Pretraining with Noise2Self-style Masked Loss)")
    logger.info("Losses: L_ss(masked-only) + TV(R_e) + channel_balance(R_e) + delta_reg")
    logger.info("=" * 80)
    # AMP (mixed precision) for ~1.5-2x speedup on modern GPUs
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("AMP (mixed precision) enabled")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_ss = 0.0
        epoch_tv = 0.0
        epoch_cb = 0.0
        epoch_reg = 0.0
        for step, batch in enumerate(tqdm(cached_loader, desc=f"Stage-R-pre Epoch {epoch}", ncols=80)):
            l_e = batch["l_e"].to(device, non_blocking=True)
            p_ref = batch["p_ref"].to(device, non_blocking=True)

            # Mask-based inpainting supervision (Noise2Self principle)
            # Always use masked input — the key Noise2Self guarantee:
            # predicting masked pixels from unmasked neighbors converges
            # to the clean signal, NOT the noise.
            p_tilde, M = construct_masked_reflectance(p_ref, mask_prob, sigma_min, sigma_max)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Network predicts residual on masked reflectance
                delta = model.adarenet(torch.cat([p_tilde, l_e], dim=1))

                # Recover reflectance from masked input
                r_tilde = p_tilde - delta

                # ── 1) Masked-pixel-only inpainting loss (Noise2Self) ──
                # Only supervise on masked pixels: the network must predict
                # them from their neighbors, so it learns the clean signal.
                diff = r_tilde - p_ref
                loss_ss = (M * diff.abs()).sum() / (M.sum() + 1e-6)

                # ── 2) TV smoothness on recovered reflectance ──
                # Implicit denoising prior: encourages R_e to be smooth.
                # Unlike gradient_loss(pred, noisy_P_ref), this does NOT
                # force the output to match noisy gradients.
                loss_tv = tv_loss_on_reflectance(r_tilde)

                # ── 3) Channel balance prior (gray-world) ──
                # Prevents systematic color shift (green tint) without
                # needing a clean reference image.
                loss_cb = channel_balance_loss(r_tilde)

                # ── 4) Delta regularization (mild) ──
                loss_reg = lambda_delta * delta.abs().mean()

                loss = loss_ss + lambda_tv * loss_tv + lambda_cb * loss_cb + loss_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            epoch_ss    += loss_ss.item()
            epoch_tv    += loss_tv.item()
            epoch_cb    += loss_cb.item()
            epoch_reg   += loss_reg.item()

            if (step + 1) % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Stage-R-pre][E{epoch}][S{step+1}] loss={loss.item():.6f} "
                    f"ss={loss_ss.item():.6f} tv={loss_tv.item():.6f} "
                    f"cb={loss_cb.item():.6f} reg={loss_reg.item():.6f} lr={lr:.6g}"
                )
                print(msg)
                logger.info(msg)

        n = max(1, len(cached_loader))
        msg = (
            f"[Stage-R-pre][E{epoch}] avg loss={epoch_loss/n:.6f} "
            f"ss={epoch_ss/n:.6f} tv={epoch_tv/n:.6f} "
            f"cb={epoch_cb/n:.6f} reg={epoch_reg/n:.6f}"
        )
        print(msg)
        logger.info(msg)

        # Periodic checkpoint save
        if epoch % save_interval == 0:
            periodic_path = periodic_dir / f"epoch_{epoch:04d}.pth"
            torch.save(model.adarenet.state_dict(), periodic_path)
            logger.info(f"Periodic checkpoint saved: {periodic_path}")

    torch.save(model.adarenet.state_dict(), save_path)
    logger.info(f"Checkpoint saved: {save_path}")
    logger.info("=" * 80)
    logger.info("Stage-R-pre training completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
