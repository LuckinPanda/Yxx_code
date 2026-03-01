from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def load_image(path: str, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
