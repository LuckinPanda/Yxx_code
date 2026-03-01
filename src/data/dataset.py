from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.utils.image import is_image_file, load_image


class LowLightDataset(Dataset):
    def __init__(
        self,
        mode: str,
        source_low_dir: Optional[str],
        source_high_dir: Optional[str],
        target_low_dir: Optional[str],
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.mode = mode
        self.resize = resize

        if mode == "source_paired":
            if not source_low_dir or not source_high_dir:
                raise ValueError("source_low_dir and source_high_dir are required for source_paired")
            self.low_paths, self.high_paths = self._build_paired_paths(source_low_dir, source_high_dir)
        elif mode == "paired_by_index":
            if not source_low_dir or not source_high_dir:
                raise ValueError("source_low_dir and source_high_dir are required for paired_by_index")
            self.low_paths, self.high_paths = self._build_paired_by_index_paths(source_low_dir, source_high_dir)
        elif mode == "source_low_only":
            if not source_low_dir:
                raise ValueError("source_low_dir is required for source_low_only")
            self.low_paths = self._build_single_paths(source_low_dir)
            self.high_paths = None
        elif mode == "target_low_only":
            if not target_low_dir:
                raise ValueError("target_low_dir is required for target_low_only")
            self.low_paths = self._build_single_paths(target_low_dir)
            self.high_paths = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _build_single_paths(self, dir_path: str) -> List[Path]:
        root = Path(dir_path)
        paths = [p for p in root.iterdir() if p.is_file() and is_image_file(p)]
        paths.sort()
        return paths

    def _build_paired_paths(self, low_dir: str, high_dir: str):
        low_root = Path(low_dir)
        high_root = Path(high_dir)
        low_map = {p.stem: p for p in low_root.iterdir() if p.is_file() and is_image_file(p)}
        high_map = {p.stem: p for p in high_root.iterdir() if p.is_file() and is_image_file(p)}
        keys = sorted(set(low_map.keys()) & set(high_map.keys()))
        low_paths = [low_map[k] for k in keys]
        high_paths = [high_map[k] for k in keys]
        return low_paths, high_paths

    def _build_paired_by_index_paths(self, low_dir: str, high_dir: str):
        low_paths = self._build_single_paths(low_dir)
        high_paths = self._build_single_paths(high_dir)
        n = min(len(low_paths), len(high_paths))
        return low_paths[:n], high_paths[:n]

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int):
        low_path = self.low_paths[idx]
        low = load_image(str(low_path), resize=self.resize).float()
        sample = {"low": low, "name": low_path.name}

        if self.high_paths is not None:
            high_path = self.high_paths[idx]
            high = load_image(str(high_path), resize=self.resize).float()
            sample["high"] = high
        return sample
