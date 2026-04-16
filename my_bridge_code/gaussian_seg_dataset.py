import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class GaussianSegDataset(Dataset):
    def __init__(self, data_dir, mode="rgb_g01234", crop_ratio=0.7):
        """
        data_dir:
            gaussian-splatting/output/test/test_1
        mode:
            "rgb", "rgb_g0", "rgb_g01234"
        """
        self.data_dir = data_dir
        self.mode = mode
        self.crop_ratio = crop_ratio
        self.to_tensor = T.ToTensor()

        self.rgb_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("rgb_") and f.endswith(".png")
        ])

        if len(self.rgb_files) == 0:
            raise RuntimeError(f"No rgb_*.png found in {data_dir}")

    def __len__(self):
        return len(self.rgb_files)

    def _center_crop_box(self, w, h):
        crop_size = int(min(w, h) * self.crop_ratio)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return left, top, crop_size

    def _load_geometry_npz(self, idx):
        npz_path = os.path.join(self.data_dir, f"geometry_map_{idx}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Geometry npz not found: {npz_path}")

        data = np.load(npz_path)
        geom = data["geometry"]  # [5, H, W], float32

        if geom.ndim != 3 or geom.shape[0] != 5:
            raise ValueError(f"Unexpected geometry shape: {geom.shape}")

        geom = torch.from_numpy(geom).float()
        return geom

    def __getitem__(self, i):
        rgb_name = self.rgb_files[i]
        idx = rgb_name.split("_")[1].split(".")[0]

        rgb_path = os.path.join(self.data_dir, rgb_name)
        rgb = Image.open(rgb_path).convert("RGB")

        w, h = rgb.size
        left, top, crop_size = self._center_crop_box(w, h)

        rgb = rgb.crop((left, top, left + crop_size, top + crop_size))
        rgb = self.to_tensor(rgb)  # [3, H, W]

        geom = self._load_geometry_npz(idx)  # [5, H, W]
        geom = geom[:, top:top + crop_size, left:left + crop_size]

        if self.mode == "rgb":
            x = rgb
        elif self.mode == "rgb_g0":
            x = torch.cat([rgb, geom[0:1]], dim=0)   # [4, H, W]
        elif self.mode == "rgb_g01234":
            x = torch.cat([rgb, geom], dim=0)        # [8, H, W]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        meta = {
            "idx": int(idx),
            "rgb_name": rgb_name,
            "crop_size": crop_size,
            "left": left,
            "top": top,
        }

        return x, meta