import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class GaussianCrackSegDataset(Dataset):
    def __init__(self, crack_dir, mask_dir, geom_dir, mode="rgb_g01234"):
        self.crack_dir = crack_dir
        self.mask_dir = mask_dir
        self.geom_dir = geom_dir
        self.mode = mode
        self.to_tensor = T.ToTensor()

        self.crack_files = sorted([
            f for f in os.listdir(crack_dir)
            if f.startswith("crack_") and f.endswith(".png")
        ])

        if len(self.crack_files) == 0:
            raise RuntimeError(f"No crack_*.png found in {crack_dir}")

    def __len__(self):
        return len(self.crack_files)

    def _load_geometry(self, idx):
        npz_path = os.path.join(self.geom_dir, f"geometry_map_{idx}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing geometry file: {npz_path}")

        data = np.load(npz_path)
        geom = data["geometry"]  # [5, H, W]
        geom = torch.from_numpy(geom).float()
        return geom

    def __getitem__(self, i):
        crack_name = self.crack_files[i]
        idx = crack_name.split("_")[1].split(".")[0]

        crack_path = os.path.join(self.crack_dir, crack_name)
        mask_path = os.path.join(self.mask_dir, f"mask_{idx}.png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

        img = Image.open(crack_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.to_tensor(img)              # [3, H, W]
        mask = self.to_tensor(mask)            # [1, H, W]
        mask = (mask > 0.5).float()

        geom = self._load_geometry(idx)        # [5, H, W]

        if img.shape[1:] != geom.shape[1:]:
            raise ValueError(
                f"Image and geometry size mismatch for idx={idx}: "
                f"img={img.shape}, geom={geom.shape}"
            )

        if self.mode == "rgb":
            x = img
        elif self.mode == "rgb_g0":
            x = torch.cat([img, geom[0:1]], dim=0)
        elif self.mode == "rgb_g01234":
            x = torch.cat([img, geom], dim=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        meta = {
            "idx": int(idx),
            "crack_name": crack_name,
        }

        return x, mask, meta