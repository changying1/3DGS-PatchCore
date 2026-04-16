import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class GaussianPatchCoreDataset(Dataset):
    def __init__(self, data_dir, crop_ratio=0.7):
        self.data_dir = data_dir
        self.crop_ratio = crop_ratio
        self.rgb_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("rgb_")
        ])

        self.to_tensor = T.ToTensor()

    def load_geometry(self, idx):
        geom_list = []
        for c in range(5):
            path = os.path.join(self.data_dir, f"geometry_c{c}_{idx}.png")
            img = Image.open(path).convert("L")
            geom_list.append(self.to_tensor(img))

        geom = torch.cat(geom_list, dim=0)   # [5, H, W]
        return geom

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, i):
        rgb_name = self.rgb_files[i]
        idx = rgb_name.split("_")[1].split(".")[0]

        rgb_path = os.path.join(self.data_dir, rgb_name)
        rgb = Image.open(rgb_path).convert("RGB")

        w, h = rgb.size
        crop_size = int(min(w, h) * self.crop_ratio)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2

        rgb = rgb.crop((left, top, left + crop_size, top + crop_size))
        rgb = self.to_tensor(rgb)

        geom = self.load_geometry(idx)
        geom = geom[:, top:top + crop_size, left:left + crop_size]

        x = torch.cat([rgb, geom], dim=0)

        return x, {
            "real_idx": int(idx),
            "crop_size": crop_size
        }