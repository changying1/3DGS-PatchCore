import os
import re
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class BackboneTrainDataset(Dataset):
    """
    正常图 / 合成裂纹图 二分类训练集
    返回:
        x: [8, H, W]
        y: 0(normal) / 1(anomaly)
    """

    def __init__(self, normal_dir, crack_dir, crop_ratio=0.7):
        self.normal_dir = normal_dir
        self.crack_dir = crack_dir
        self.crop_ratio = crop_ratio
        self.to_tensor = T.ToTensor()

        self.rgb_files = sorted(
            [f for f in os.listdir(normal_dir) if f.startswith("rgb_") and f.endswith(".png")],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        self.crack_files = sorted(
            [f for f in os.listdir(crack_dir) if f.startswith("crack_") and f.endswith(".png")],
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        # 建立“顺序编号 -> 真实 rgb/geometry idx”映射
        self.real_idx_list = []
        for rgb_name in self.rgb_files:
            real_idx = int(rgb_name.split("_")[1].split(".")[0])
            self.real_idx_list.append(real_idx)

        self.samples = []

        # 正常样本：直接用真实 idx
        for rgb_name in self.rgb_files:
            idx = rgb_name.split("_")[1].split(".")[0]
            self.samples.append(("normal", idx, rgb_name))

        # 异常样本：按 crack 的顺序号映射到真实 idx
        num_crack = min(len(self.crack_files), len(self.real_idx_list))
        for i in range(num_crack):
            crack_name = self.crack_files[i]
            real_idx = str(self.real_idx_list[i])
            self.samples.append(("crack", real_idx, crack_name))

    def __len__(self):
        return len(self.samples)

    def _center_crop_box(self, w, h):
        crop_size = int(min(w, h) * self.crop_ratio)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return left, top, crop_size

    def _load_geometry(self, idx, top, left, crop_size):
        geom_list = []
        for c in range(5):
            path = os.path.join(self.normal_dir, f"geometry_c{c}_{idx}.png")
            img = Image.open(path).convert("L")
            img = self.to_tensor(img)
            img = img[:, top:top + crop_size, left:left + crop_size]
            geom_list.append(img)

        geom = torch.cat(geom_list, dim=0)
        return geom

    def __getitem__(self, i):
        kind, idx, img_name = self.samples[i]

        if kind == "normal":
            img_path = os.path.join(self.normal_dir, img_name)
            y = 0
        else:
            img_path = os.path.join(self.crack_dir, img_name)
            y = 1

        rgb = Image.open(img_path).convert("RGB")

        w, h = rgb.size
        left, top, crop_size = self._center_crop_box(w, h)

        rgb = rgb.crop((left, top, left + crop_size, top + crop_size))
        rgb = self.to_tensor(rgb)

        geom = self._load_geometry(idx, top, left, crop_size)

        x = torch.cat([rgb, geom], dim=0)

        return x, torch.tensor(y, dtype=torch.long)