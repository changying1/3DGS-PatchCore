import torch
import torch.nn.functional as F
import torchvision
import os
from tqdm import tqdm

from gaussian_patchcore_dataset import GaussianPatchCoreDataset
from gaussian_patchcore_model import GaussianPatchCoreBackbone


# ------------------------
# config
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = os.environ.get(
    "DATA_DIR",
    "../gaussian-splatting/output/test/test_0"
)
memory_bank_path = "memory_bank_coreset.pt"
save_dir = "anomaly_maps/test_0"
weight_path = os.environ.get("BACKBONE_CKPT", "backbone_finetuned.pth")

os.makedirs(save_dir, exist_ok=True)


# ------------------------
# helper: laplacian prior
# ------------------------

def get_laplacian_prior(x_rgb: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    x_rgb: [1, 3, Hc, Wc], value range roughly [0,1]
    return: [H, W] normalized laplacian map
    """
    # 转灰度
    gray = (
        0.2989 * x_rgb[:, 0:1, :, :] +
        0.5870 * x_rgb[:, 1:2, :, :] +
        0.1140 * x_rgb[:, 2:3, :, :]
    )

    # 3x3 Laplacian
    lap_kernel = torch.tensor(
        [[0.0,  1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0,  1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype
    ).view(1, 1, 3, 3)

    lap = F.conv2d(gray, lap_kernel, padding=1)
    lap = torch.abs(lap)

    # 上采样到 anomaly feature map 分辨率
    lap = F.interpolate(
        lap,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    # 归一化
    lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-6)

    return lap


# ------------------------
# load memory bank
# ------------------------

memory_bank = torch.load(memory_bank_path, map_location=device)
memory_bank = F.normalize(memory_bank, dim=1).to(device)

print("Memory bank:", memory_bank.shape)


# ------------------------
# load model
# ------------------------

model = GaussianPatchCoreBackbone().to(device)

if os.path.exists(weight_path):
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    print(f"Loaded backbone: {weight_path}")
else:
    print(f"[WARN] backbone checkpoint not found: {weight_path}")

model.eval()


# ------------------------
# dataset
# ------------------------

dataset = GaussianPatchCoreDataset(data_dir)


# ------------------------
# detection
# ------------------------

for idx in tqdm(range(len(dataset))):
    x, meta = dataset[idx]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x)

    B, C, H, W = feat.shape

    patches = feat.permute(0, 2, 3, 1).reshape(-1, C)
    patches = F.normalize(patches, dim=1)

    # compute distance
    dist = torch.cdist(patches, memory_bank)

    topk = torch.topk(dist, k=9, largest=False)[0]
    score_plain = topk.mean(dim=1)
    anomaly_map = score_plain.reshape(H, W)

    # normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-6
    )

    # gaussian smoothing
    anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

    kernel_size = 11
    sigma = 2.0

    grid = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    anomaly_map = F.conv2d(
        anomaly_map,
        kernel,
        padding=kernel_size // 2
    )

    anomaly_map = anomaly_map[0, 0]

    # normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-6
    )

    # ------------------------
    # scheme C: laplacian prior reweight
    # ------------------------
    rgb = x[:, 0:3, :, :]  # [1,3,Hc,Wc]
    lap_map = get_laplacian_prior(rgb, H, W)

    # 只做轻量增强，避免把所有边缘都打亮
    alpha = 0.20
    anomaly_map = anomaly_map * (1.0 + alpha * lap_map)

    # re-normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-6
    )

    # upsample back to cropped image size
    anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(meta["crop_size"], meta["crop_size"]),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    save_path = os.path.join(save_dir, f"anomaly_{idx}.png")
    torchvision.utils.save_image(anomaly_map.cpu(), save_path)

print("Done")