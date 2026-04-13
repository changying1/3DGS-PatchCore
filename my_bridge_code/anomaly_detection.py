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

    """
    # geometry-aware reweight
    geo = x[0, 3:, :, :]   # [5, Hc, Wc]
    geo_edge = geo[2:5].mean(dim=0, keepdim=True).unsqueeze(0)
    geo_edge = F.interpolate(
        geo_edge,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    geo_edge = (geo_edge - geo_edge.min()) / (
        geo_edge.max() - geo_edge.min() + 1e-6
    )

    anomaly_map = anomaly_map * (1.0 + 0.35 * geo_edge)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-6
    )
    """
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