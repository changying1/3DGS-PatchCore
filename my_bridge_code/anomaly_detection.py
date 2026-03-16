import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import os
from tqdm import tqdm

from gaussian_patchcore_dataset import GaussianPatchCoreDataset
from gaussian_patchcore_model import GaussianPatchCoreBackbone


# ------------------------
# config
# ------------------------

data_dir = os.environ.get(
    "DATA_DIR",
    "../gaussian-splatting/output/test/test_0"
)
memory_bank_path = "memory_bank_coreset.pt"
save_dir = "anomaly_maps/test_0"

os.makedirs(save_dir, exist_ok=True)


# ------------------------
# load memory bank
# ------------------------

memory_bank = torch.load(memory_bank_path)
memory_bank = F.normalize(memory_bank, dim=1)

print("Memory bank:", memory_bank.shape)


# ------------------------
# load model
# ------------------------

model = GaussianPatchCoreBackbone()
model.eval()


# ------------------------
# dataset
# ------------------------

dataset = GaussianPatchCoreDataset(data_dir)


# ------------------------
# detection
# ------------------------

for idx in tqdm(range(len(dataset))):

    x = dataset[idx].unsqueeze(0)

    with torch.no_grad():

        f2, f3 = model(x)

    f3 = torch.nn.functional.interpolate(
        f3,
        size=f2.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    feat = torch.cat([f2, f3], dim=1)
    
    B,C,H,W = feat.shape

    patches = feat.permute(0,2,3,1).reshape(-1,C)

    patches = F.normalize(patches,dim=1)


    # compute distance
    dist = torch.cdist(patches, memory_bank)

    topk = torch.topk(dist, k=9, largest=False)[0]
    min_dist = topk.mean(dim=1)
    
    anomaly_map = min_dist.reshape(H,W)

    # normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)

    # gaussian smoothing
    anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

    kernel_size = 21
    sigma = 4.0

    grid = torch.arange(kernel_size).float() - kernel_size//2
    gauss = torch.exp(-(grid**2)/(2*sigma**2))
    kernel = gauss[:,None] * gauss[None,:]
    kernel = kernel / kernel.sum()

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    anomaly_map = F.conv2d(
        anomaly_map,
        kernel,
        padding=kernel_size//2
    )

    anomaly_map = anomaly_map[0,0]


    # normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)


    # upsample
    anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(663,994),
        mode="bilinear",
        align_corners=False
    )[0,0]


    save_path = os.path.join(save_dir, f"anomaly_{idx}.png")

    torchvision.utils.save_image(anomaly_map, save_path)


print("Done")