import torch
import os
from tqdm import tqdm

from gaussian_patchcore_dataset import GaussianPatchCoreDataset
from gaussian_patchcore_model import GaussianPatchCoreBackbone

data_dir = "../gaussian-splatting/output/test"

dataset = GaussianPatchCoreDataset(data_dir)

model = GaussianPatchCoreBackbone()
model.eval()

memory_bank = []

for i in tqdm(range(len(dataset))):

    x = dataset[i].unsqueeze(0)

    with torch.no_grad():
        feat = model(x)

    B, C, H, W = feat.shape

    patches = feat.permute(0,2,3,1).reshape(-1,C)

    memory_bank.append(patches)

memory_bank = torch.cat(memory_bank)

print("Memory bank size:", memory_bank.shape)

torch.save(memory_bank, "memory_bank.pt")