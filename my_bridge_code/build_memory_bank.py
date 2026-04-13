import os
import torch
from tqdm import tqdm
from gaussian_patchcore_dataset import GaussianPatchCoreDataset
from gaussian_patchcore_model import GaussianPatchCoreBackbone

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = os.environ.get("DATA_DIR", "../gaussian-splatting/output/test/test_0")
weight_path = os.environ.get("BACKBONE_CKPT", "backbone_finetuned.pth")

dataset = GaussianPatchCoreDataset(data_dir)
model = GaussianPatchCoreBackbone().to(device)

if os.path.exists(weight_path):
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded backbone: {weight_path}")

model.eval()

memory_bank = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(device)
        feat = model(x)                         # [1,C,H,W]
        patches = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        patches = torch.nn.functional.normalize(patches, dim=1)
        memory_bank.append(patches.cpu())

memory_bank = torch.cat(memory_bank, dim=0)
print("Memory bank size:", memory_bank.shape)
torch.save(memory_bank, "memory_bank.pt")