from gaussian_patchcore_dataset import GaussianPatchCoreDataset
from gaussian_patchcore_model import GaussianPatchCoreBackbone

dataset = GaussianPatchCoreDataset(
    "../gaussian-splatting/output/test"
)

x = dataset[0]

print("input shape:", x.shape)

model = GaussianPatchCoreBackbone()

out = model(x.unsqueeze(0))

print("feature shape:", out.shape)