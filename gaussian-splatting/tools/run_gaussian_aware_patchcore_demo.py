import torch

from models.backbones.resnet_feature import ResNetFeatureExtractor
from models.gaussian_aware_patchcore import GaussianAwarePatchCore


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = ResNetFeatureExtractor(name="resnet50", pretrained=False, out_layer="layer3").to(device)
    model = GaussianAwarePatchCore(backbone=backbone, geo_channels=5).to(device)
    model.eval()

    # Fake batch
    B, H, W = 2, 256, 256
    image = torch.rand(B, 3, H, W, device=device)
    geometry_map = torch.rand(B, 5, H, W, device=device)  # assume normalized [0,1]

    # Build memory bank with a few batches (demo)
    with torch.no_grad():
        for _ in range(3):
            feat_fused, _ = model.forward_features(image, geometry_map)
            model.add_to_memory(feat_fused)

        score_plain, score_geo = model.compute_anomaly(image, geometry_map, geo_weight_mode="sigmoid")

    print("memory_bank:", tuple(model.memory_bank.shape))
    print("score_plain:", tuple(score_plain.shape))
    print("score_geo  :", tuple(score_geo.shape))


if __name__ == "__main__":
    main()