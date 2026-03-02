import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAwarePatchCore(nn.Module):
    """
    PatchCore-style memory bank anomaly detection with Gaussian geometry priors.

    Inputs:
      - image: (B,3,H,W)
      - geometry_map: (B,Cg,H,W)  (e.g., 5 channels from GaussianFeatureExtractor)
    Outputs:
      - fused feature map: (B,Cr,H',W')
      - anomaly maps:
          score_plain: (B,H',W')  baseline distance
          score_geo:   (B,H',W')  geometry-reweighted distance
    """

    def __init__(self, backbone, geo_channels=5, fusion_out_channels=None):
        super().__init__()
        self.backbone = backbone
        self.geo_channels = geo_channels

        Cr = backbone.out_channels
        Cout = Cr if fusion_out_channels is None else fusion_out_channels

        # feature-level fusion (concat -> 1x1 conv)
        self.fusion_conv = nn.Conv2d(
            in_channels=Cr + geo_channels,
            out_channels=Cout,
            kernel_size=1
        )

        # memory bank (NxC)
        self.register_buffer("memory_bank", torch.empty(0))

    def forward_features(self, image, geometry_map):
        feat_rgb = self.backbone(image)  # (B,Cr,H',W')

        geometry_map = F.interpolate(
            geometry_map,
            size=feat_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        feat_fused = torch.cat([feat_rgb, geometry_map], dim=1)
        feat_fused = self.fusion_conv(feat_fused)

        return feat_fused, geometry_map

    @torch.no_grad()
    def add_to_memory(self, feat_fused):
        """
        feat_fused: (B,C,H',W') -> append to memory bank as (N,C)
        """
        B, C, H, W = feat_fused.shape
        flat = feat_fused.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        if self.memory_bank.numel() == 0:
            self.memory_bank = flat.detach()
        else:
            self.memory_bank = torch.cat([self.memory_bank, flat.detach()], dim=0)

    def _nn_min_dist(self, flat_feat):
        """
        flat_feat: (M,C)
        returns: (M,) min distance to memory_bank
        """
        # For large memory banks, torch.cdist can be heavy.
        # This is a baseline implementation; later we can add FAISS or chunked cdist.
        dists = torch.cdist(flat_feat, self.memory_bank)  # (M,N)
        return dists.min(dim=1).values

    @torch.no_grad()
    def compute_anomaly(self, image, geometry_map, geo_weight_mode="sigmoid"):
        """
        Returns:
          score_plain: (B,H',W')
          score_geo:   (B,H',W')
        """
        feat_fused, geo_aligned = self.forward_features(image, geometry_map)
        B, C, H, W = feat_fused.shape

        flat = feat_fused.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W,C)
        min_dist = self._nn_min_dist(flat)  # (B*H*W,)

        score_plain = min_dist.reshape(B, H, W)

        # ----- Geometry-guided reweighting -----
        # We build a weight map w in [1, 1+alpha] (or [0,1]) and apply:
        # score_geo = score_plain * w
        #
        # geo_aligned: (B,Cg,H,W) assumed normalized to [0,1]
        w = self._geo_weight(geo_aligned, mode=geo_weight_mode)  # (B,1,H,W)
        score_geo = score_plain * w.squeeze(1)

        return score_plain, score_geo

    def _geo_weight(self, geo_aligned, mode="sigmoid", alpha=1.0):
        """
        geo_aligned: (B,Cg,H,W), normalized to [0,1]
        Produce a (B,1,H,W) weight map.
        """
        # A simple, interpretable weighting strategy:
        # - density channel (index 3): high density -> more reliable regions
        # - opacity channel (index 2): low opacity often indicates missing material -> may be abnormal
        # - anisotropy channel (index 4): high anisotropy -> erosion / directional structures
        #
        # You can tune channel usage later.
        density = geo_aligned[:, 3:4] if geo_aligned.shape[1] > 3 else 0.0
        opacity = geo_aligned[:, 2:3] if geo_aligned.shape[1] > 2 else 0.0
        anis = geo_aligned[:, 4:5] if geo_aligned.shape[1] > 4 else 0.0

        # base score in [0,1] roughly
        base = 0.5 * density + 0.25 * (1.0 - opacity) + 0.25 * anis  # (B,1,H,W)

        if mode == "linear":
            # w in [1, 1+alpha]
            w = 1.0 + alpha * base
        elif mode == "sigmoid":
            # w in [1, 1+alpha] with softer saturation
            w = 1.0 + alpha * torch.sigmoid(4.0 * (base - 0.5))
        elif mode == "raw01":
            # w in [0,1] (not recommended for scaling distances)
            w = base.clamp(0, 1)
        else:
            raise ValueError(f"Unknown geo_weight_mode: {mode}")

        return w