import torch
import torch.nn.functional as F


class GaussianFeatureExtractor:

    def __init__(self, downsample_factor=8):
        """
        downsample_factor:
            用于对齐 backbone feature 尺度
            ResNet 通常为 8 或 16
        """
        self.downsample_factor = downsample_factor

    def extract_geometry_map(self, render_pkg, gaussian_model):
        """
        输出:
            geometry_map: (5, H, W)
        """

        image = render_pkg["render"]
        _, H, W = image.shape
        device = image.device

        geometry_map = torch.zeros((5, H, W), device=device)

        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]

        scales = gaussian_model.get_scaling
        opacities = gaussian_model.get_opacity

        visible_idx = visibility_filter.nonzero(as_tuple=False).squeeze()

        if visible_idx.numel() == 0:
            return geometry_map

        pts = viewspace_points[visible_idx]
        xs = pts[:, 0].long()
        ys = pts[:, 1].long()

        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

        xs = xs[valid]
        ys = ys[valid]
        vidx = visible_idx[valid]

        if vidx.numel() == 0:
            return geometry_map

        scale_values = scales[vidx]
        opacity_values = opacities[vidx].squeeze()

        # 1️⃣ mean scale
        geometry_map[0, ys, xs] += scale_values.mean(dim=1)

        # 2️⃣ max scale
        geometry_map[1, ys, xs] += scale_values.max(dim=1).values

        # 3️⃣ opacity
        geometry_map[2, ys, xs] += opacity_values

        # 4️⃣ density count
        geometry_map[3, ys, xs] += 1.0

        # 5️⃣ anisotropy (scale variance)
        geometry_map[4, ys, xs] += scale_values.var(dim=1)

        geometry_map = self.normalize(geometry_map)

        return geometry_map

    def normalize(self, geometry_map):
        """
        每通道归一化到 [0,1]
        """
        for c in range(geometry_map.shape[0]):
            channel = geometry_map[c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                geometry_map[c] = (channel - min_val) / (max_val - min_val + 1e-6)
        return geometry_map

    def downsample(self, geometry_map):
        """
        对齐 backbone 特征分辨率
        """
        return F.interpolate(
            geometry_map.unsqueeze(0),
            scale_factor=1.0 / self.downsample_factor,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)