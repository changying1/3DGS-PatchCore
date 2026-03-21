print("LOADED gaussian_feature_extractor:", __file__)
import torch
import torch.nn.functional as F


class GaussianFeatureExtractor:
    def __init__(self, downsample_factor=8):
        self.downsample_factor = downsample_factor

    def extract_geometry_map(self, render_pkg, gaussian_model=None):
        """
        输出:
            geometry_map: (5, H, W)

        改为基于 depth 的 5 通道几何图，不再使用 viewspace_points。
        通道定义:
            c0: normalized depth
            c1: |dx|
            c2: |dy|
            c3: gradient magnitude
            c4: laplacian magnitude
        """
        image = render_pkg["render"]
        _, H, W = image.shape
        device = image.device

        geometry_map = torch.zeros((5, H, W), device=device, dtype=image.dtype)

        depth = render_pkg.get("depth", None)
        if depth is None:
            return geometry_map

        # depth: (1,H,W) -> (H,W)
        if depth.dim() == 3:
            depth = depth.squeeze(0)

        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # 有效深度掩码
        valid = torch.isfinite(depth) & (depth > 1e-3)
        if valid.sum() == 0:
            return geometry_map

        ys, xs = torch.where(valid)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        pad = 8
        y0 = max(int(y0) - pad, 0)
        y1 = min(int(y1) + pad, H - 1)
        x0 = max(int(x0) - pad, 0)
        x1 = min(int(x1) + pad, W - 1)

        roi_mask = torch.zeros_like(valid, dtype=torch.bool)
        roi_mask[y0:y1+1, x0:x1+1] = True

        valid = valid & roi_mask

        # 归一化深度（仅对有效区域）
        d = depth.clone()
        d_valid = d[valid]
        d_min = d_valid.min()
        d_max = d_valid.max()

        d_norm = torch.zeros_like(d)
        if (d_max - d_min) > 1e-8:
            d_norm[valid] = (d[valid] - d_min) / (d_max - d_min + 1e-6)

        # 一阶差分
        dx = torch.zeros_like(d_norm)
        dy = torch.zeros_like(d_norm)

        dx[:, :-1] = d_norm[:, 1:] - d_norm[:, :-1]
        dy[:-1, :] = d_norm[1:, :] - d_norm[:-1, :]

        dx = dx.abs()
        dy = dy.abs()
        grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-12)

        # 二阶差分 / 粗略曲率
        lap = torch.zeros_like(d_norm)
        lap[1:-1, 1:-1] = (
            d_norm[1:-1, :-2]
            + d_norm[1:-1, 2:]
            + d_norm[:-2, 1:-1]
            + d_norm[2:, 1:-1]
            - 4.0 * d_norm[1:-1, 1:-1]
        ).abs()

        # 关键：弱响应阈值抑制
        dx[dx < 0.03] = 0
        dy[dy < 0.03] = 0
        grad_mag[grad_mag < 0.03] = 0
        lap[lap < 0.05] = 0

        mask = valid.float()
        dx *= mask
        dy *= mask
        grad_mag *= mask
        lap *= mask
        d_norm *= mask

        geometry_map[0] = d_norm
        geometry_map[1] = dx
        geometry_map[2] = dy
        geometry_map[3] = grad_mag
        geometry_map[4] = lap

        # 无效区域清零
        geometry_map *= mask.unsqueeze(0)

        geometry_map = self.normalize(geometry_map)
        geometry_map *= mask.unsqueeze(0)
        return geometry_map

    def normalize(self, geometry_map):
        """
        每通道归一化到 [0,1]
        """
        for c in range(geometry_map.shape[0]):
            channel = torch.nan_to_num(
                geometry_map[c], nan=0.0, posinf=0.0, neginf=0.0
            )
            min_val = channel.min()
            max_val = channel.max()
            if (max_val - min_val) > 1e-8:
                geometry_map[c] = (channel - min_val) / (max_val - min_val + 1e-6)
            else:
                geometry_map[c] = torch.zeros_like(channel)
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