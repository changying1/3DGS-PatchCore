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

        通道定义:
            c0: normalized depth
            c1: |dx|
            c2: |dy|
            c3: gradient magnitude
            c4: laplacian magnitude
        """
        image = render_pkg["render"]   # (3,H,W)
        _, H, W = image.shape
        device = image.device
        dtype = image.dtype

        geometry_map = torch.zeros((5, H, W), device=device, dtype=dtype)

        depth = render_pkg.get("depth", None)
        if depth is None:
            return geometry_map

        if depth.dim() == 3:
            depth = depth.squeeze(0)

        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # --------------------------------------------------
        # 1) 白背景下不要只靠 depth，有些漂浮噪点也会有正深度
        #    用“与白背景的差异”做前景 mask，更稳
        # --------------------------------------------------
        valid = torch.isfinite(depth) & (depth > 1e-4)

        if valid.sum() == 0:
            return geometry_map

        # --------------------------------------------------
        # 2) 对 mask 做一点平滑/闭合，减少洞和碎噪声
        # --------------------------------------------------
        valid_f = valid.float().unsqueeze(0).unsqueeze(0)
        valid_f = F.max_pool2d(valid_f, kernel_size=5, stride=1, padding=2)
        valid_f = F.avg_pool2d(valid_f, kernel_size=5, stride=1, padding=2)
        valid = (valid_f.squeeze() > 0.35)

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
        roi_mask[y0:y1 + 1, x0:x1 + 1] = True
        valid = valid & roi_mask

        if valid.sum() == 0:
            return geometry_map

        # --------------------------------------------------
        # 3) 白背景更容易被异常深度值拉坏，改成鲁棒分位数归一化
        # --------------------------------------------------
        d = depth.clone()
        d_valid = d[valid]

        q_low = torch.quantile(d_valid, 0.05)
        q_high = torch.quantile(d_valid, 0.95)

        d_clip = torch.clamp(d, q_low, q_high)
        d_norm = torch.zeros_like(d)

        if (q_high - q_low) > 1e-8:
            d_norm[valid] = (d_clip[valid] - q_low) / (q_high - q_low + 1e-6)

        # --------------------------------------------------
        # 4) 先轻微平滑，再求梯度，减少白背景下边缘闪噪
        # --------------------------------------------------
        d_smooth = F.avg_pool2d(
            d_norm.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze(0).squeeze(0)

        dx = torch.zeros_like(d_smooth)
        dy = torch.zeros_like(d_smooth)

        dx[:, :-1] = d_smooth[:, 1:] - d_smooth[:, :-1]
        dy[:-1, :] = d_smooth[1:, :] - d_smooth[:-1, :]

        dx = dx.abs()
        dy = dy.abs()
        grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-12)

        lap = torch.zeros_like(d_smooth)
        lap[1:-1, 1:-1] = (
            d_smooth[1:-1, :-2]
            + d_smooth[1:-1, 2:]
            + d_smooth[:-2, 1:-1]
            + d_smooth[2:, 1:-1]
            - 4.0 * d_smooth[1:-1, 1:-1]
        ).abs()

        # --------------------------------------------------
        # 5) 不要固定阈值，白背景下改成自适应阈值更稳
        # --------------------------------------------------
        def adaptive_suppress(x, mask, q=0.70):
            vals = x[mask]
            if vals.numel() < 16:
                return x * mask.float()
            thr = torch.quantile(vals, q)
            x = torch.where(x >= thr, x, torch.zeros_like(x))
            return x * mask.float()

        mask = valid.float()

        d_norm = d_norm * mask
        dx = adaptive_suppress(dx, valid, q=0.60)
        dy = adaptive_suppress(dy, valid, q=0.60)
        grad_mag = adaptive_suppress(grad_mag, valid, q=0.65)
        lap = adaptive_suppress(lap, valid, q=0.68)

        geometry_map[0] = d_norm
        geometry_map[1] = dx
        geometry_map[2] = dy
        geometry_map[3] = grad_mag
        geometry_map[4] = lap

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