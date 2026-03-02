import torch


class GaussianFeatureExtractor:

    def __init__(self):
        pass

    def extract_geometry_map(self, render_pkg, gaussian_model):
        """
        从 render 输出构建几何特征图
        输出: (3, H, W)
        """

        image = render_pkg["render"]
        _, H, W = image.shape

        device = image.device
        geometry_map = torch.zeros((3, H, W), device=device)

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

        geometry_map[0, ys, xs] += scales[vidx].mean(dim=1)
        geometry_map[1, ys, xs] += opacities[vidx].squeeze()
        geometry_map[2, ys, xs] += 1.0

        return geometry_map