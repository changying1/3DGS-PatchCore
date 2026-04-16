from gaussian_seg_dataset import GaussianSegDataset

data_dir = r"../gaussian-splatting/output/test/test_1"

for mode in ["rgb", "rgb_g0", "rgb_g01234"]:
    ds = GaussianSegDataset(data_dir=data_dir, mode=mode, crop_ratio=0.7)
    x, meta = ds[0]
    print("=" * 50)
    print("mode:", mode)
    print("x.shape:", x.shape)
    print("x.dtype:", x.dtype)
    print("meta:", meta)
    print("x.min:", x.min().item(), "x.max:", x.max().item())