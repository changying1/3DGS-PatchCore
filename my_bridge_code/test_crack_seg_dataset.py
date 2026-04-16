from gaussian_crack_seg_dataset import GaussianCrackSegDataset

crack_dir = r"./synthetic_crack_images/test_r1"
mask_dir = r"./synthetic_crack_masks/test_r1"
geom_dir = r"../gaussian-splatting/output/test/test_r1"

for mode in ["rgb", "rgb_g0", "rgb_g01234"]:
    ds = GaussianCrackSegDataset(
        crack_dir=crack_dir,
        mask_dir=mask_dir,
        geom_dir=geom_dir,
        mode=mode
    )
    x, mask, meta = ds[0]
    print("=" * 50)
    print("mode:", mode)
    print("x.shape:", x.shape, x.dtype)
    print("mask.shape:", mask.shape, mask.dtype)
    print("mask unique:", mask.unique())
    print("meta:", meta)
    print("x min/max:", x.min().item(), x.max().item())