
import os
import argparse
import random
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset, random_split

from gaussian_crack_seg_dataset import GaussianCrackSegDataset
from unet_model import UNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_in_channels(mode: str) -> int:
    if mode == "rgb":
        return 3
    elif mode == "rgb_g0":
        return 4
    elif mode == "rgb_g0123":
        return 7
    elif mode == "rgb_g01234":
        return 8
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_uint8_gray(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def save_rgb_tensor(x: torch.Tensor, save_path: str):
    """
    x: [C, H, W], 这里只保存前3通道作为RGB输入图
    """
    rgb = x[:3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(save_path)


def save_mask_tensor(mask: torch.Tensor, save_path: str):
    """
    mask: [1, H, W] 或 [H, W]
    """
    if mask.ndim == 3:
        mask = mask[0]
    mask_np = mask.detach().cpu().numpy()
    Image.fromarray(to_uint8_gray(mask_np)).save(save_path)


def make_overlay(rgb_tensor: torch.Tensor, pred_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    pred_mask: [H, W], 0/1
    把预测结果以红色叠加到 RGB 上
    """
    rgb = rgb_tensor[:3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0.0, 1.0)

    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0  # 红色通道

    mask_3 = pred_mask[..., None].astype(np.float32)
    overlay = overlay * (1 - alpha * mask_3) + red * (alpha * mask_3)
    overlay = np.clip(overlay, 0.0, 1.0)

    overlay_u8 = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_u8)


def build_dataset(
    crack_dir: str,
    mask_dir: str,
    geom_dir: str,
    mode: str,
):
    return GaussianCrackSegDataset(
        crack_dir=crack_dir,
        mask_dir=mask_dir,
        geom_dir=geom_dir,
        mode=mode,
    )


def build_subset(
    dataset,
    split: str,
    val_ratio: float,
    seed: int,
):
    total_len = len(dataset)

    if split == "all":
        return dataset

    val_len = max(1, int(total_len * val_ratio))
    train_len = total_len - val_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    if split == "train":
        return train_set
    elif split == "val":
        return val_set
    else:
        raise ValueError(f"Unsupported split: {split}")


def infer_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    threshold: float,
    max_samples: Optional[int] = None,
):
    model.eval()
    ensure_dir(save_dir)

    count = 0
    with torch.no_grad():
        for batch in loader:
            x, mask, meta = batch
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            bs = x.size(0)
            for b in range(bs):
                idx = int(meta["idx"][b].item()) if torch.is_tensor(meta["idx"][b]) else int(meta["idx"][b])
                crack_name = meta["crack_name"][b]

                sample_dir = os.path.join(save_dir, f"{idx:04d}")
                ensure_dir(sample_dir)

                x_b = x[b].cpu()
                mask_b = mask[b].cpu()
                prob_b = probs[b, 0].cpu().numpy()
                pred_b = preds[b, 0].cpu().numpy()

                # 1. 输入RGB
                save_rgb_tensor(x_b, os.path.join(sample_dir, "input_rgb.png"))

                # 2. GT
                save_mask_tensor(mask_b, os.path.join(sample_dir, "gt_mask.png"))

                # 3. 概率图
                Image.fromarray(to_uint8_gray(prob_b)).save(os.path.join(sample_dir, "pred_prob.png"))

                # 4. 二值图
                Image.fromarray((pred_b * 255).astype(np.uint8)).save(os.path.join(sample_dir, "pred_mask.png"))

                # 5. overlay
                overlay = make_overlay(x_b, pred_b.astype(np.uint8), alpha=0.45)
                overlay.save(os.path.join(sample_dir, "overlay.png"))

                # 6. 保存简单meta
                with open(os.path.join(sample_dir, "meta.txt"), "w", encoding="utf-8") as f:
                    f.write(f"idx: {idx}\n")
                    f.write(f"crack_name: {crack_name}\n")
                    f.write(f"threshold: {threshold}\n")

                count += 1
                if max_samples is not None and count >= max_samples:
                    print(f"[OK] reach max_samples={max_samples}, stop inference.")
                    return

    print(f"[OK] inference finished, total saved samples: {count}")


def main():
    parser = argparse.ArgumentParser()

    # checkpoint / mode
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint, e.g. ./checkpoints_unet/rgb_g01234/best.pth")
    parser.add_argument("--mode", type=str, default=None, choices=["rgb", "rgb_g0", "rgb_g0123", "rgb_g01234"],
                        help="If not given, try reading mode from checkpoint")

    # data
    parser.add_argument("--crack_dir", type=str, default=r"./synthetic_crack_images/test_r1")
    parser.add_argument("--mask_dir", type=str, default=r"./synthetic_crack_masks/test_r1")
    parser.add_argument("--geom_dir", type=str, default=r"../gaussian-splatting/output/test/test_r1")

    # split
    parser.add_argument("--split", type=str, default="val", choices=["all", "train", "val"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--base_channels", type=int, default=16)

    # inference
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)

    # save
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)

    # mode 优先级：命令行 > checkpoint
    mode = args.mode if args.mode is not None else ckpt.get("mode", None)
    if mode is None:
        raise ValueError("Mode is not provided and not found in checkpoint.")

    print(f"mode: {mode}")

    dataset = build_dataset(
        crack_dir=args.crack_dir,
        mask_dir=args.mask_dir,
        geom_dir=args.geom_dir,
        mode=mode,
    )

    subset = build_subset(
        dataset=dataset,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(
        in_channels=get_in_channels(mode),
        out_channels=1,
        base_channels=args.base_channels
    ).to(device)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[OK] loaded checkpoint: {args.ckpt}")
    if "epoch" in ckpt:
        print(f"[OK] checkpoint epoch: {ckpt['epoch']}")

    if args.save_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_dir = os.path.join("./infer_results", mode, args.split, ckpt_name)
    else:
        save_dir = args.save_dir

    ensure_dir(save_dir)
    print(f"save_dir: {save_dir}")

    infer_one_epoch(
        model=model,
        loader=loader,
        device=device,
        save_dir=save_dir,
        threshold=args.threshold,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()