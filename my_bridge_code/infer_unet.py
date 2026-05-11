
import os
import argparse
import random
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

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
    elif mode == "g0":
        return 1
    elif mode == "g0123":
        return 4
    elif mode == "g01234":
        return 5
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
    Save an input visualization.
    For RGB-containing modes, save first 3 channels as RGB.
    For geometry-only modes, save geometry channels as grayscale visualization.
    """
    x_np = x.detach().cpu().numpy()

    # RGB or RGB+geometry: first 3 channels are RGB
    if x_np.shape[0] >= 3:
        rgb = x_np[:3].transpose(1, 2, 0)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(save_path)
        return

    # geometry-only g0: one channel
    if x_np.shape[0] == 1:
        g = x_np[0]
        g = g.astype(np.float32)
        g_min, g_max = np.percentile(g, 1), np.percentile(g, 99)
        if g_max - g_min < 1e-8:
            g_vis = np.zeros_like(g, dtype=np.uint8)
        else:
            g_vis = (np.clip((g - g_min) / (g_max - g_min), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(g_vis).save(save_path)
        return

    raise ValueError(f"Unsupported input shape for visualization: {x.shape}")


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


def build_one_dataset(
    data_id: str,
    crack_root: str,
    mask_root: str,
    geom_root: str,
    mode: str,
):
    """
    按组级划分读取数据，例如：
    data_id=0 -> test_0
    data_id=1 -> test_1
    data_id=2 -> test_2
    """
    crack_dir = os.path.join(crack_root, f"test_{data_id}")
    mask_dir = os.path.join(mask_root, f"test_{data_id}")
    geom_dir = os.path.join(geom_root, f"test_{data_id}")

    print(f"[DATA] test_{data_id}")
    print(f"  crack_dir = {crack_dir}")
    print(f"  mask_dir  = {mask_dir}")
    print(f"  geom_dir  = {geom_dir}")

    return GaussianCrackSegDataset(
        crack_dir=crack_dir,
        mask_dir=mask_dir,
        geom_dir=geom_dir,
        mode=mode,
    )


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
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[
            "rgb",
            "g0",
            "g0123",
            "g01234",
            "rgb_g0",
            "rgb_g0123",
            "rgb_g01234",
        ],
        help="If not given, try reading mode from checkpoint"
    )

    # data: 组级推理，与 train_unet.py 的 fold 划分保持一致
    parser.add_argument("--data_id", type=str, default="0", help="要推理的组编号，例如 0/1/2")
    parser.add_argument("--crack_root", type=str, default="./synthetic_crack_images_v2")
    parser.add_argument("--mask_root", type=str, default="./synthetic_crack_masks_v2")
    parser.add_argument("--geom_root", type=str, default="../gaussian-splatting/output/test")

    # seed 只用于固定随机性；当前不再做 random_split
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

    dataset = build_one_dataset(
        data_id=args.data_id,
        crack_root=args.crack_root,
        mask_root=args.mask_root,
        geom_root=args.geom_root,
        mode=mode,
    )

    loader = DataLoader(
        dataset,
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
        save_dir = os.path.join(
            "./infer_results_v3",
            f"test_{args.data_id}",
            mode,
            ckpt_name
        )
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
