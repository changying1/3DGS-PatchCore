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


ALL_MODES = [
    "rgb",
    "g0",
    "g0123",
    "g01234",
    "rgb_g0",
    "rgb_g0123",
    "rgb_g01234",
]


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


def geometry_to_uint8(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    g_min, g_max = np.percentile(g, 1), np.percentile(g, 99)
    if g_max - g_min < 1e-8:
        return np.zeros_like(g, dtype=np.uint8)
    return (np.clip((g - g_min) / (g_max - g_min), 0, 1) * 255).astype(np.uint8)


# ===========================
# 修改后的输入可视化函数
# ===========================
def save_input_visualization(x: torch.Tensor, mode: str, sample_dir: str):
    x_np = x.detach().cpu().numpy()

    if mode == "rgb":
        rgb = np.clip(x_np[:3].transpose(1, 2, 0), 0.0, 1.0)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(sample_dir, "input_rgb.png"))
    else:
        # 单通道 / 多通道 geom 模式只保存第一个通道
        first_geom = geometry_to_uint8(x_np[0])
        Image.fromarray(first_geom).save(os.path.join(sample_dir, "input_rgb.png"))


def save_mask_tensor(mask: torch.Tensor, save_path: str):
    if mask.ndim == 3:
        mask = mask[0]
    mask_np = mask.detach().cpu().numpy()
    Image.fromarray(to_uint8_gray(mask_np)).save(save_path)


def load_crack_rgb(meta, b: int, fallback: torch.Tensor) -> np.ndarray:
    crack_paths = meta.get("crack_path")
    if crack_paths is not None:
        crack_path = crack_paths[b]
        if os.path.exists(crack_path):
            return np.array(Image.open(crack_path).convert("RGB"), dtype=np.float32) / 255.0

    x_np = fallback.detach().cpu().numpy()
    if x_np.shape[0] >= 3:
        return np.clip(x_np[:3].transpose(1, 2, 0), 0.0, 1.0)

    gray = geometry_to_uint8(x_np[0]).astype(np.float32) / 255.0
    return np.repeat(gray[..., None], 3, axis=2)


def make_overlay(base_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = np.clip(base_rgb, 0.0, 1.0)
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0

    mask_3 = pred_mask[..., None].astype(np.float32)
    overlay = overlay * (1 - alpha * mask_3) + red * (alpha * mask_3)
    overlay = np.clip(overlay, 0.0, 1.0)

    return Image.fromarray((overlay * 255).astype(np.uint8))


# ===========================
# 数据集 / 推理函数
# ===========================
def build_one_dataset(data_id: str, crack_root: str, mask_root: str, geom_root: str, mode: str):
    crack_dir = os.path.join(crack_root, f"test_{data_id}")
    mask_dir = os.path.join(mask_root, f"test_{data_id}")
    geom_dir = os.path.join(geom_root, f"test_{data_id}")

    print(f"[DATA] test_{data_id}")
    print(f"  crack_dir = {crack_dir}")
    print(f"  mask_dir  = {mask_dir}")
    print(f"  geom_dir  = {geom_dir}")

    return GaussianCrackSegDataset(crack_dir, mask_dir, geom_dir, mode)


def infer_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    threshold: float,
    mode: str,
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

                save_input_visualization(x_b, mode, sample_dir)
                save_mask_tensor(mask_b, os.path.join(sample_dir, "gt_mask.png"))
                Image.fromarray(to_uint8_gray(prob_b)).save(os.path.join(sample_dir, "pred_prob.png"))
                Image.fromarray((pred_b * 255).astype(np.uint8)).save(os.path.join(sample_dir, "pred_mask.png"))

                base_rgb = load_crack_rgb(meta, b, x_b)
                overlay = make_overlay(base_rgb, pred_b.astype(np.uint8), alpha=0.45)
                overlay.save(os.path.join(sample_dir, "overlay.png"))

                with open(os.path.join(sample_dir, "meta.txt"), "w", encoding="utf-8") as f:
                    f.write(f"idx: {idx}\n")
                    f.write(f"crack_name: {crack_name}\n")
                    f.write(f"mode: {mode}\n")
                    f.write(f"threshold: {threshold}\n")

                count += 1
                if max_samples is not None and count >= max_samples:
                    print(f"[OK] reach max_samples={max_samples}, stop inference.")
                    return

    print(f"[OK] inference finished, total saved samples: {count}")


def run_inference(args, mode: str, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_mode = ckpt.get("mode", None)
    if ckpt_mode is not None and ckpt_mode != mode:
        print(f"[WARN] checkpoint mode={ckpt_mode}, command mode={mode}; use command mode.")

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
    print(f"[OK] loaded checkpoint: {ckpt_path}")
    if "epoch" in ckpt:
        print(f"[OK] checkpoint epoch: {ckpt['epoch']}")

    if args.save_dir is None:
        if args.ablation:
            save_dir = os.path.join("./infer_results_v3", f"test_{args.data_id}", mode)
        else:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            save_dir = os.path.join("./infer_results_v3", f"test_{args.data_id}", mode, ckpt_name)
    else:
        save_dir = os.path.join(args.save_dir, mode) if args.ablation else args.save_dir

    ensure_dir(save_dir)
    print(f"save_dir: {save_dir}")

    infer_one_epoch(
        model=model,
        loader=loader,
        device=device,
        save_dir=save_dir,
        threshold=args.threshold,
        mode=mode,
        max_samples=args.max_samples,
    )


def parse_modes(modes: str):
    parsed = [m.strip() for m in modes.split(",") if m.strip()]
    bad_modes = [m for m in parsed if m not in ALL_MODES]
    if bad_modes:
        raise ValueError(f"Unsupported modes: {bad_modes}")
    return parsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint for single-mode inference")
    parser.add_argument("--mode", type=str, default=None, choices=ALL_MODES, help="If not given, try reading mode from checkpoint")
    parser.add_argument("--ablation", action="store_true", help="Run multiple ablation modes from a checkpoint root")
    parser.add_argument("--modes", type=str, default="rgb,g0,g0123,g01234,rgb_g0,rgb_g0123,rgb_g01234")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_unet_v3")
    parser.add_argument("--val_id", type=str, default="0")
    parser.add_argument("--ckpt_name", type=str, default="best_iou.pth")

    parser.add_argument("--data_id", type=str, default="0")
    parser.add_argument("--crack_root", type=str, default="./synthetic_crack_images_v2")
    parser.add_argument("--mask_root", type=str, default="./synthetic_crack_masks_v2")
    parser.add_argument("--geom_root", type=str, default="../gaussian-splatting/output/test")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=16)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.ablation:
        for mode in parse_modes(args.modes):
            ckpt_path = os.path.join(
                args.ckpt_root,
                f"fold_val{args.val_id}",
                mode,
                f"seed{args.seed}",
                args.ckpt_name,
            )
            print(f"\n========== ablation mode: {mode} ==========")
            run_inference(args, mode, ckpt_path, device)
        return

    if args.ckpt is None:
        raise ValueError("--ckpt is required unless --ablation is used.")

    ckpt = torch.load(args.ckpt, map_location=device)
    mode = args.mode if args.mode is not None else ckpt.get("mode", None)
    if mode is None:
        raise ValueError("Mode is not provided and not found in checkpoint.")

    run_inference(args, mode, args.ckpt, device)


if __name__ == "__main__":
    main()
