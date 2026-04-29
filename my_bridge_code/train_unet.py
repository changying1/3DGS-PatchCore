import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from gaussian_crack_seg_dataset import GaussianCrackSegDataset
from unet_model import UNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def compute_iou_from_logits(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def get_in_channels(mode):
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

def build_one_dataset(data_id, crack_root, mask_root, geom_root, mode):
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
        mode=mode
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rgb", choices=["rgb", "rgb_g0", "rgb_g0123", "rgb_g01234"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # 组级训练/验证划分
    parser.add_argument("--train_ids", type=str, default="1,2", help="训练集编号，例如 1,2")
    parser.add_argument("--val_id", type=str, default="0", help="验证集编号，例如 0")

    # 数据根目录
    parser.add_argument("--crack_root", type=str, default="./synthetic_crack_images")
    parser.add_argument("--mask_root", type=str, default="./synthetic_crack_masks")
    parser.add_argument("--geom_root", type=str, default="../gaussian-splatting/output/test")

    args = parser.parse_args()

    # ============= 训练参数 =============
    mode = args.mode
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_workers = 0
    base_channels = args.base_channels
    seed = args.seed

    train_ids = [x.strip() for x in args.train_ids.split(",") if x.strip()]
    val_id = args.val_id.strip()

    save_dir = f"./checkpoints_unet/fold_val{val_id}/{mode}/seed{seed}"
    # ===================================

    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_datasets = [
        build_one_dataset(
            data_id=train_id,
            crack_root=args.crack_root,
            mask_root=args.mask_root,
            geom_root=args.geom_root,
            mode=mode
        )
        for train_id in train_ids
    ]

    train_set = ConcatDataset(train_datasets)

    val_set = build_one_dataset(
        data_id=val_id,
        crack_root=args.crack_root,
        mask_root=args.mask_root,
        geom_root=args.geom_root,
        mode=mode
    )

    train_len = len(train_set)
    val_len = len(val_set)
    total_len = train_len + val_len

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = UNet(
        in_channels=get_in_channels(mode),
        out_channels=1,
        base_channels=base_channels
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = 1e9

    print(f"train_ids={train_ids}, val_id={val_id}")
    print(f"dataset total={total_len}, train={train_len}, val={val_len}")
    print(f"mode={mode}, in_channels={get_in_channels(mode)}")
    print(f"save_dir={save_dir}")

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        train_count = 0

        for x, mask, meta in train_loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            loss_bce = bce_loss(logits, mask)
            loss_dice = dice_loss_from_logits(logits, mask)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_iou_sum += compute_iou_from_logits(logits.detach(), mask) * bs
            train_count += bs

        train_loss = train_loss_sum / train_count
        train_iou = train_iou_sum / train_count

        # -------- val --------
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, mask, meta in val_loader:
                x = x.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                logits = model(x)
                loss_bce = bce_loss(logits, mask)
                loss_dice = dice_loss_from_logits(logits, mask)
                loss = loss_bce + loss_dice

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_iou_sum += compute_iou_from_logits(logits, mask) * bs
                val_count += bs

        val_loss = val_loss_sum / val_count
        val_iou = val_iou_sum / val_count

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}, train_iou={train_iou:.6f} | "
            f"val_loss={val_loss:.6f}, val_iou={val_iou:.6f}"
        )

        latest_path = os.path.join(save_dir, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mode": mode,
                "val_loss": val_loss,
            },
            latest_path
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mode": mode,
                    "val_loss": val_loss,
                },
                best_path
            )
            print(f"  [OK] best model saved to {best_path}")


if __name__ == "__main__":
    main()