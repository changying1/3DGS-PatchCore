import os
import csv
import argparse
import random
import numpy as np

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


def build_one_dataset(data_id, crack_root, mask_root, geom_root, mode):
    crack_dir = os.path.join(crack_root, f"test_{data_id}")
    mask_dir = os.path.join(mask_root, f"test_{data_id}")
    geom_dir = os.path.join(geom_root, f"test_{data_id}")

    return GaussianCrackSegDataset(
        crack_dir=crack_dir,
        mask_dir=mask_dir,
        geom_dir=geom_dir,
        mode=mode,
    )


def evaluate(model, loader, device):
    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    iou_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, mask, meta in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(x)

            loss_bce = bce_loss(logits, mask)
            loss_dice = dice_loss_from_logits(logits, mask)
            loss = loss_bce + loss_dice

            bs = x.size(0)
            loss_sum += loss.item() * bs
            iou_sum += compute_iou_from_logits(logits, mask) * bs
            count += bs

    return loss_sum / count, iou_sum / count


def infer_train_ids_from_val_id(val_id):
    all_ids = ["0", "1", "2"]
    return [x for x in all_ids if x != str(val_id)]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_val", type=str, default="0")
    parser.add_argument("--modes", type=str, default="rgb_g0,rgb_g0123,rgb_g01234")
    parser.add_argument("--seeds", type=str, default="1,2,3")

    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints_unet")
    parser.add_argument("--crack_root", type=str, default="./synthetic_crack_images")
    parser.add_argument("--mask_root", type=str, default="./synthetic_crack_masks")
    parser.add_argument("--geom_root", type=str, default="../gaussian-splatting/output/test")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--save_csv", type=str, default=None)

    args = parser.parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    val_id = args.fold_val
    train_ids = infer_train_ids_from_val_id(val_id)

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]

    if args.save_csv is None:
        args.save_csv = os.path.join(
            args.checkpoint_root,
            f"fold_val{val_id}",
            "rescued_eval_results.csv"
        )

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    rows = []

    for mode in modes:
        print(f"\n========== mode: {mode} ==========")

        train_datasets = [
            build_one_dataset(
                data_id=train_id,
                crack_root=args.crack_root,
                mask_root=args.mask_root,
                geom_root=args.geom_root,
                mode=mode,
            )
            for train_id in train_ids
        ]

        train_set = ConcatDataset(train_datasets)

        val_set = build_one_dataset(
            data_id=val_id,
            crack_root=args.crack_root,
            mask_root=args.mask_root,
            geom_root=args.geom_root,
            mode=mode,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for seed in seeds:
            ckpt_dir = os.path.join(
                args.checkpoint_root,
                f"fold_val{val_id}",
                mode,
                f"seed{seed}"
            )

            for ckpt_name in ["best.pth", "latest.pth"]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)

                if not os.path.exists(ckpt_path):
                    print(f"[SKIP] missing: {ckpt_path}")
                    continue

                print(f"[EVAL] {ckpt_path}")

                ckpt = torch.load(ckpt_path, map_location=device)

                model = UNet(
                    in_channels=get_in_channels(mode),
                    out_channels=1,
                    base_channels=args.base_channels,
                ).to(device)

                model.load_state_dict(ckpt["model_state_dict"], strict=True)

                train_loss, train_iou = evaluate(model, train_loader, device)
                val_loss, val_iou = evaluate(model, val_loader, device)

                row = {
                    "fold_val": val_id,
                    "train_ids": ",".join(train_ids),
                    "val_id": val_id,
                    "mode": mode,
                    "seed": seed,
                    "checkpoint": ckpt_name,
                    "checkpoint_epoch": ckpt.get("epoch", ""),
                    "checkpoint_saved_val_loss": ckpt.get("val_loss", ""),
                    "eval_train_loss": train_loss,
                    "eval_train_iou": train_iou,
                    "eval_val_loss": val_loss,
                    "eval_val_iou": val_iou,
                }

                rows.append(row)

                print(
                    f"  epoch={row['checkpoint_epoch']} | "
                    f"train_loss={train_loss:.6f}, train_iou={train_iou:.6f} | "
                    f"val_loss={val_loss:.6f}, val_iou={val_iou:.6f}"
                )

    fieldnames = [
        "fold_val",
        "train_ids",
        "val_id",
        "mode",
        "seed",
        "checkpoint",
        "checkpoint_epoch",
        "checkpoint_saved_val_loss",
        "eval_train_loss",
        "eval_train_iou",
        "eval_val_loss",
        "eval_val_iou",
    ]

    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[OK] saved rescued eval csv to: {args.save_csv}")


if __name__ == "__main__":
    main()