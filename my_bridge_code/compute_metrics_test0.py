import os
import numpy as np
import pandas as pd
from PIL import Image

# ===========================
# 配置
# ===========================
BASE_DIR = "./infer_results_v3/test_0"  # test_0 根目录
MODES = ["rgb", "g0", "g0123", "g01234", "rgb_g0", "rgb_g0123", "rgb_g01234"]
OUTPUT_CSV = "./metrics_summary_test0.csv"
THRESHOLD = 0.5  # 二值化阈值

# ===========================
# 指标函数
# ===========================
def compute_metrics(pred, gt):
    pred_bin = (pred > THRESHOLD).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)

    TP = np.sum(pred_bin * gt_bin)
    FP = np.sum(pred_bin * (1 - gt_bin))
    FN = np.sum((1 - pred_bin) * gt_bin)

    iou = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = 2 * TP / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)

    return iou, precision, recall, f1, dice

# ===========================
# 扫描样本
# ===========================
sample_ids = set()
for mode in MODES:
    mode_dir = os.path.join(BASE_DIR, mode)
    if not os.path.exists(mode_dir):
        continue
    for d in os.listdir(mode_dir):
        if os.path.isdir(os.path.join(mode_dir, d)):
            sample_ids.add(d)

sample_ids = sorted(list(sample_ids))
print(f"[INFO] Found {len(sample_ids)} samples.")

# ===========================
# 主循环
# ===========================
records = []

for sample_id in sample_ids:
    row = {"sample_id": sample_id}
    rgb_iou = None

    for mode in MODES:
        mode_dir = os.path.join(BASE_DIR, mode, sample_id)
        gt_path = os.path.join(mode_dir, "gt_mask.png")
        pred_path = os.path.join(mode_dir, "pred_mask.png")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"[WARN] Missing gt or pred for {mode}/{sample_id}, skip.")
            continue

        gt = np.array(Image.open(gt_path).convert("L")) / 255.0
        pred = np.array(Image.open(pred_path).convert("L")) / 255.0

        iou, precision, recall, f1, dice = compute_metrics(pred, gt)

        row[f"{mode}_iou"] = iou
        row[f"{mode}_precision"] = precision
        row[f"{mode}_recall"] = recall
        row[f"{mode}_f1"] = f1
        row[f"{mode}_dice"] = dice

        if mode == "rgb":
            rgb_iou = iou

    # 计算增益：相对 RGB-only
    for mode in MODES:
        if mode == "rgb":
            continue
        key = f"{mode}_gain_vs_rgb"
        if rgb_iou is not None and f"{mode}_iou" in row:
            row[key] = row[f"{mode}_iou"] - rgb_iou
        else:
            row[key] = np.nan

    records.append(row)

# ===========================
# 保存 CSV
# ===========================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Metrics saved to {OUTPUT_CSV}")
