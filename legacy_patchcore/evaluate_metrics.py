import os
import re
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


# -----------------------
# config
# -----------------------

anomaly_dir = "anomaly_maps/test_0"
mask_dir = "synthetic_crack_masks/test_0"
geometry_dir = "../gaussian-splatting/output/test/test_0"
rgb_dir = "../gaussian-splatting/output/test/test_0"

crop_ratio = 0.7
geom_thresh = 10

anomaly_files = sorted([
    f for f in os.listdir(anomaly_dir)
    if f.startswith("anomaly_") and f.endswith(".png")
], key=lambda x: int(re.findall(r"\d+", x)[0]))

mask_files = sorted([
    f for f in os.listdir(mask_dir)
    if f.startswith("mask_") and f.endswith(".png")
], key=lambda x: int(re.findall(r"\d+", x)[0]))

rgb_files = sorted([
    f for f in os.listdir(rgb_dir)
    if f.startswith("rgb_") and f.endswith(".png")
], key=lambda x: int(re.findall(r"\d+", x)[0]))

all_scores = []
all_labels = []


# -----------------------
# helper
# -----------------------

def extract_numeric_idx(filename, prefix):
    m = re.match(rf"^{prefix}_(\d+)\.png$", filename)
    if m is None:
        return None
    return int(m.group(1))


def get_center_crop_box(h, w, crop_ratio=0.7):
    crop_size = int(min(w, h) * crop_ratio)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return left, top, crop_size


def load_geometry_valid_mask(data_dir, real_idx, h, w, thresh=10):
    geom_valid = np.zeros((h, w), dtype=np.uint8)
    found_any = False

    for c in range(5):
        geom_path = os.path.join(data_dir, f"geometry_c{c}_{real_idx}.png")
        if not os.path.exists(geom_path):
            print(f"[WARN] geometry file not found: {geom_path}")
            continue

        geom = cv2.imread(geom_path, 0)
        if geom is None:
            print(f"[WARN] failed to read: {geom_path}")
            continue

        found_any = True

        if geom.shape[:2] != (h, w):
            geom = cv2.resize(geom, (w, h), interpolation=cv2.INTER_NEAREST)

        geom_valid = np.logical_or(geom_valid > 0, geom > thresh).astype(np.uint8) * 255

    return geom_valid, found_any


def clean_valid_mask(valid_mask):
    kernel = np.ones((5, 5), np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    return valid_mask


# -----------------------
# build sequential-to-real mapping
# -----------------------

# 第 i 个 anomaly / mask -> 第 i 个 rgb 的真实编号
real_idx_list = []
for rgb_name in rgb_files:
    idx = extract_numeric_idx(rgb_name, "rgb")
    if idx is not None:
        real_idx_list.append(idx)

num_samples = min(len(anomaly_files), len(mask_files), len(real_idx_list))

print(f"Found {len(anomaly_files)} anomaly files")
print(f"Found {len(mask_files)} mask files")
print(f"Found {len(real_idx_list)} rgb files")
print(f"Using {num_samples} aligned samples")


# -----------------------
# evaluate
# -----------------------

for i in range(num_samples):
    anomaly_path = os.path.join(anomaly_dir, anomaly_files[i])
    mask_path = os.path.join(mask_dir, mask_files[i])
    real_idx = real_idx_list[i]

    print(f"{anomaly_path} vs {mask_path} -> real rgb/geometry idx = {real_idx}")

    anomaly = cv2.imread(anomaly_path, 0)
    mask = cv2.imread(mask_path, 0)

    if anomaly is None:
        print(f"[WARN] failed to read anomaly map: {anomaly_path}")
        continue
    if mask is None:
        print(f"[WARN] failed to read mask: {mask_path}")
        continue

    h0, w0 = mask.shape[:2]

    geom_valid, found_any = load_geometry_valid_mask(
        geometry_dir, real_idx, h0, w0, thresh=geom_thresh
    )
    if not found_any:
        print(f"[WARN] no geometry found for real idx {real_idx}, skip.")
        continue

    geom_valid = clean_valid_mask(geom_valid)

    left, top, crop_size = get_center_crop_box(h0, w0, crop_ratio=crop_ratio)
    crop_mask = np.zeros((h0, w0), dtype=np.uint8)
    crop_mask[top:top + crop_size, left:left + crop_size] = 255

    eval_valid = np.logical_and(geom_valid > 0, crop_mask > 0).astype(np.uint8)

    mask_crop = mask[top:top + crop_size, left:left + crop_size]
    valid_crop = eval_valid[top:top + crop_size, left:left + crop_size]

    mask_crop = cv2.resize(
        mask_crop,
        (anomaly.shape[1], anomaly.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    valid_crop = cv2.resize(
        valid_crop.astype(np.uint8),
        (anomaly.shape[1], anomaly.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    anomaly = anomaly.astype(np.float32) / 255.0
    mask_bin = (mask_crop > 128).astype(np.uint8)
    valid_bin = (valid_crop > 0).astype(np.uint8)

    anomaly_valid = anomaly[valid_bin == 1]
    mask_valid = mask_bin[valid_bin == 1]

    if len(anomaly_valid) == 0:
        print(f"[WARN] no valid eval pixels for sample {i} (real idx {real_idx}), skip.")
        continue

    all_scores.extend(anomaly_valid.tolist())
    all_labels.extend(mask_valid.tolist())

all_scores = np.array(all_scores, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.uint8)

if len(all_scores) == 0:
    raise RuntimeError("No valid pixels found for evaluation.")

if len(np.unique(all_labels)) < 2:
    raise RuntimeError("Ground-truth labels in valid region contain only one class, AUROC cannot be computed.")

auroc = roc_auc_score(all_labels, all_scores)

precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
f1_list = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = np.argmax(f1_list)
best_f1 = f1_list[best_idx]
best_thr = thresholds[max(best_idx - 1, 0)] if len(thresholds) > 0 else 0.5
aupr = auc(recall, precision)

print("Pixel AUROC:", auroc)
print("Pixel AUPR:", aupr)
print("Best Pixel F1:", best_f1)
print("Best threshold:", best_thr)
print("Pixel AUROC (geometry-aware valid region):", auroc)
print("Best Pixel F1 (geometry-aware valid region):", best_f1)
print("Num valid pixels:", len(all_scores))
print("Num positive pixels:", int(all_labels.sum()))
print("Num negative pixels:", int((all_labels == 0).sum()))