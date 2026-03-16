import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

anomaly_dir = "anomaly_maps"
mask_dir = "synthetic_crack_masks"

anomaly_files = sorted(os.listdir(anomaly_dir))
mask_files = sorted(os.listdir(mask_dir))

all_scores = []
all_labels = []

num_samples = len(anomaly_files)

for i in range(num_samples):

    anomaly_path = os.path.join(anomaly_dir, f"anomaly_{i}.png")
    mask_path = os.path.join(mask_dir, f"mask_{i}.png")
    print(anomaly_path, "vs", mask_path)

    anomaly = cv2.imread(anomaly_path, 0)
    mask = cv2.imread(mask_path, 0)

    anomaly = anomaly.astype(np.float32) / 255.0
    mask = (mask > 128).astype(np.uint8)

    mask = cv2.resize(mask, (anomaly.shape[1], anomaly.shape[0]))

    anomaly = anomaly.flatten()
    mask = mask.flatten()

    all_scores.extend(anomaly)
    all_labels.extend(mask)

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# Pixel-level AUROC
auroc = roc_auc_score(all_labels, all_scores)

# Pixel-level F1
pred = (all_scores > 0.5).astype(np.uint8)
f1 = f1_score(all_labels, pred)

print("Pixel AUROC:", auroc)
print("Pixel F1:", f1)