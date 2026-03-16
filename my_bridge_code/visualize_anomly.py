import cv2
import os
import numpy as np

rgb_dir = "../gaussian-splatting/output/test/test_0"
anomaly_dir = "anomaly_maps/test_0"

save_dir = "visualizations/test_0"
os.makedirs(save_dir, exist_ok=True)

rgb_files = sorted([
    f for f in os.listdir(rgb_dir)
    if f.startswith("rgb_")
])

for i, rgb_name in enumerate(rgb_files):

    rgb_path = os.path.join(rgb_dir, rgb_name)
    anomaly_path = os.path.join(anomaly_dir, f"anomaly_{i}.png")

    rgb = cv2.imread(rgb_path)
    anomaly = cv2.imread(anomaly_path, 0)

    anomaly = cv2.resize(anomaly, (rgb.shape[1], rgb.shape[0]))

    anomaly = anomaly.astype(np.float32)
    anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-6)
    anomaly = (anomaly * 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    anomaly = anomaly * mask

    heatmap = cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(save_dir, f"vis_{i}.png")

    cv2.imwrite(save_path, overlay)

print("Visualization complete")