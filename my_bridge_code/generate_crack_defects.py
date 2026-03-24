import os
import cv2
import numpy as np
import random

# -----------------------
# config
# -----------------------

rgb_dir = "../gaussian-splatting/output/test/test_0"

save_img_dir = "synthetic_crack_images/test_0"
save_mask_dir = "synthetic_crack_masks/test_0"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

rgb_files = sorted([
    f for f in os.listdir(rgb_dir)
    if f.startswith("rgb_")
])


# -----------------------
# helper
# -----------------------

def get_center_crop_mask(h, w, crop_ratio=0.7):
    """
    与 GaussianPatchCoreDataset 里的中心裁剪保持一致
    """
    crop_size = int(min(w, h) * crop_ratio)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2

    crop_mask = np.zeros((h, w), dtype=np.uint8)
    crop_mask[top:top + crop_size, left:left + crop_size] = 255
    return crop_mask, left, top, crop_size


def load_geometry_valid_mask(data_dir, idx, h, w, thresh=10):
    """
    读取 geometry_c0 ~ geometry_c4，生成 geometry 有效区域 mask
    规则：
    - 任意一个 geometry 通道大于阈值，就认为该像素属于有效几何区域
    """
    geom_valid = np.zeros((h, w), dtype=np.uint8)

    for c in range(5):
        geom_path = os.path.join(data_dir, f"geometry_c{c}_{idx}.png")
        if not os.path.exists(geom_path):
            print(f"[WARN] geometry file not found: {geom_path}")
            continue

        geom = cv2.imread(geom_path, 0)
        if geom is None:
            print(f"[WARN] failed to read: {geom_path}")
            continue

        if geom.shape[:2] != (h, w):
            geom = cv2.resize(geom, (w, h), interpolation=cv2.INTER_NEAREST)

        geom_valid = np.logical_or(geom_valid > 0, geom > thresh).astype(np.uint8) * 255

    return geom_valid


def clean_valid_mask(valid_mask):
    """
    做一点形态学清理，减少碎噪声
    """
    kernel = np.ones((5, 5), np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    return valid_mask


def sample_start_point(valid_mask, margin=10, max_tries=500):
    """
    在有效区域里随机选一个起点
    """
    ys, xs = np.where(valid_mask > 0)
    if len(xs) == 0:
        return None

    h, w = valid_mask.shape
    valid_idx = []

    for k in range(len(xs)):
        x = xs[k]
        y = ys[k]
        if margin <= x < w - margin and margin <= y < h - margin:
            valid_idx.append(k)

    if len(valid_idx) == 0:
        return None

    choose_k = random.choice(valid_idx)
    return int(xs[choose_k]), int(ys[choose_k])


def draw_random_crack_in_valid_region(img, valid_mask, max_attempts=20):
    """
    只在 valid_mask 区域内画裂纹
    """
    h, w = img.shape[:2]
    out_img = img.copy()
    crack_mask = np.zeros((h, w), dtype=np.uint8)

    # 如果有效区域太少，直接返回空 mask
    if np.sum(valid_mask > 0) < 200:
        print("[WARN] valid region too small, skip crack generation.")
        return out_img, crack_mask

    success = False

    for _ in range(max_attempts):
        start = sample_start_point(valid_mask, margin=10)
        if start is None:
            break

        x, y = start

        # 裂纹参数
        length = random.randint(50, 140)
        thickness = random.randint(1, 3)
        angle = random.uniform(0, np.pi)

        pts = []
        cur_x, cur_y = x, y

        for i in range(length):
            # 让裂纹有一点弯曲抖动
            local_angle = angle + random.uniform(-0.08, 0.08)

            dx = int(np.cos(local_angle) * random.randint(1, 2))
            dy = int(np.sin(local_angle) * random.randint(1, 2))

            cur_x = cur_x + dx
            cur_y = cur_y + dy

            if not (0 <= cur_x < w and 0 <= cur_y < h):
                break

            # 只允许在有效几何区域里继续生长
            if valid_mask[cur_y, cur_x] == 0:
                break

            pts.append((cur_x, cur_y))

        # 点太少，说明这条裂纹无效，重新采样
        if len(pts) < 25:
            continue

        # 画主裂纹
        for p in pts:
            cv2.circle(out_img, p, thickness, (20, 20, 20), -1)
            cv2.circle(crack_mask, p, thickness, 255, -1)

        # 偶尔加一个细小分叉，增加真实性
        if len(pts) > 40 and random.random() < 0.5:
            branch_start = pts[random.randint(len(pts)//3, 2*len(pts)//3)]
            bx, by = branch_start
            branch_angle = angle + random.uniform(-0.9, 0.9)
            branch_len = random.randint(15, 40)

            cur_bx, cur_by = bx, by
            for _ in range(branch_len):
                dx = int(np.cos(branch_angle) * 1)
                dy = int(np.sin(branch_angle) * 1)
                cur_bx += dx + random.randint(-1, 1)
                cur_by += dy + random.randint(-1, 1)

                if not (0 <= cur_bx < w and 0 <= cur_by < h):
                    break
                if valid_mask[cur_by, cur_bx] == 0:
                    break

                cv2.circle(out_img, (cur_bx, cur_by), max(1, thickness - 1), (20, 20, 20), -1)
                cv2.circle(crack_mask, (cur_bx, cur_by), max(1, thickness - 1), 255, -1)

        success = True
        break

    if not success:
        print("[WARN] failed to generate a valid crack, return empty mask.")

    return out_img, crack_mask


# -----------------------
# generate
# -----------------------

for idx_file, rgb_name in enumerate(rgb_files):
    rgb_path = os.path.join(rgb_dir, rgb_name)
    img = cv2.imread(rgb_path)

    if img is None:
        print(f"[WARN] failed to read rgb: {rgb_path}")
        continue

    h, w = img.shape[:2]
    idx = rgb_name.split("_")[1].split(".")[0]

    # geometry 有效区域
    geom_valid = load_geometry_valid_mask(rgb_dir, idx, h, w, thresh=10)
    geom_valid = clean_valid_mask(geom_valid)

    # 中心裁剪区域（与 dataset 保持一致）
    crop_mask, left, top, crop_size = get_center_crop_mask(h, w, crop_ratio=0.7)

    # 最终有效区域 = geometry 有效区域 ∩ 中心裁剪区域
    valid_mask = np.logical_and(geom_valid > 0, crop_mask > 0).astype(np.uint8) * 255
    valid_mask = clean_valid_mask(valid_mask)

    # 生成裂纹
    img_crack, crack_mask = draw_random_crack_in_valid_region(img, valid_mask)

    img_save = os.path.join(save_img_dir, f"crack_{idx_file}.png")
    mask_save = os.path.join(save_mask_dir, f"mask_{idx_file}.png")

    cv2.imwrite(img_save, img_crack)
    cv2.imwrite(mask_save, crack_mask)

    print(f"[OK] {rgb_name} -> crack_{idx_file}.png / mask_{idx_file}.png")

print("Crack dataset generated with geometry-aware valid region.")