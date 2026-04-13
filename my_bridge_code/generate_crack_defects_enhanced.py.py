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
    任意 geometry 通道大于阈值，就认为该像素属于有效几何区域
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


def erode_mask(mask, k=19):
    """
    腐蚀一下 valid 区域，让裂隙更倾向于落在器物内部，减少贴边情况
    """
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded


def sample_start_point(valid_mask, max_tries=500):
    ys, xs = np.where(valid_mask > 0)
    if len(xs) == 0:
        return None

    idx = random.randint(0, len(xs) - 1)
    return int(xs[idx]), int(ys[idx])


def get_local_gray_mean(img, x, y, r=7):
    h, w = img.shape[:2]
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return 128.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def darken_along_mask(img, line_mask, strength=55, blur_ksize=9):
    """
    对 line_mask 区域做“局部变暗”，而不是固定画纯黑色
    """
    out = img.astype(np.float32).copy()

    soft = cv2.GaussianBlur(line_mask, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0
    soft = soft[..., None]  # [H,W,1]

    out = out - strength * soft
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def draw_polyline(mask, pts, thickness):
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts_np], False, 255, thickness=thickness, lineType=cv2.LINE_AA)


def grow_crack_points(valid_mask, start, main_angle, length_steps):
    """
    生长一条更自然的裂隙折线
    """
    h, w = valid_mask.shape
    x, y = start

    pts = [(x, y)]
    cur_angle = main_angle

    for _ in range(length_steps):
        # 角度慢变，避免太僵硬
        cur_angle += random.uniform(-0.10, 0.10)

        step_len = random.choice([1, 1, 2, 2, 3])
        dx = int(round(np.cos(cur_angle) * step_len))
        dy = int(round(np.sin(cur_angle) * step_len))

        if dx == 0 and dy == 0:
            continue

        nx = x + dx
        ny = y + dy

        if not (0 <= nx < w and 0 <= ny < h):
            break
        if valid_mask[ny, nx] == 0:
            break

        # 偶尔出现微小断续，但整体仍连续
        if random.random() < 0.03:
            x, y = nx, ny
            continue

        x, y = nx, ny
        pts.append((x, y))

    return pts


def draw_variable_width_crack(img, pts, crack_mask, halo_mask):
    """
    宽度略有变化的主裂隙：
    - crack_mask: 核心裂隙
    - halo_mask: 周围弱暗带
    """
    if len(pts) < 2:
        return

    # 分段画，让宽度有变化
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]

        core_t = random.randint(2, 4)         # 主体更明显一点
        halo_t = core_t + random.randint(2, 4)

        cv2.line(halo_mask, p1, p2, 255, halo_t, lineType=cv2.LINE_AA)
        cv2.line(crack_mask, p1, p2, 255, core_t, lineType=cv2.LINE_AA)

    # 局部节点再补一点不规则性
    for p in pts[::max(1, len(pts)//12)]:
        core_r = random.randint(1, 2)
        halo_r = core_r + random.randint(2, 3)
        cv2.circle(halo_mask, p, halo_r, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(crack_mask, p, core_r, 255, -1, lineType=cv2.LINE_AA)


def generate_branch(valid_mask, pts, crack_mask, halo_mask):
    if len(pts) < 40:
        return

    if random.random() > 0.7:
        return

    branch_start = pts[random.randint(len(pts)//4, 3*len(pts)//4)]
    bx, by = branch_start

    # 用附近走势估计方向
    ref_idx = random.randint(max(1, len(pts)//4), min(len(pts)-2, 3*len(pts)//4))
    x1, y1 = pts[ref_idx - 1]
    x2, y2 = pts[ref_idx + 1]
    base_angle = np.arctan2(y2 - y1, x2 - x1)

    branch_angle = base_angle + random.uniform(-1.1, 1.1)
    branch_len = random.randint(20, 60)

    bpts = grow_crack_points(valid_mask, (bx, by), branch_angle, branch_len)
    if len(bpts) < 12:
        return

    draw_variable_width_crack(None, bpts, crack_mask, halo_mask)


def draw_random_crack_in_valid_region(img, valid_mask, max_attempts=30):
    """
    在有效区域内部生成更自然、更容易被学到的裂隙
    """
    h, w = img.shape[:2]
    out_img = img.copy()

    crack_mask = np.zeros((h, w), dtype=np.uint8)   # 核心裂隙
    halo_mask = np.zeros((h, w), dtype=np.uint8)    # 周围弱暗带

    if np.sum(valid_mask > 0) < 500:
        print("[WARN] valid region too small, skip crack generation.")
        return out_img, crack_mask

    # 更偏向内部区域，减少贴边裂隙
    inner_valid = erode_mask(valid_mask, k=19)
    if np.sum(inner_valid > 0) < 200:
        inner_valid = valid_mask.copy()

    success = False

    for _ in range(max_attempts):
        start = sample_start_point(inner_valid)
        if start is None:
            break

        x, y = start
        main_angle = random.uniform(0, np.pi)

        # 更合适的长度
        length_steps = random.randint(80, 180)

        pts = grow_crack_points(valid_mask, (x, y), main_angle, length_steps)

        if len(pts) < 35:
            continue

        draw_variable_width_crack(out_img, pts, crack_mask, halo_mask)
        generate_branch(valid_mask, pts, crack_mask, halo_mask)

        # 少量样本再加第二条短裂隙
        if random.random() < 0.25:
            start2 = sample_start_point(inner_valid)
            if start2 is not None:
                angle2 = random.uniform(0, np.pi)
                pts2 = grow_crack_points(valid_mask, start2, angle2, random.randint(40, 90))
                if len(pts2) >= 20:
                    draw_variable_width_crack(out_img, pts2, crack_mask, halo_mask)

        success = True
        break

    if not success:
        print("[WARN] failed to generate a valid crack, return empty mask.")
        return out_img, crack_mask

    # halo 只保留 crack 外围区域，避免把主体一起冲淡
    halo_only = cv2.subtract(halo_mask, crack_mask)

    # 根据局部亮度决定暗化强度，让裂隙更自然
    ys, xs = np.where(crack_mask > 0)
    if len(xs) > 0:
        choose_idx = random.randint(0, len(xs) - 1)
        lx, ly = int(xs[choose_idx]), int(ys[choose_idx])
        local_mean = get_local_gray_mean(img, lx, ly, r=9)

        # 亮区域可以暗得更明显，暗区域则收一点
        core_strength = int(np.clip(0.30 * local_mean + 20, 35, 85))
        halo_strength = int(np.clip(0.12 * local_mean, 10, 30))
    else:
        core_strength = 60
        halo_strength = 18

    out_img = darken_along_mask(out_img, halo_only, strength=halo_strength, blur_ksize=11)
    out_img = darken_along_mask(out_img, crack_mask, strength=core_strength, blur_ksize=5)

    # 让 mask 更干净，主体清晰
    crack_mask = (crack_mask > 0).astype(np.uint8) * 255

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

    # 中心裁剪区域
    crop_mask, left, top, crop_size = get_center_crop_mask(h, w, crop_ratio=0.7)

    # 最终有效区域 = geometry 有效区域 ∩ 中心裁剪区域
    valid_mask = np.logical_and(geom_valid > 0, crop_mask > 0).astype(np.uint8) * 255
    valid_mask = clean_valid_mask(valid_mask)

    # 生成裂隙
    img_crack, crack_mask = draw_random_crack_in_valid_region(img, valid_mask)

    img_save = os.path.join(save_img_dir, f"crack_{idx_file}.png")
    mask_save = os.path.join(save_mask_dir, f"mask_{idx_file}.png")

    cv2.imwrite(img_save, img_crack)
    cv2.imwrite(mask_save, crack_mask)

    print(f"[OK] {rgb_name} -> crack_{idx_file}.png / mask_{idx_file}.png")

print("Crack dataset generated with enhanced geometry-aware valid region.")