import os
import re
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def robust_normalize(x, mask=None, low_q=1.0, high_q=99.0, eps=1e-8):
    """
    Robust min-max normalize to [0,1] using percentiles.
    """
    x = x.astype(np.float32)
    if mask is not None:
        vals = x[mask > 0]
    else:
        vals = x.reshape(-1)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = np.percentile(vals, low_q)
    hi = np.percentile(vals, high_q)

    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo + eps)
    y = np.clip(y, 0.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def list_sample_ids(group_dir):
    """
    Discover sample ids by intersecting:
      rgb_{id}.png and geometry_map_{id}.npz
    """
    group_dir = Path(group_dir)
    rgb_ids = set()
    geo_ids = set()

    rgb_pat = re.compile(r"rgb_(\d+)\.png$")
    geo_pat = re.compile(r"geometry_map_(\d+)\.npz$")

    for f in group_dir.iterdir():
        if f.is_file():
            m = rgb_pat.match(f.name)
            if m:
                rgb_ids.add(int(m.group(1)))

            m = geo_pat.match(f.name)
            if m:
                geo_ids.add(int(m.group(1)))

    ids = sorted(list(rgb_ids.intersection(geo_ids)))
    return ids


def load_rgb(group_dir, idx):
    p = Path(group_dir) / f"rgb_{idx}.png"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read RGB image: {p}")
    return img


def load_geometry(group_dir, idx):
    p = Path(group_dir) / f"geometry_map_{idx}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Cannot read geometry map: {p}")

    data = np.load(str(p))
    if "geometry" not in data:
        raise KeyError(f"'geometry' key not found in {p}")

    geo = data["geometry"].astype(np.float32)  # [C,H,W]
    if geo.ndim != 3:
        raise ValueError(f"geometry should be [C,H,W], got shape={geo.shape}")
    return geo


def build_valid_mask(geo):
    """
    Build a valid region mask from geometry.
    Very conservative: finite + not-all-zero.
    """
    finite = np.all(np.isfinite(geo), axis=0)
    nonzero = np.any(np.abs(geo) > 1e-8, axis=0)
    valid = finite & nonzero

    # Fallback if too sparse
    if valid.mean() < 0.05:
        valid = finite.copy()

    valid = valid.astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)

    return valid.astype(np.uint8)


def build_geometry_score_map(geo, valid_mask):
    """
    Build a geometry-aware score map for true crack generation.
    Recommended channels:
      G0 = geo[0]
      G3 = geo[3] (gradient magnitude)
      G4 = geo[4] (laplacian / curvature-like)
    """
    C, H, W = geo.shape

    g0 = geo[0] if C > 0 else np.zeros((H, W), dtype=np.float32)
    g3 = geo[3] if C > 3 else np.zeros((H, W), dtype=np.float32)
    g4 = geo[4] if C > 4 else np.zeros((H, W), dtype=np.float32)

    g0n = robust_normalize(g0, valid_mask > 0)
    g3n = robust_normalize(g3, valid_mask > 0)
    g4n = robust_normalize(g4, valid_mask > 0)

    # Main score: stronger dependence on G3/G4, G0 weaker role
    score = 0.15 * g0n + 0.50 * g3n + 0.35 * g4n
    score *= (valid_mask > 0).astype(np.float32)

    # Smooth a bit to make path guidance less noisy
    score = cv2.GaussianBlur(score, (0, 0), 1.2)
    score = robust_normalize(score, valid_mask > 0)

    # Slight sharpening of high-score region
    score = np.power(score, 1.2)

    # Zero outside valid region
    score[valid_mask == 0] = 0.0
    return score.astype(np.float32)


def sample_start_point_from_score(score, valid_mask, rng, quantile=0.75):
    """
    Sample a start point from the top-score region.
    """
    valid = valid_mask > 0
    vals = score[valid]
    if vals.size == 0:
        ys, xs = np.where(valid)
        if len(xs) == 0:
            raise RuntimeError("No valid pixels found.")
        k = rng.integers(0, len(xs))
        return np.array([xs[k], ys[k]], dtype=np.float32)

    thr = np.quantile(vals, quantile)
    candidate = valid & (score >= thr)

    ys, xs = np.where(candidate)
    if len(xs) == 0:
        ys, xs = np.where(valid)
        k = rng.integers(0, len(xs))
        return np.array([xs[k], ys[k]], dtype=np.float32)

    weights = score[ys, xs] + 1e-3
    weights = weights / weights.sum()

    k = rng.choice(len(xs), p=weights)
    return np.array([xs[k], ys[k]], dtype=np.float32)


def inside_image(x, y, W, H):
    return (0 <= x < W) and (0 <= y < H)


def normalize_vec(vx, vy, eps=1e-8):
    n = math.sqrt(vx * vx + vy * vy)
    if n < eps:
        return 1.0, 0.0
    return vx / n, vy / n


def guided_random_walk(score_map, valid_mask, rng,
                       min_steps=40, max_steps=110,
                       min_step_len=1.5, max_step_len=3.5,
                       angle_spread_deg=65,
                       num_candidates=13,
                       revisit_penalty=0.20,
                       direction_keep=0.80):
    """
    Generate a geometry-guided path for true crack.
    """
    H, W = score_map.shape
    visited = np.zeros((H, W), dtype=np.float32)

    start = sample_start_point_from_score(score_map, valid_mask, rng, quantile=0.75)
    x, y = float(start[0]), float(start[1])

    theta = rng.uniform(0, 2 * math.pi)
    dirx, diry = math.cos(theta), math.sin(theta)

    n_steps = int(rng.integers(min_steps, max_steps + 1))
    points = [(int(round(x)), int(round(y)))]

    for _ in range(n_steps):
        best = None
        best_score = -1e9

        # Candidate angles around current direction
        base_angle = math.atan2(diry, dirx)
        angles = np.linspace(base_angle - math.radians(angle_spread_deg),
                             base_angle + math.radians(angle_spread_deg),
                             num_candidates)

        for a in angles:
            step_len = float(rng.uniform(min_step_len, max_step_len))
            nx = x + step_len * math.cos(a)
            ny = y + step_len * math.sin(a)

            ix, iy = int(round(nx)), int(round(ny))
            if not inside_image(ix, iy, W, H):
                continue
            if valid_mask[iy, ix] == 0:
                continue

            cand_dirx, cand_diry = normalize_vec(nx - x, ny - y)
            align = cand_dirx * dirx + cand_diry * diry  # [-1,1]
            geom_score = float(score_map[iy, ix])
            revisit = float(visited[iy, ix])

            # Score: geometry + direction persistence - revisit penalty + tiny noise
            s = (1.4 * geom_score
                 + 0.35 * align
                 - revisit_penalty * revisit
                 + 0.03 * rng.normal())

            if s > best_score:
                best_score = s
                best = (nx, ny, cand_dirx, cand_diry)

        if best is None:
            break

        nx, ny, cand_dirx, cand_diry = best
        x, y = nx, ny

        # Update direction with persistence
        dirx = direction_keep * dirx + (1.0 - direction_keep) * cand_dirx
        diry = direction_keep * diry + (1.0 - direction_keep) * cand_diry
        dirx, diry = normalize_vec(dirx, diry)

        ix, iy = int(round(x)), int(round(y))
        points.append((ix, iy))

        if inside_image(ix, iy, W, H):
            visited[iy, ix] += 1.0

    return simplify_points(points)


def random_walk_distractor(valid_mask, rng,
                           min_steps=35, max_steps=95,
                           min_step_len=1.5, max_step_len=3.2,
                           angle_jitter_deg=40):
    """
    Generate a less geometry-aware distractor path.
    """
    H, W = valid_mask.shape
    ys, xs = np.where(valid_mask > 0)
    if len(xs) == 0:
        return []

    k = rng.integers(0, len(xs))
    x, y = float(xs[k]), float(ys[k])

    theta = rng.uniform(0, 2 * math.pi)
    dirx, diry = math.cos(theta), math.sin(theta)

    n_steps = int(rng.integers(min_steps, max_steps + 1))
    points = [(int(round(x)), int(round(y)))]

    for _ in range(n_steps):
        jitter = math.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
        theta = math.atan2(diry, dirx) + jitter
        step_len = float(rng.uniform(min_step_len, max_step_len))

        nx = x + step_len * math.cos(theta)
        ny = y + step_len * math.sin(theta)
        ix, iy = int(round(nx)), int(round(ny))

        if not inside_image(ix, iy, W, H):
            break
        if valid_mask[iy, ix] == 0:
            # small retry by changing direction
            theta = theta + math.radians(rng.uniform(60, 180))
            nx = x + step_len * math.cos(theta)
            ny = y + step_len * math.sin(theta)
            ix, iy = int(round(nx)), int(round(ny))
            if not inside_image(ix, iy, W, H):
                break
            if valid_mask[iy, ix] == 0:
                break

        x, y = nx, ny
        dirx, diry = normalize_vec(math.cos(theta), math.sin(theta))
        points.append((ix, iy))

    return simplify_points(points)


def simplify_points(points, min_dist=2.0):
    """
    Remove nearly-duplicate points.
    """
    if len(points) <= 1:
        return points

    new_pts = [points[0]]
    lastx, lasty = points[0]
    for x, y in points[1:]:
        if math.hypot(x - lastx, y - lasty) >= min_dist:
            new_pts.append((x, y))
            lastx, lasty = x, y
    return new_pts


def draw_polyline_mask(shape, points, thickness, rng, gap_prob=0.0):
    """
    Draw path onto mask, optionally skipping some short segments to create weak discontinuity.
    """
    H, W = shape[:2]
    canvas = np.zeros((H, W), dtype=np.uint8)

    if len(points) < 2:
        return canvas

    for i in range(len(points) - 1):
        if gap_prob > 0 and rng.random() < gap_prob:
            continue
        p1 = points[i]
        p2 = points[i + 1]
        cv2.line(canvas, p1, p2, 255, thickness=int(thickness), lineType=cv2.LINE_AA)

    return canvas


def overlay_dark_crack(image_bgr, line_mask, alpha=0.35, dark_strength=40, blur_sigma=0.8):
    """
    Darken image along line_mask.
    """
    img = image_bgr.astype(np.float32)

    m = line_mask.astype(np.float32) / 255.0
    if blur_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_sigma)
    m = np.clip(m, 0.0, 1.0)

    # Darkening amount
    delta = alpha * dark_strength * m
    out = img.copy()
    for c in range(3):
        out[..., c] = out[..., c] - delta

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def overlay_multiple_cracks(image_bgr, crack_items):
    """
    crack_items: list of dicts
      {
        "mask": mask_uint8,
        "alpha": float,
        "dark_strength": int/float,
        "blur_sigma": float
      }
    """
    out = image_bgr.copy()
    for item in crack_items:
        out = overlay_dark_crack(
            out,
            item["mask"],
            alpha=item["alpha"],
            dark_strength=item["dark_strength"],
            blur_sigma=item["blur_sigma"]
        )
    return out


def colorize_score_map(score):
    score_u8 = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)


# =========================================================
# Main generation for one sample
# =========================================================

def generate_one_sample(rgb_bgr, geo, rng, cfg):
    """
    Returns:
      crack_img_bgr, gt_mask_uint8, metadata, debug_dict
    """
    H, W = rgb_bgr.shape[:2]
    valid_mask = build_valid_mask(geo)
    score_map = build_geometry_score_map(geo, valid_mask)

    # -----------------------------
    # True cracks
    # -----------------------------
    n_true = int(rng.integers(cfg["n_true_min"], cfg["n_true_max"] + 1))
    true_masks = []
    true_meta = []

    for _ in range(n_true):
        pts = guided_random_walk(
            score_map=score_map,
            valid_mask=valid_mask,
            rng=rng,
            min_steps=cfg["true_min_steps"],
            max_steps=cfg["true_max_steps"],
            min_step_len=cfg["true_min_step_len"],
            max_step_len=cfg["true_max_step_len"],
            angle_spread_deg=cfg["true_angle_spread_deg"],
            num_candidates=cfg["true_num_candidates"],
            revisit_penalty=cfg["true_revisit_penalty"],
            direction_keep=cfg["true_direction_keep"],
        )

        thickness = int(rng.integers(cfg["true_thickness_min"], cfg["true_thickness_max"] + 1))
        gap_prob = float(rng.uniform(cfg["true_gap_prob_min"], cfg["true_gap_prob_max"]))

        m = draw_polyline_mask((H, W), pts, thickness=thickness, rng=rng, gap_prob=gap_prob)
        true_masks.append(m)

        true_meta.append({
            "n_points": len(pts),
            "thickness": thickness,
            "gap_prob": gap_prob,
        })

    # GT mask only contains true cracks
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    for m in true_masks:
        gt_mask = np.maximum(gt_mask, (m > 0).astype(np.uint8) * 255)

    # -----------------------------
    # Distractors
    # -----------------------------
    n_dist = int(rng.integers(cfg["n_dist_min"], cfg["n_dist_max"] + 1))
    dist_masks = []
    dist_meta = []

    for _ in range(n_dist):
        pts = random_walk_distractor(
            valid_mask=valid_mask,
            rng=rng,
            min_steps=cfg["dist_min_steps"],
            max_steps=cfg["dist_max_steps"],
            min_step_len=cfg["dist_min_step_len"],
            max_step_len=cfg["dist_max_step_len"],
            angle_jitter_deg=cfg["dist_angle_jitter_deg"],
        )

        thickness = int(rng.integers(cfg["dist_thickness_min"], cfg["dist_thickness_max"] + 1))
        gap_prob = float(rng.uniform(cfg["dist_gap_prob_min"], cfg["dist_gap_prob_max"]))

        m = draw_polyline_mask((H, W), pts, thickness=thickness, rng=rng, gap_prob=gap_prob)
        dist_masks.append(m)

        dist_meta.append({
            "n_points": len(pts),
            "thickness": thickness,
            "gap_prob": gap_prob,
        })

    # -----------------------------
    # Render to RGB image
    # -----------------------------
    crack_items = []

    # True cracks: weaker appearance
    for m in true_masks:
        alpha = float(rng.uniform(cfg["true_alpha_min"], cfg["true_alpha_max"]))
        dark_strength = float(rng.uniform(cfg["true_dark_min"], cfg["true_dark_max"]))
        blur_sigma = float(rng.uniform(cfg["true_blur_sigma_min"], cfg["true_blur_sigma_max"]))
        crack_items.append({
            "mask": m,
            "alpha": alpha,
            "dark_strength": dark_strength,
            "blur_sigma": blur_sigma
        })

    # Distractors: stronger appearance
    for m in dist_masks:
        alpha = float(rng.uniform(cfg["dist_alpha_min"], cfg["dist_alpha_max"]))
        dark_strength = float(rng.uniform(cfg["dist_dark_min"], cfg["dist_dark_max"]))
        blur_sigma = float(rng.uniform(cfg["dist_blur_sigma_min"], cfg["dist_blur_sigma_max"]))
        crack_items.append({
            "mask": m,
            "alpha": alpha,
            "dark_strength": dark_strength,
            "blur_sigma": blur_sigma
        })

    crack_img = overlay_multiple_cracks(rgb_bgr, crack_items)

    metadata = {
        "n_true_cracks": n_true,
        "n_distractors": n_dist,
        "true_meta": true_meta,
        "dist_meta": dist_meta,
        "score_formula": "0.15*G0 + 0.50*G3 + 0.35*G4",
        "notes": "GT mask contains only true cracks; distractors are not written into mask."
    }

    debug_dict = {
        "valid_mask": valid_mask,
        "score_map": score_map,
        "true_union": np.max(np.stack(true_masks, axis=0), axis=0) if len(true_masks) > 0 else np.zeros((H, W), np.uint8),
        "dist_union": np.max(np.stack(dist_masks, axis=0), axis=0) if len(dist_masks) > 0 else np.zeros((H, W), np.uint8),
    }

    return crack_img, gt_mask, metadata, debug_dict


# =========================================================
# Config
# =========================================================

def default_config():
    return {
        # true crack count
        "n_true_min": 1,
        "n_true_max": 1,

        # distractor count
        "n_dist_min": 1,
        "n_dist_max": 3,

        # true crack path
        "true_min_steps": 45,
        "true_max_steps": 110,
        "true_min_step_len": 1.5,
        "true_max_step_len": 3.2,
        "true_angle_spread_deg": 65,
        "true_num_candidates": 13,
        "true_revisit_penalty": 0.20,
        "true_direction_keep": 0.80,

        # true crack rendering (weak)
        "true_thickness_min": 1,
        "true_thickness_max": 2,
        "true_gap_prob_min": 0.03,
        "true_gap_prob_max": 0.10,
        "true_alpha_min": 0.25,
        "true_alpha_max": 0.45,
        "true_dark_min": 22,
        "true_dark_max": 45,
        "true_blur_sigma_min": 0.3,
        "true_blur_sigma_max": 1.0,

        # distractor path
        "dist_min_steps": 35,
        "dist_max_steps": 95,
        "dist_min_step_len": 1.4,
        "dist_max_step_len": 3.0,
        "dist_angle_jitter_deg": 45,

        # distractor rendering (stronger)
        "dist_thickness_min": 1,
        "dist_thickness_max": 2,
        "dist_gap_prob_min": 0.00,
        "dist_gap_prob_max": 0.06,
        "dist_alpha_min": 0.45,
        "dist_alpha_max": 0.75,
        "dist_dark_min": 35,
        "dist_dark_max": 70,
        "dist_blur_sigma_min": 0.0,
        "dist_blur_sigma_max": 0.8,
    }


# =========================================================
# Process one group
# =========================================================

def process_group(group_name, input_root, out_img_root, out_mask_root, out_debug_root,
                  rng, cfg, max_images=None, save_debug=False):
    group_in = Path(input_root) / group_name
    if not group_in.exists():
        print(f"[WARN] group not found: {group_in}")
        return

    group_out_img = Path(out_img_root) / group_name
    group_out_mask = Path(out_mask_root) / group_name
    ensure_dir(group_out_img)
    ensure_dir(group_out_mask)

    if save_debug:
        group_out_debug = Path(out_debug_root) / group_name
        ensure_dir(group_out_debug)
    else:
        group_out_debug = None

    ids = list_sample_ids(group_in)
    if max_images is not None:
        ids = ids[:max_images]

    print(f"[INFO] Processing {group_name}: {len(ids)} samples")

    for idx in ids:
        try:
            rgb = load_rgb(group_in, idx)
            geo = load_geometry(group_in, idx)

            crack_img, gt_mask, metadata, debug_dict = generate_one_sample(
                rgb_bgr=rgb,
                geo=geo,
                rng=rng,
                cfg=cfg
            )

            img_path = group_out_img / f"crack_{idx}.png"
            mask_path = group_out_mask / f"mask_{idx}.png"
            meta_path = group_out_img / f"meta_{idx}.json"

            cv2.imwrite(str(img_path), crack_img)
            cv2.imwrite(str(mask_path), gt_mask)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            if save_debug and group_out_debug is not None:
                score_vis = colorize_score_map(debug_dict["score_map"])
                valid_vis = (debug_dict["valid_mask"] * 255).astype(np.uint8)
                true_vis = debug_dict["true_union"]
                dist_vis = debug_dict["dist_union"]

                cv2.imwrite(str(group_out_debug / f"score_{idx}.png"), score_vis)
                cv2.imwrite(str(group_out_debug / f"valid_{idx}.png"), valid_vis)
                cv2.imwrite(str(group_out_debug / f"true_union_{idx}.png"), true_vis)
                cv2.imwrite(str(group_out_debug / f"dist_union_{idx}.png"), dist_vis)

            print(f"  [OK] {group_name} idx={idx}")

        except Exception as e:
            print(f"  [ERROR] {group_name} idx={idx}: {e}")


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="../gaussian-splatting/output/test",
        help="Root folder containing test_0, test_1, test_2"
    )
    parser.add_argument(
        "--out_img_root",
        type=str,
        default="./synthetic_crack_images_v3",
        help="Output root for crack images"
    )
    parser.add_argument(
        "--out_mask_root",
        type=str,
        default="./synthetic_crack_masks_v3",
        help="Output root for crack masks"
    )
    parser.add_argument(
        "--out_debug_root",
        type=str,
        default="./synthetic_crack_debug_v3",
        help="Optional debug output root"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["test_0", "test_1", "test_2"],
        help="Groups to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Process only first N images per group for quick inspection"
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Whether to save score/valid/union debug images"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = default_config()

    ensure_dir(args.out_img_root)
    ensure_dir(args.out_mask_root)
    if args.save_debug:
        ensure_dir(args.out_debug_root)

    print("[INFO] v3 crack generation started")
    print(f"[INFO] input_root    = {args.input_root}")
    print(f"[INFO] out_img_root  = {args.out_img_root}")
    print(f"[INFO] out_mask_root = {args.out_mask_root}")
    print(f"[INFO] groups        = {args.groups}")
    print(f"[INFO] seed          = {args.seed}")
    print(f"[INFO] max_images    = {args.max_images}")
    print(f"[INFO] save_debug    = {args.save_debug}")

    for group_name in args.groups:
        process_group(
            group_name=group_name,
            input_root=args.input_root,
            out_img_root=args.out_img_root,
            out_mask_root=args.out_mask_root,
            out_debug_root=args.out_debug_root,
            rng=rng,
            cfg=cfg,
            max_images=args.max_images,
            save_debug=args.save_debug
        )

    print("[INFO] v3 crack generation finished.")


if __name__ == "__main__":
    main()