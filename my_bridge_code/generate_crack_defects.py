import os
import cv2
import numpy as np
import random

# -----------------------
# config
# -----------------------

rgb_dir = "../gaussian-splatting/output/test/test_0"

save_img_dir = "synthetic_crack_images"
save_mask_dir = "synthetic_crack_masks"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)


rgb_files = sorted([
    f for f in os.listdir(rgb_dir)
    if f.startswith("rgb_")
])


def draw_random_crack(img, mask):

    h, w = img.shape[:2]

    # crack parameters
    length = random.randint(40, 120)
    thickness = random.randint(1, 3)

    # start point
    x = random.randint(w//4, 3*w//4)
    y = random.randint(h//4, 3*h//4)

    angle = random.uniform(0, np.pi)

    pts = []

    for i in range(length):

        dx = int(i * np.cos(angle))
        dy = int(i * np.sin(angle))

        px = x + dx + random.randint(-1,1)
        py = y + dy + random.randint(-1,1)

        if 0 <= px < w and 0 <= py < h:
            pts.append((px,py))

    # draw crack
    for p in pts:
        cv2.circle(img, p, thickness, (20,20,20), -1)
        cv2.circle(mask, p, thickness, 255, -1)

    return img, mask


# -----------------------
# generate
# -----------------------

for rgb_name in rgb_files:

    rgb_path = os.path.join(rgb_dir, rgb_name)

    img = cv2.imread(rgb_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    img_crack, mask = draw_random_crack(img.copy(), mask)

    idx = rgb_files.index(rgb_name)

    img_save = os.path.join(save_img_dir, f"crack_{idx}.png")
    mask_save = os.path.join(save_mask_dir, f"mask_{idx}.png")

    cv2.imwrite(img_save, img_crack)
    cv2.imwrite(mask_save, mask)

print("Crack dataset generated.")