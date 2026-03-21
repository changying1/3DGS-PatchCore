import cv2
import numpy as np
import os

def white_to_black_bg(img):
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 生成mask（白背景）
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # 反转mask（前景）
    mask_inv = cv2.bitwise_not(mask)

    # 让边缘更平滑（关键！）
    mask_inv = cv2.GaussianBlur(mask_inv, (3, 3), 0)

    # 归一化
    alpha = mask_inv.astype(float) / 255.0

    # 生成黑背景
    black_bg = np.zeros_like(img)

    # 前景 + 背景融合（保边缘）
    result = (img * alpha[..., None] + black_bg * (1 - alpha[..., None])).astype(np.uint8)

    return result


# 批量处理
input_dir = r"D:\My_Thesis_Project\3DGS-PatchCore\heritage_dataset\images\4-r"
output_dir = r"D:\My_Thesis_Project\3DGS-PatchCore\heritage_dataset\images\4"
os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(input_dir):
    if name.endswith(".png") or name.endswith(".jpg"):
        path = os.path.join(input_dir, name)
        img = cv2.imread(path)

        result = white_to_black_bg(img)

        cv2.imwrite(os.path.join(output_dir, name), result)