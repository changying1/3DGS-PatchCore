import sys
sys.path.append("..")

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from gaussian_model import GaussianModel


def save_rgb(image, path):
    image = torch.clamp(image, 0, 1)
    image = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(path)


def save_depth(depth, path):
    depth = depth.squeeze().cpu().numpy()
    np.save(path, depth)


def export_dataset(args):

    safe_state(False)

    # ---------- Load Dataset ----------
    dataset = ModelParams.extract(args)
    pipe = PipelineParams.extract(args)
    opt = OptimizationParams.extract(args)

    # 强制稳定设置（文保病害检测专用）
    pipe.antialiasing = False
    args.use_trained_exp = False

    # ---------- Load Model ----------
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    checkpoint_path = os.path.join(
        args.model_path,
        f"chkpnt{args.iteration}.pth"
    )

    print("Loading checkpoint:", checkpoint_path)

    model_params, _ = torch.load(checkpoint_path)
    gaussians.restore(model_params, opt)

    # ---------- Select Cameras ----------
    if args.split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()

    output_dir = os.path.join(args.output_path, args.split)
    os.makedirs(output_dir, exist_ok=True)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Exporting {len(cameras)} views...")

    for idx, cam in enumerate(tqdm(cameras)):

        render_pkg = render(
            cam,
            gaussians,
            pipe,
            background,
            use_trained_exp=False,
            separate_sh=True
        )

        rgb = render_pkg["render"]
        depth = render_pkg.get("depth", None)

        save_rgb(rgb, os.path.join(output_dir, f"view_{idx:03d}_rgb.png"))

        if depth is not None:
            save_depth(depth, os.path.join(output_dir, f"view_{idx:03d}_depth.npy"))

    print("Export finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 注册参数组
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./exported_dataset")

    args = parser.parse_args()

    export_dataset(args)