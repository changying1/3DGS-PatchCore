#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# 这是专门针对白背景下的数据集进行训练的脚本

import os
import sys
import torch
from random import randint
from argparse import ArgumentParser
import torchvision

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import loadCam
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, focal2fov
from tqdm import tqdm
from utils.general_utils import get_expon_lr_func
# from utils.depth_utils import depth_to_normal
from models.gaussian_feature_extractor_white import GaussianFeatureExtractor

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def apply_wenbao_disease_preset(args, dataset, opt, pipe):
    """Preset tuned for cultural-heritage disease detection (3DGS renders -> PatchCore features).

    Goals:
      - Avoid photometric transforms by default (exposure transform changes feature distribution)
      - Preserve micro-structure (cracks, exfoliation) during training renders
      - Conservative optimization (avoid over-densifying / "painting over" disease texture)

    This function only touches attributes if they exist, to stay compatible with upstream code.
    """
    if getattr(args, "preset", "none") != "wenbao_disease":
        return

    # Core switches (most important)
    args.use_trained_exp = False   # keep illumination differences as potential anomaly signal
    args.separate_sh = True        # more stable / explicit SH handling

    # Rendering AA strategy: off for training loop, on for viewer/eval
    # (render() reads pipe.antialiasing)
    pipe.antialiasing = getattr(args, "antialiasing_train", False)

    # Dataset-side: do not auto-enable exposure training/eval unless user explicitly wants it
    if hasattr(dataset, "train_test_exp"):
        dataset.train_test_exp = False

    # Optimization tuning (safe, conservative defaults for disease detection)
    conservative = {
        # "iterations": 200,   # Commented out to respect command line --iterations
        "densify_from_iter": 500,
        "densify_until_iter": 12000,
        "densification_interval": 200,
        "opacity_reset_interval": 3000,
        "percent_dense": 0.01,
        "lambda_dssim": 0.0,
        "random_background": False,
    }
    for k, v in conservative.items():
        if hasattr(opt, k):
            setattr(opt, k, v)

    # Model tuning (if present)
    # Lower SH degree reduces overfitting to lighting; good for anomaly-feature stability
    if hasattr(dataset, "sh_degree"):
        dataset.sh_degree = min(getattr(dataset, "sh_degree", 3), 2)

def save_single_channel_vis(ch, save_path, low=0.01, high=0.99):
    ch = ch.detach().clone()
    ch = torch.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)

    nz = ch[ch > 0]
    if nz.numel() < 16:
        torchvision.utils.save_image(ch.clamp(0, 1).unsqueeze(0), save_path)
        return

    q_low = torch.quantile(nz, low)
    q_high = torch.quantile(nz, high)

    ch = torch.clamp(ch, q_low, q_high)
    if (q_high - q_low) > 1e-8:
        ch = (ch - q_low) / (q_high - q_low + 1e-6)
    else:
        ch = torch.zeros_like(ch)

    torchvision.utils.save_image(ch.clamp(0, 1).unsqueeze(0), save_path)

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    # Apply preset overrides (if --preset wenbao_disease)
    apply_wenbao_disease_preset(args, dataset, opt, pipe)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    extractor = GaussianFeatureExtractor()
    gaussians = GaussianModel(
        dataset.sh_degree,
        opt.optimizer_type,
        separate_sh=args.separate_sh,
        use_trained_exp=args.use_trained_exp
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    # Viewer render: AA ON by default (stable preview)
                    pipe.antialiasing = getattr(args, "antialiasing_view", True)
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=args.use_trained_exp,
                        separate_sh=args.separate_sh
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_indices:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        if len(viewpoint_indices) == 0:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        random_index = randint(0, len(viewpoint_indices) - 1)
        cam_index = viewpoint_indices.pop(random_index)
        viewpoint_cam = viewpoint_stack[cam_index]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Training render: AA OFF by default (preserve micro-cracks high-frequency)
        pipe.antialiasing = getattr(args, "antialiasing_train", False)
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            use_trained_exp=args.use_trained_exp,
            separate_sh=args.separate_sh
        )

        image = render_pkg["render"]

        # 保留 geometry 供后续 8 通道链路使用
        #geometry_map = extractor.extract_geometry_map(render_pkg, gaussians)

        # 只在首轮和可视化轮次做打印/保存，避免频繁 IO
        save_geometry_vis = (iteration == first_iter) or (iteration % 20 == 0)

        geometry_map = None
        if save_geometry_vis:
            with torch.no_grad():
                geometry_map = extractor.extract_geometry_map(render_pkg, gaussians)

        if iteration == first_iter and geometry_map is not None:
            print("Geometry map shape:", tuple(geometry_map.shape))

        if save_geometry_vis and geometry_map is not None:
            print(f"\n[Iter {iteration}] geometry stats:")
            for c in range(geometry_map.shape[0]):
                save_path = os.path.join(
                    scene.model_path, f"geometry_c{c}_{iteration}.png"
                )
                save_single_channel_vis(geometry_map[c], save_path)

            rgb_path = os.path.join(scene.model_path, f"rgb_{iteration}.png")
            torchvision.utils.save_image(image.clamp(0, 1), rgb_path)
            
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        # DSSIM weight is small in this preset to preserve thin cracks
        if opt.lambda_dssim > 0:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            loss = Ll1

        if hasattr(viewpoint_cam, "depth") and viewpoint_cam.depth is not None:
            depth = render_pkg.get("depth", None)
            if depth is not None:
                gt_depth = viewpoint_cam.depth.cuda()
                Ldepth = torch.abs(depth - gt_depth).mean()
                w = depth_l1_weight(iteration)
                loss = loss + w * Ldepth
                ema_Ll1depth_for_log = 0.4 * Ldepth.item() + 0.6 * ema_Ll1depth_for_log

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Test
            if iteration in testing_iterations:
                test_and_log(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), scene, render, (pipe, background),
                            use_trained_exp=args.use_trained_exp, separate_sh=args.separate_sh)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(randint(0, 999999))
        args.model_path = os.path.join("./output/", unique_str)

    # Create output folder
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(args))

    # Create Tensorboard writer
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(args.model_path)
    except Exception:
        pass
    return tb_writer


def test_and_log(tb_writer, iteration, Ll1, loss, elapsed, scene: Scene, renderFunc, renderArgs,
                 use_trained_exp: bool, separate_sh: bool):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config["cameras"]):
                pipe, background = renderArgs
                # For eval renders you may prefer AA ON; keep as current pipe state
                image = renderFunc(
                    viewpoint, scene.gaussians, pipe, background,
                    use_trained_exp=use_trained_exp,
                    separate_sh=separate_sh
                )["render"]
                gt_image = viewpoint.original_image.cuda()
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()

            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print(
                "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config["name"], l1_test, psnr_test
                )
            )
            if tb_writer:
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # Wenbao (cultural heritage) disease-detection preset (3DGS -> PatchCore)
    parser.add_argument(
        "--preset", type=str, default="wenbao_disease",
        choices=["none", "wenbao_disease"],
        help="Parameter preset for cultural-heritage disease detection."
    )

    # Rendering / feature-stability switches for disease detection
    parser.add_argument(
        "--use_trained_exp", action="store_true", default=False,
        help="Apply trained exposure transform in render(). For disease detection: default OFF."
    )
    parser.add_argument(
        "--separate_sh", action="store_true", default=True,
        help="Use separate SH parameterization (dc + sh_rest). Recommended ON."
    )

    # Anti-aliasing control: often OFF during training to keep micro-cracks high-frequency, ON for viewer/eval
    parser.add_argument(
        "--antialiasing_train", action="store_true", default=False,
        help="Enable gaussian anti-aliasing during TRAIN renders (default OFF)."
    )
    parser.add_argument(
        "--antialiasing_view", action="store_true", default=True,
        help="Enable gaussian anti-aliasing for VIEWER renders (default ON)."
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args
    )

    print("\nTraining complete.")