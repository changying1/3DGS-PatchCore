# 多fold训练——fold0
# python train_unet.py --mode rgb --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g0 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g0123 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g01234 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# 多fold训练——fold0可视化推理
python .\infer_unet.py --ablation --modes rgb --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

python .\infer_unet.py --ablation --modes rgb_g0 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

python .\infer_unet.py --ablation --modes rgb_g0123 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

python .\infer_unet.py --ablation --modes rgb_g01234 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

# 多fold训练——fold0单独训练几何分支（消融）
# python train_unet.py --mode g0 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode g0123 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode g01234 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# 多fold训练——fold0单独训练几何分支（消融）可视化推理
python .\infer_unet.py --ablation --modes g0 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

python .\infer_unet.py --ablation --modes g0123 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1

python .\infer_unet.py --ablation --modes g01234 --ckpt_root .\checkpoints_unet_v3 --val_id 0 --seed 1 --data_id 0 --crack_root .\synthetic_crack_images_v3 --mask_root .\synthetic_crack_masks_v3 --geom_root ..\gaussian-splatting\output\test --save_dir .\infer_results_v3\test_0 --batch_size 1