# python train_unet.py --mode rgb --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g0 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g0123 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

# python train_unet.py --mode rgb_g01234 --train_ids 1,2 --val_id 0 --seed 1 --epochs 50 --batch_size 2 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test

python infer_unet.py --ckpt ./checkpoints_unet_v3/fold_val0/rgb/seed1/best_iou.pth --mode rgb --data_id 0 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test --threshold 0.5 --max_samples 30

python infer_unet.py --ckpt ./checkpoints_unet_v3/fold_val0/rgb_g0/seed1/best_iou.pth --mode rgb_g0 --data_id 0 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test --threshold 0.5 --max_samples 30

python infer_unet.py --ckpt ./checkpoints_unet_v3/fold_val0/rgb_g0123/seed1/best_iou.pth --mode rgb_g0123 --data_id 0 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test --threshold 0.5 --max_samples 30

python infer_unet.py --ckpt ./checkpoints_unet_v3/fold_val0/rgb_g01234/seed1/best_iou.pth --mode rgb_g01234 --data_id 0 --crack_root ./synthetic_crack_images_v3 --mask_root ./synthetic_crack_masks_v3 --geom_root ../gaussian-splatting/output/test --threshold 0.5 --max_samples 30