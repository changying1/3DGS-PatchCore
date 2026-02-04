#!/bin/bash

# 设置 PYTHONPATH 确保能找到包
export PYTHONPATH=src

# 数据集路径 (指向我们刚才生成的那个文件夹)
# 注意：这里需要填你实际的 my_bridge_code/dataset/mvtec_format 路径
DATASET_PATH="../my_bridge_code/dataset/mvtec_format"

# 运行 PatchCore
python bin/run_patchcore.py \
--gpu 0 \
--seed 0 \
--save_patchcore_model \
--log_group my_relic_experiment \
--log_project MVTecAD_Results \
--results_path results \
patch_core \
-b (Wait, looking at your provided files, arguments might differ, let me check provided args) \
# ...