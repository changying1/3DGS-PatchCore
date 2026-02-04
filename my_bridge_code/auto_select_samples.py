import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from PIL import Image
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 假设这是3DGS渲染出来的图片路径（等服务器有了之后，填写真实路径）
INPUT_IMAGES_DIR = "../gaussian-splatting/output/my_relic/train/renders"
# 输出给 PatchCore 用的数据集路径
OUTPUT_DATASET_DIR = "./dataset/mvtec_format/my_relic"

PATCH_SIZE = 224      # 切片大小 (PatchCore默认喜欢224或256)
STRIDE = 224          # 步长 (如果想重叠切片，可以设小一点，比如112)
NUM_CLUSTERS = 2      # 聚类数量 (2类：正常纹理 vs 异常/背景)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def load_feature_extractor():
    """加载一个预训练的 ResNet50 用于提取图像特征"""
    print(f"正在加载 ResNet50 (Device: {DEVICE})...")
    # 使用 ResNet50，去掉最后的全连接层，只取特征
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]) 
    model.to(DEVICE)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess

def extract_features(model, preprocess, patch_img):
    """输入一张图片(PIL)，输出特征向量"""
    img_tensor = preprocess(patch_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = model(img_tensor)
    # 展平特征: [1, 2048, 1, 1] -> [2048]
    return feature.cpu().numpy().flatten()

def main():
    # 1. 准备目录结构 (MVTec AD 格式)
    # MVTec 格式要求：
    #   train/good/ (存放正常样本)
    #   test/defect/ (存放测试样本，这里我们先把所有图都放进去做测试)
    train_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "good")
    test_dir = os.path.join(OUTPUT_DATASET_DIR, "test", "defect")
    
    if os.path.exists(OUTPUT_DATASET_DIR):
        shutil.rmtree(OUTPUT_DATASET_DIR)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # 2. 扫描图片
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"警告: 输入目录 {INPUT_IMAGES_DIR} 不存在。请等待 3DGS 渲染完成后再运行。")
        # 为了让你现在能跑通逻辑，我们创建一个伪造的空列表
        image_files = [] 
    else:
        image_files = [os.path.join(INPUT_IMAGES_DIR, f) for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]

    print(f"找到 {len(image_files)} 张图片。")
    if len(image_files) == 0:
        print("没有图片，程序退出。")
        return

    model, preprocess = load_feature_extractor()
    
    all_patches = []      # 存放切片本身(文件名+位置)
    all_features = []     # 存放特征向量

    print("开始切片并提取特征...")
    patch_id = 0
    
    # 3. 遍历图片进行切片
    for img_path in tqdm(image_files):
        # 同时把所有原图复制到 test/defect 文件夹，作为待检测的全图
        shutil.copy(img_path, test_dir)
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 滑动窗口切片
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                patch = img.crop(box)
                
                # 提取特征
                feat = extract_features(model, preprocess, patch)
                
                all_patches.append({
                    'id': patch_id,
                    'image': patch,
                    'source': os.path.basename(img_path)
                })
                all_features.append(feat)
                patch_id += 1

    if not all_features:
        print("未提取到特征。")
        return

    # 4. K-Means 聚类
    print(f"正在对 {len(all_features)} 个切片进行聚类 (K={NUM_CLUSTERS})...")
    all_features = np.array(all_features)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10).fit(all_features)
    labels = kmeans.labels_

    # 5. 找出“最大簇”（认为是正常纹理）
    counts = np.bincount(labels)
    major_cluster_label = np.argmax(counts)
    print(f"聚类完成。各簇数量: {counts}。 判定簇 {major_cluster_label} 为【正常纹理】。")

    # 6. 保存正常切片到 train/good
    print("正在保存训练样本...")
    save_count = 0
    for i, item in enumerate(all_patches):
        if labels[i] == major_cluster_label:
            # 可以在这里加一个随机筛选，如果数量太多只存前50张，防止训练太慢
            if save_count < 50: 
                save_path = os.path.join(train_dir, f"patch_{item['id']}.png")
                item['image'].save(save_path)
                save_count += 1
    
    print(f"处理完成！\n训练集路径: {train_dir} (共 {save_count} 张)\n测试集路径: {test_dir}")
    print("下一步：使用 PatchCore 运行该数据集。")

if __name__ == "__main__":
    main()