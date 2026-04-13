import os
import math
import torch
import torch.nn.functional as F


def batched_cdist_to_single(features: torch.Tensor,
                            single_feature: torch.Tensor,
                            chunk_size: int = 8192) -> torch.Tensor:
    """
    分块计算 features 到 single_feature 的欧氏距离

    features: [N, C]
    single_feature: [1, C]
    return: [N]
    """
    device = features.device
    n = features.shape[0]
    dist_list = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = features[start:end]  # [b, C]
        dist = torch.cdist(chunk, single_feature).squeeze(1)  # [b]
        dist_list.append(dist)

    return torch.cat(dist_list, dim=0).to(device)


def greedy_coreset_chunked(
    features: torch.Tensor,
    sample_ratio: float = 0.1,
    device: str = "cuda",
    chunk_size: int = 8192
) -> torch.Tensor:
    """
    基于 greedy coreset 的分块 GPU 采样

    features: [N, C]
    return: sampled_features [M, C]
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, C], but got shape {features.shape}")

    # 放到 device
    features = features.to(device, non_blocking=True)
    features = F.normalize(features, dim=1)

    n = features.shape[0]
    m = max(1, int(n * sample_ratio))

    print(f"Total features: {n}")
    print(f"Sample ratio: {sample_ratio}")
    print(f"Coreset size: {m}")
    print(f"Chunk size: {chunk_size}")
    print(f"Using device: {device}")

    selected_indices = []

    # 固定第一个点
    first_idx = 0
    selected_indices.append(first_idx)

    first_feature = features[first_idx:first_idx + 1]  # [1, C]

    # 初始 min_distances: 所有点到第一个点的距离
    min_distances = batched_cdist_to_single(
        features, first_feature, chunk_size=chunk_size
    )

    for i in range(m - 1):
        if i % 50 == 0 or i == m - 2:
            current_max = torch.max(min_distances).item()
            current_mean = torch.mean(min_distances).item()
            print(
                f"[{i + 1}/{m - 1}] "
                f"max min-dist = {current_max:.6f}, mean min-dist = {current_mean:.6f}"
            )

        # 选当前离已选集合最远的点
        farthest_idx = torch.argmax(min_distances).item()
        selected_indices.append(farthest_idx)

        new_selected = features[farthest_idx:farthest_idx + 1]  # [1, C]

        # 分块计算所有点到新选点的距离
        new_dist = batched_cdist_to_single(
            features, new_selected, chunk_size=chunk_size
        )

        # 维护“到已选集合的最小距离”
        min_distances = torch.minimum(min_distances, new_dist)

    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
    sampled_features = features[selected_indices].detach().cpu()
    return sampled_features


if __name__ == "__main__":
    # ------------------------
    # config
    # ------------------------
    memory_bank_path = "memory_bank.pt"
    save_path = "memory_bank_coreset.pt"

    # 你可以按需调整
    sample_ratio = 0.1       # 先用 10%
    chunk_size = 4096        # 8GB 显存建议先 4096，稳一点
    prefer_gpu = True

    # ------------------------
    # device
    # ------------------------
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # ------------------------
    # load memory bank
    # ------------------------
    if not os.path.exists(memory_bank_path):
        raise FileNotFoundError(f"memory bank not found: {memory_bank_path}")

    memory_bank = torch.load(memory_bank_path, map_location="cpu")
    print("Original memory bank:", memory_bank.shape)

    # ------------------------
    # run coreset
    # ------------------------
    sampled = greedy_coreset_chunked(
        memory_bank,
        sample_ratio=sample_ratio,
        device=device,
        chunk_size=chunk_size
    )

    print("Coreset memory bank:", sampled.shape)

    # ------------------------
    # save
    # ------------------------
    torch.save(sampled, save_path)
    print("Saved to", save_path)