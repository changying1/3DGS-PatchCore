import torch
import os


def greedy_coreset(features: torch.Tensor, sample_ratio: float = 0.1):
    """
    features: [N, C]
    return: sampled_features [M, C]
    """
    N = features.shape[0]
    M = max(1, int(N * sample_ratio))

    features = torch.nn.functional.normalize(features, dim=1)

    selected_indices = []
    remaining_indices = list(range(N))

    # 先随机选一个起点
    first_idx = 0
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    selected = features[first_idx:first_idx+1]
    min_distances = torch.cdist(features, selected).squeeze(1)

    for _ in range(M - 1):
        farthest_idx = torch.argmax(min_distances).item()
        selected_indices.append(farthest_idx)

        new_selected = features[farthest_idx:farthest_idx+1]
        new_dist = torch.cdist(features, new_selected).squeeze(1)
        min_distances = torch.minimum(min_distances, new_dist)

    sampled_features = features[selected_indices]
    return sampled_features


if __name__ == "__main__":
    memory_bank_path = "memory_bank.pt"
    save_path = "memory_bank_coreset.pt"

    memory_bank = torch.load(memory_bank_path)
    print("Original memory bank:", memory_bank.shape)

    sampled = greedy_coreset(memory_bank, sample_ratio=0.1)

    print("Coreset memory bank:", sampled.shape)

    torch.save(sampled, save_path)
    print("Saved to", save_path)