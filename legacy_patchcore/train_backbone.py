import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from gaussian_patchcore_model import GaussianPatchCoreBackbone
from train_backbone_dataset import BackboneTrainDataset


class BackboneBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GaussianPatchCoreBackbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(256, 2)  # resnet18 layer3 输出 256 通道

    def forward(self, x):
        f2, f3 = self.backbone(x)
        feat = self.pool(f3).flatten(1)
        logits = self.head(feat)
        return logits


def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()
            total_loss += loss.item() * y.size(0)

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return avg_loss, acc


def main():
    normal_dir = os.environ.get(
        "NORMAL_DIR",
        "../gaussian-splatting/output/test/test_0"
    )
    crack_dir = os.environ.get(
        "CRACK_DIR",
        "synthetic_crack_images/test_0"
    )

    batch_size = 8
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-4
    save_path = "backbone_finetuned.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BackboneTrainDataset(
        normal_dir=normal_dir,
        crack_dir=crack_dir,
        crop_ratio=0.7
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = BackboneBinaryClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)

            batch_size_now = y.size(0)
            total += batch_size_now
            correct += (pred == y).sum().item()
            running_loss += loss.item() * batch_size_now

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.backbone.state_dict(), save_path)
            print(f"[OK] Saved best backbone to {save_path}")

    print("Training done.")
    print("Best val acc:", best_acc)


if __name__ == "__main__":
    main()