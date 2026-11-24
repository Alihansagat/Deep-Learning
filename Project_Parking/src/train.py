"""Train and compare two small CNNs for parking-patch occupancy classification.

This dataset contains one already-cropped parking space per image. The task is
therefore binary classification (free/busy), not object detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABEL_TO_ID = {"busy": 0, "free": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass
class Sample:
    path: Path
    label: int
    camera: str
    date: str
    time_key: str
    group: str


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch so both experiment runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_samples(data_dir: Path) -> list[Sample]:
    """Read the CNRPark folder layout: CAMERA/{free,busy}/YYYYMMDD_HHMM_ID.jpg."""
    samples: list[Sample] = []
    pattern = re.compile(r"^(\d{8})_(\d{4})_(\d+)\.(jpg|jpeg|png)$", re.I)
    for image_path in sorted(data_dir.glob("*/*/*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        camera = image_path.parent.parent.name
        class_name = image_path.parent.name.lower()
        if class_name not in LABEL_TO_ID:
            continue
        match = pattern.match(image_path.name)
        if not match:
            continue
        date, hhmm, _slot, _ext = match.groups()
        # Keep every capture from the same 30-minute interval together. The
        # timestamp has 5-minute resolution in CNRPark; grouping nearby frames
        # reduces leakage from almost identical images across train/validation.
        minutes = int(hhmm[2:])
        hour = int(hhmm[:2])
        bucket = (hour * 60 + minutes) // 30
        group = f"{camera}_{date}_{bucket:02d}"
        samples.append(Sample(image_path, LABEL_TO_ID[class_name], camera, date, hhmm, group))
    if not samples:
        raise FileNotFoundError(f"No CNRPark image files found under {data_dir}")
    return samples


def make_splits(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample], list[Sample], dict]:
    """Use the earlier capture day for train/val and hold out the later day for test.

    The downloaded CNRPark subset has only two days, each paired with a different
    camera (July 3 / camera A, July 8 / camera B). Thus the test measures a
    combined date-and-viewpoint shift; this limitation is recorded in results.
    """
    dates = sorted({s.date for s in samples})
    if len(dates) < 2:
        raise ValueError("Need at least two capture dates for the held-out-day test split.")
    train_date, test_date = dates[0], dates[-1]
    train_pool = [s for s in samples if s.date == train_date]
    test = [s for s in samples if s.date == test_date]
    # GroupShuffleSplit holds out complete camera/date/time blocks for validation.
    y = np.array([s.label for s in train_pool])
    groups = np.array([s.group for s in train_pool])
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(split.split(np.zeros(len(train_pool)), y, groups))
    train = [train_pool[i] for i in train_idx]
    val = [train_pool[i] for i in val_idx]
    meta = {
        "method": "held-out capture date; grouped 30-minute blocks for train/validation",
        "train_validation_date": train_date,
        "test_date": test_date,
        "train_cameras": sorted({s.camera for s in train_pool}),
        "test_cameras": sorted({s.camera for s in test}),
        "limitation": "Only two dates are present and each date maps to a different camera; date and viewpoint effects are confounded.",
    }
    return train, val, test, meta


class ParkingPatchDataset(Dataset):
    """Load RGB crops and resize them to the configured image size."""
    def __init__(self, samples: list[Sample], image_size: int, augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        # Simple horizontal flip augmentation is identical in both experiments.
        if self.augment and random.random() < 0.5:
            array = np.flip(array, axis=1).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, sample.label


class SmallParkingCNN(nn.Module):
    """Compact CNN with adaptive pooling, so it accepts multiple image sizes."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # 3x3 divides the 18x18 feature map produced from 150x150 inputs;
            # this also avoids an adaptive-pooling limitation on Apple MPS.
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 3 * 3, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_all.extend(labels.numpy().tolist())
            preds_all.extend(probs.argmax(axis=1).tolist())
            probs_all.extend(probs.tolist())
    y = np.asarray(labels_all)
    pred = np.asarray(preds_all)
    prob = np.asarray(probs_all)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_free": float(p),
        "recall_free": float(r),
        "f1_free": float(f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "average_precision_free": float(average_precision_score(y == 1, prob[:, 1])),
        "confusion_matrix_labels_busy_free": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "support_busy": int((y == 0).sum()),
        "support_free": int((y == 1).sum()),
    }
    return metrics, y, pred, prob


def train_one(train_samples, val_samples, *, lr, args, device, run_dir):
    """Run one controlled configuration and keep the best validation checkpoint."""
    # Reset the seed so both LR runs start from exactly the same initialization.
    set_seed(args.seed)
    train_loader = DataLoader(ParkingPatchDataset(train_samples, args.image_size, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ParkingPatchDataset(val_samples, args.image_size),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SmallParkingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        model.eval()
        val_loss_sum, val_seen, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss_sum += criterion(logits, labels).item() * labels.size(0)
                val_seen += labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
        train_loss = running_loss / max(seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_accuracy = val_correct / max(val_seen, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"lr={lr:g} epoch {epoch+1:02d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
        # Select the checkpoint using validation loss only; test is untouched here.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    val_metrics, *_ = evaluate(model, val_loader, device)
    return model, history, val_metrics, elapsed


def write_manifest(splits, output_dir):
    manifest_path = output_dir / "data_splits.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "split", "label", "camera", "date", "time", "group"])
        for split_name, records in splits.items():
            for s in records:
                writer.writerow([s.path.name, split_name, ID_TO_LABEL[s.label], s.camera, s.date, s.time_key, s.group])
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("CNRPark-Patches-150x150"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=1409)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--experiment-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    samples = discover_samples(args.data_dir)
    train_samples, val_samples, test_samples, split_meta = make_splits(samples, args.seed)
    split_summary = {"split_protocol": split_meta, "counts": {}}
    for name, records in (("train", train_samples), ("validation", val_samples), ("test", test_samples)):
        split_summary["counts"][name] = {
            "total": len(records),
            "labels": {ID_TO_LABEL[k]: v for k, v in Counter(s.label for s in records).items()},
            "cameras": dict(Counter(s.camera for s in records)),
            "dates": dict(Counter(s.date for s in records)),
        }
    (args.output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    write_manifest({"train": train_samples, "validation": val_samples, "test": test_samples}, args.output_dir)
    print("Device:", device)
    print(json.dumps(split_summary["counts"], indent=2))

    run_results = {}
    for name, lr in (("baseline_lr_0.001", args.baseline_lr), ("experiment_lr_0.0001", args.experiment_lr)):
        run_dir = args.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        model, history, val_metrics, elapsed = train_one(
            train_samples, val_samples, lr=lr, args=args, device=device, run_dir=run_dir)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        run_results[name] = {
            "learning_rate": lr,
            "training_seconds": elapsed,
            "validation_metrics": val_metrics,
        }
        plt.figure(figsize=(7, 4))
        plt.plot(history["train_loss"], label="Train loss")
        plt.plot(history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"Learning curves — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curves.png", dpi=160)
        plt.close()

    # Choose the final run using validation macro-F1 (test metrics are not used).
    selected_name = max(run_results, key=lambda k: run_results[k]["validation_metrics"]["macro_f1"])
    selected_lr = run_results[selected_name]["learning_rate"]
    selected_run_dir = args.output_dir / selected_name
    model = SmallParkingCNN().to(device)
    model.load_state_dict(torch.load(selected_run_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loader = DataLoader(ParkingPatchDataset(test_samples, args.image_size),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_metrics, y_test, pred_test, prob_test = evaluate(model, test_loader, device)

    # Save a compact final checkpoint with metadata for later inference.
    final_path = args.output_dir / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "class_to_idx": LABEL_TO_ID,
                "image_size": args.image_size,
                "selected_learning_rate": selected_lr,
                "seed": args.seed}, final_path)

    # Save several test examples, prioritizing mistakes or uncertain predictions.
    order = np.argsort(np.abs(prob_test[:, 1] - 0.5))
    chosen = list(order[:10])
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, idx in zip(axes.flat, chosen):
        sample = test_samples[int(idx)]
        ax.imshow(Image.open(sample.path).convert("RGB"))
        ax.set_title(f"true={ID_TO_LABEL[y_test[idx]]}\npred={ID_TO_LABEL[pred_test[idx]]} ({prob_test[idx].max():.2f})", fontsize=9)
        ax.axis("off")
    plt.suptitle("Held-out test examples (hardest confidence cases)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "test_examples.png", dpi=160)
    plt.close()

    result = {
        "task": "binary image classification of pre-cropped parking-space patches",
        "detection_metric_note": "mAP@0.50 is not applicable: this dataset has no object bounding boxes and the model outputs one class per crop.",
        "experiment_hypothesis": "A lower learning rate (1e-4) may improve validation generalization by making smaller updates than the 1e-3 baseline.",
        "settings": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "image_size": args.image_size, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "model": "SmallParkingCNN", "device": str(device)},
        "runs": run_results,
        "selected_run_by_validation_macro_f1": selected_name,
        "held_out_test_metrics": test_metrics,
        "limitations": [split_meta["limitation"], "Performance on a university parking lot is unverified without images from that site.",
                        "This classification project does not satisfy the original object-detection requirement as written."],
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Selected by validation:", selected_name)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print("Saved final checkpoint:", final_path)


if __name__ == "__main__":
    main()
