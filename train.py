import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

from dataset import get_video_paths, make_dataloader, split_dataset
from model import DenseNetBiLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DenseNet + BiLSTM model for deepfake video detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset folder containing real/ and fake/ directories")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating"):
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    return acc, precision


def save_results(path: str, results: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths, labels = get_video_paths(args.data_dir)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        video_paths, labels, random_state=args.seed
    )

    train_loader = make_dataloader(
        train_paths,
        train_labels,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_paths,
        val_labels,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_dataloader(
        test_paths,
        test_labels,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = DenseNetBiLSTM(num_classes=2, hidden_size=args.hidden_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_acc, val_precision = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val Precision: {val_precision:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # Load best model and evaluate on the test split.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, test_precision = evaluate(model, test_loader, device)

    results = {
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "best_validation_accuracy": float(best_val_acc),
    }

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

    save_results("results.json", results)


if __name__ == "__main__":
    main()
