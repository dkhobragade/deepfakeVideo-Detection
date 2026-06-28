import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def get_video_paths(data_dir: str) -> Tuple[List[str], List[int]]:
    """Collect video paths and labels from a dataset folder.

    Expected structure:
        data/
            real/vid1.mp4
            fake/vid2.mp4
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(
            "Expected 'real' and 'fake' folders inside the dataset directory."
        )

    video_paths = []
    labels = []

    for class_dir, label in [(real_dir, 0), (fake_dir, 1)]:
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(VIDEO_EXTENSIONS):
                    video_paths.append(os.path.join(root, file))
                    labels.append(label)

    if len(video_paths) == 0:
        raise ValueError(
            f"No video files found in {data_dir}. Expected .mp4, .avi, .mov, .mkv, or .webm files."
        )

    return video_paths, labels


def split_dataset(
    video_paths: List[str], labels: List[int], test_size: float = 0.15,
    val_size: float = 0.15, random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """Split the dataset into train/val/test using a 70/15/15 ratio."""
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must satisfy test_size + val_size < 1")

    # First split: keep 70% train + 15% val + 15% test.
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths,
        labels,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=labels,
    )

    # Second split: split remaining data into val and test.
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        stratify=temp_labels,
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def read_video_frames(video_path: str, num_frames: int = 16, resize: Tuple[int, int] = (224, 224)):
    """Read a fixed number of frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        else:
            # If a frame cannot be read, use the last successfully read frame.
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((*resize, 3), dtype=np.uint8))

    cap.release()
    frames = np.stack(frames, axis=0).astype(np.float32)
    frames = frames / 255.0
    return frames


class DeepfakeVideoDataset(Dataset):
    def __init__(self, video_paths: List[str], labels: List[int], num_frames: int = 16):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = read_video_frames(video_path, self.num_frames)

        # Convert to tensor: (T, C, H, W)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)
        label = torch.tensor(label, dtype=torch.long)
        return frames, label


def make_dataloader(
    video_paths: List[str], labels: List[int], batch_size: int = 4,
    num_frames: int = 16, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    dataset = DeepfakeVideoDataset(video_paths, labels, num_frames=num_frames)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
