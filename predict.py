import argparse
import os

import torch

from dataset import read_video_frames
from model import DenseNetBiLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction on a single video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--num_frames", type=int, default=16)
    return parser.parse_args()



def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetBiLSTM(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    frames = read_video_frames(args.video_path, num_frames=args.num_frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)
        pred = torch.argmax(outputs, dim=1).item()

    label = "real" if pred == 0 else "fake"
    print(f"Predicted class: {label} (class index: {pred})")


if __name__ == "__main__":
    main()
