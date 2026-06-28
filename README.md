# Deepfake Video Detection

This project implements a complete deepfake video detection pipeline using a **DenseNet backbone + BiLSTM** temporal model.

## Dataset layout
Place your dataset in the following structure:

```text
data/
  real/
    video_001.mp4
    video_002.mp4
  fake/
    video_101.mp4
    video_102.mp4
```

The code automatically splits the videos into:
- 70% training
- 15% validation
- 15% testing

## Training command

```bash
python train.py --data_dir data --epochs 10 --batch_size 4 --num_frames 16
```

## Output
The script prints:
- training loss
- validation accuracy
- test accuracy
- test precision

It also saves the best model to `best_model.pth`.
