# Semantic Prompt-Guided Human Action Recognition

This project combines video-based human action recognition with natural-language descriptions of actions. The initial system predicts an action from video frames; later stages add text embeddings, vision-language fusion, and natural-language explanations.

## Dataset

The first version uses a 9-class subset of UCF101:

1. Archery
2. Basketball
3. Biking
4. Bowling
5. Drumming
6. JavelinThrow
7. PlayingGuitar
8. RopeClimbing
9. Typing

Each video is represented by 16 evenly spaced, 224 x 224 JPEG frames.

## Split policy

The data is split approximately 70% / 15% / 15% into training, validation, and test sets. UCF101 clips with the same class and recording group (for example, `v_Archery_g01_c01` and `v_Archery_g01_c02`) are always assigned to the same partition. This prevents recording-group leakage and makes evaluation reliable.

## Setup

Prerequisites:

- Windows and Python 3.10 or later
- Git

Create and activate a virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements use the official CPU-only PyTorch wheel index and PyTorch 2.11, which supports Python 3.14 experimentally. This is the appropriate default for this machine because no NVIDIA CUDA tooling was detected.

Confirm the deep-learning environment:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Reproducing the Week 1 dataset preparation

Place the selected UCF101 videos in `data/raw_videos/<class_name>/`. The required class names are listed in `metadata/classes.txt`.

Then run the following commands from the repository root:

```powershell
python scripts/check_dataset.py
python scripts/create_splits.py
python scripts/extract_frames.py
python scripts/verify_frames.py
python scripts/check_descriptions.py
python scripts/check_split_leakage.py
```

`create_splits.py` and `extract_frames.py` recreate their derived output directories. Do not place manual files in `data/splits/` or `data/frames/`.

## Project layout

```text
app/        Streamlit application (Week 6)
data/       Raw videos, generated splits, and extracted frames
metadata/   Class list, action descriptions, and split manifest
models/     Saved models
notebooks/  Exploratory notebooks
reports/    Results and documentation
scripts/    Data preparation and validation scripts
```

## Current status

Week 1 is complete: the dataset, group-safe splits, frame extraction, and action descriptions have been verified. Week 2 will implement a video-only PyTorch baseline using pretrained visual features.
