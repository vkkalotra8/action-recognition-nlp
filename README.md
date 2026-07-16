# Semantic Prompt-Guided Human Action Recognition

## Project Overview

This project develops a human action recognition system using both video data and natural-language action descriptions.

The first version uses a selected subset of the UCF101 dataset. Video frames are extracted and later used with visual models such as CNNs. Natural-language action descriptions will later be encoded using Sentence-BERT or BERT.

The final system will predict:

- Action class
- Confidence score
- Natural-language explanation

## Dataset

Dataset used:

- UCF101

Number of selected classes:

- 9 classes

## Selected Action Classes

1. Archery
2. Basketball
3. Biking
4. Bowling
5. Drumming
6. JavelinThrow
7. PlayingGuitar
8. RopeClimbing
9. Typing

## Week 1 Objectives

Week 1 focuses on:

- Selecting a subset of UCF101
- Organizing the project folders
- Checking all videos
- Creating train, validation, and test splits
- Extracting video frames
- Verifying extracted frames
- Preparing natural-language action descriptions

## Dataset Split

The dataset is divided at the video level into:

- Training: 70%
- Validation: 15%
- Testing: 15%

Splitting is performed before frame extraction so frames from the same video do not appear in multiple data splits.

## Frame Extraction

For each video:

- 16 evenly spaced frames are extracted
- Each frame is resized to 224 x 224 pixels
- Frames are saved as JPG files

Example:

```text
data/frames/train/Biking/v_Biking_g01_c01/
├── frame_000.jpg
├── frame_001.jpg
├── frame_002.jpg
├── ...
└── frame_015.jpg