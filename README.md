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



## Week 2: Video-Only Baseline Action Recognition

### Objective

Week 2 develops the first working action-recognition baseline using only visual information from video frames.

A pretrained ResNet18 model is used to extract visual features from each frame. The frame-level features are averaged to create one feature vector for each video, and a Logistic Regression classifier is trained to predict the action class.

This baseline provides a comparison point for the vision-language fusion model that will be developed during Week 4.

### Dataset summary

The project currently contains 9 action classes:

* Archery
* Basketball
* Biking
* Bowling
* Drumming
* JavelinThrow
* PlayingGuitar
* RopeClimbing
* Typing

The dataset contains:

```text
Training videos:   867
Validation videos: 197
Test videos:       197
Total videos:      1261
```

Each video is represented using:

```text
16 evenly sampled RGB frames
Frame size: 224 × 224 pixels
```

The total extracted-frame dataset contains:

```text
Training frames:   13872
Validation frames: 3152
Test frames:       3152
Total frames:      20176
```

All video folders contain exactly 16 frames.

The frame inspection process confirmed:

```text
Empty video folders: 0
Unreadable frames:   0
Classes per split:   9
```

### Visual feature extraction model

The project uses:

```text
ResNet18
```

ResNet18 is loaded using pretrained ImageNet weights from the `torchvision` library.

The original ResNet18 classification layer is removed so that the model returns visual features instead of ImageNet class predictions.

Each video frame is converted into a:

```text
512-dimensional visual feature vector
```

For one video containing 16 frames, the initial feature shape is:

```text
(16, 512)
```

The 16 frame features are averaged using mean pooling to create one final video feature.

The final representation of every video is:

```text
512-dimensional video feature vector
```

### Video feature pipeline

```text
Input video
    ↓
16 sampled frames
    ↓
Image preprocessing
    ↓
Pretrained ResNet18
    ↓
16 frame embeddings
    ↓
Mean pooling
    ↓
One video embedding
```

The final feature shape for one video is:

```text
(512,)
```

The generated feature data type is:

```text
float32
```

### Generated dataset shapes

The individual video features are combined into training, validation, and test arrays.

```text
Training features:   (867, 512)
Training labels:     (867,)

Validation features: (197, 512)
Validation labels:   (197,)

Test features:       (197, 512)
Test labels:         (197,)
```

Each row represents one video, and each row contains 512 visual features.

### Week 2 scripts

```text
scripts/check_environment.py
scripts/inspect_frames.py
scripts/extract_cnn_features.py
scripts/build_feature_dataset.py
scripts/train_baseline.py
scripts/evaluate_baseline.py
scripts/predict_video.py
```

#### `check_environment.py`

Checks:

* Python version
* Python executable path
* PyTorch installation
* Torchvision installation
* CUDA availability
* ResNet18 loading
* ResNet18 output feature shape

The verified environment was:

```text
Python:       3.12.10
PyTorch:      2.13.0+cpu
Torchvision:  0.28.0+cpu
Device:       CPU
```

The ResNet18 test produced:

```text
Input shape:  (1, 3, 224, 224)
Output shape: (1, 512)
```

#### `inspect_frames.py`

Checks:

* number of classes
* number of videos
* number of frames
* videos per class
* frames per video
* empty video folders
* unreadable frames
* class consistency across splits

The inspection confirmed that the extracted-frame dataset was ready for CNN feature extraction.

#### `extract_cnn_features.py`

Loads the extracted frames, preprocesses them using the ResNet18 image transforms, and generates one mean-pooled visual feature vector for every video.

The script processes:

```text
data/frames/train/
data/frames/val/
data/frames/test/
```

and saves the features in:

```text
data/features/train/
data/features/val/
data/features/test/
```

Example:

```text
Input:
data/frames/train/Archery/v_Archery_g03_c01/

Output:
data/features/train/Archery/v_Archery_g03_c01.npy
```

Feature extraction completed successfully for:

```text
Training videos:   867
Validation videos: 197
Test videos:       197
Failed videos:     0
```

#### `build_feature_dataset.py`

Combines the individual video feature files into complete NumPy datasets.

The script creates:

```text
train_features.npy
train_labels.npy
val_features.npy
val_labels.npy
test_features.npy
test_labels.npy
class_names.json
```

It also stores the video identifiers for every split.

#### `train_baseline.py`

Loads the training and validation features and trains the video-only baseline classifier.

The training pipeline uses:

```text
LabelEncoder
StandardScaler
Logistic Regression
```

The feature scaler is fitted only on the training set and then applied to the validation set.

The Logistic Regression model uses:

```text
Solver:        lbfgs
Maximum steps: 3000
Class weights: balanced
Random state:  42
```

#### `evaluate_baseline.py`

Loads the saved model, feature scaler, and label encoder.

The script evaluates the model on the untouched test set and calculates:

* accuracy
* macro precision
* macro recall
* macro F1-score
* weighted F1-score
* class-wise precision
* class-wise recall
* class-wise F1-score
* confusion matrix
* common misclassification pairs

#### `predict_video.py`

This script is reserved for predicting the action class of one new video.

It will reuse the trained model, scaler, label encoder, ResNet18 feature extractor, and mean-pooling process.

### Generated visual feature files

```text
data/features/
├── train/
│   ├── Archery/
│   ├── Basketball/
│   ├── Biking/
│   ├── Bowling/
│   ├── Drumming/
│   ├── JavelinThrow/
│   ├── PlayingGuitar/
│   ├── RopeClimbing/
│   ├── Typing/
│   └── metadata.json
│
├── val/
│   ├── Archery/
│   ├── Basketball/
│   ├── Biking/
│   ├── Bowling/
│   ├── Drumming/
│   ├── JavelinThrow/
│   ├── PlayingGuitar/
│   ├── RopeClimbing/
│   ├── Typing/
│   └── metadata.json
│
├── test/
│   ├── Archery/
│   ├── Basketball/
│   ├── Biking/
│   ├── Bowling/
│   ├── Drumming/
│   ├── JavelinThrow/
│   ├── PlayingGuitar/
│   ├── RopeClimbing/
│   ├── Typing/
│   └── metadata.json
│
├── train_features.npy
├── train_labels.npy
├── train_video_ids.json
├── val_features.npy
├── val_labels.npy
├── val_video_ids.json
├── test_features.npy
├── test_labels.npy
├── test_video_ids.json
└── class_names.json
```

### Saved model files

```text
models/
├── logistic_regression_baseline.joblib
├── feature_scaler.joblib
└── label_encoder.joblib
```

#### `logistic_regression_baseline.joblib`

Contains the trained Logistic Regression action classifier.

#### `feature_scaler.joblib`

Contains the fitted StandardScaler used to normalize the 512 visual features.

#### `label_encoder.joblib`

Contains the mapping between action-class names and numerical class labels.

### Generated reports and plots

```text
results/reports/validation_classification_report.txt
results/reports/validation_metrics.json
results/reports/test_classification_report.txt
results/reports/test_metrics.json
results/reports/test_misclassifications.json
results/plots/test_confusion_matrix.png
```

### Validation results

The validation set contained:

```text
197 videos
```

The baseline achieved:

```text
Validation accuracy: 0.9340
```

This is equivalent to:

```text
93.40%
```

The validation classification results were:

```text
               precision    recall    f1-score

Archery           1.0000    1.0000      1.0000
Basketball        0.7895    0.7500      0.7692
Biking            1.0000    1.0000      1.0000
Bowling           1.0000    1.0000      1.0000
Drumming          0.8667    1.0000      0.9286
JavelinThrow      0.9444    0.9444      0.9444
PlayingGuitar     1.0000    1.0000      1.0000
RopeClimbing      0.8182    1.0000      0.9000
Typing            1.0000    0.6316      0.7742
```

The overall validation scores were:

```text
Accuracy:      0.9340
Macro F1:      0.9240
Weighted F1:   0.9312
```

### Official test results

The untouched test set contained:

```text
197 videos
```

The official test metrics were:

```text
Accuracy:        0.9340
Macro precision: 0.9289
Macro recall:    0.9284
Macro F1-score:  0.9277
Weighted F1:     0.9341
```

This is equivalent to:

```text
Test accuracy:       93.40%
Macro precision:     92.89%
Macro recall:        92.84%
Macro F1-score:      92.77%
Weighted F1-score:   93.41%
```

### Test classification results

```text
               precision    recall    f1-score   support

Archery           0.9565    0.8800      0.9167        25
Basketball        0.7391    0.8500      0.7907        20
Biking            0.9524    1.0000      0.9756        20
Bowling           1.0000    1.0000      1.0000        23
Drumming          1.0000    1.0000      1.0000        28
JavelinThrow      0.8235    0.7368      0.7778        19
PlayingGuitar     1.0000    1.0000      1.0000        25
RopeClimbing      0.8889    0.8889      0.8889        18
Typing            1.0000    1.0000      1.0000        19
```

### Best-performing action classes

The following classes achieved a perfect test F1-score:

```text
Bowling        : 1.0000
Drumming       : 1.0000
PlayingGuitar  : 1.0000
Typing         : 1.0000
```

Biking also performed strongly:

```text
Biking: 0.9756
```

These classes contain strong visual cues.

For example:

* Bowling frequently includes a distinctive bowling lane.
* Drumming includes visible drums and repeated playing poses.
* PlayingGuitar includes a clearly recognizable musical instrument.
* Typing often includes a keyboard, desk, and seated human posture.
* Biking includes a visible bicycle and cycling posture.

### Difficult action classes

The lowest test F1-scores were:

```text
JavelinThrow: 0.7778
Basketball:   0.7907
RopeClimbing: 0.8889
```

These classes can share similar visual features such as:

* standing human poses
* raised arms
* sports environments
* large human figures
* similar body orientations

### Most common misclassifications

The most common incorrect prediction pairs were:

```text
JavelinThrow  -> Basketball:    4
Archery       -> JavelinThrow:  3
RopeClimbing  -> Basketball:    2
Basketball    -> Archery:       1
Basketball    -> Biking:        1
Basketball    -> RopeClimbing:  1
JavelinThrow  -> RopeClimbing:  1
```

The most frequent error was:

```text
JavelinThrow -> Basketball
```

This may occur because both actions can contain:

* raised-arm poses
* sports clothing
* outdoor or court-like backgrounds
* fast upper-body movement
* visually similar body alignment

The confusion between Archery and JavelinThrow is also reasonable because both actions can include:

* a standing person
* outdoor backgrounds
* long sports equipment
* side-facing body poses
* extended arms

### Baseline limitations

The Week 2 baseline performs well, but it has several limitations.

#### No explicit temporal modeling

The 16 frame features are averaged using mean pooling.

Mean pooling does not preserve:

* frame order
* motion direction
* action speed
* movement transitions
* temporal relationships

For example, throwing, climbing, and shooting actions may contain visually similar frames even though their movements occur in different sequences.

#### Visual information only

The model does not use any natural-language descriptions or semantic class information.

It learns only from:

* objects
* human poses
* backgrounds
* textures
* image composition

#### Static-image backbone

ResNet18 processes each frame independently.

It does not directly model the complete video as a temporal sequence.

#### Dataset-specific visual cues

Some classes may be recognized partly because of common objects or backgrounds instead of the action movement itself.

For example:

* bowling lanes may strongly identify Bowling
* musical instruments may identify Drumming and PlayingGuitar
* keyboards may identify Typing
* bicycles may identify Biking

### Run the Week 2 pipeline

Check the Python and PyTorch environment:

```powershell
python scripts/check_environment.py
```

Inspect the extracted-frame dataset:

```powershell
python scripts/inspect_frames.py
```

Extract training video features:

```powershell
python scripts/extract_cnn_features.py --split train
```

Extract validation video features:

```powershell
python scripts/extract_cnn_features.py --split val
```

Extract test video features:

```powershell
python scripts/extract_cnn_features.py --split test
```

Create the combined feature datasets:

```powershell
python scripts/build_feature_dataset.py
```

Train the Logistic Regression baseline:

```powershell
python scripts/train_baseline.py
```

Evaluate the trained model:

```powershell
python scripts/evaluate_baseline.py
```

### Week 2 outcome

Week 2 successfully produced a complete video-only action-recognition baseline.

The project now contains:

```text
Videos:                   1261
Frames:                   20176
Visual feature dimension: 512
Action classes:           9
Validation accuracy:      93.40%
Test accuracy:            93.40%
Test macro F1-score:      92.77%
```

The complete Week 2 pipeline is:

```text
Video
  ↓
16 extracted frames
  ↓
Pretrained ResNet18
  ↓
512-dimensional frame features
  ↓
Mean pooling
  ↓
512-dimensional video feature
  ↓
Feature scaling
  ↓
Logistic Regression
  ↓
Predicted action
```

This video-only model establishes the baseline that will be compared against the future vision-language fusion model.

During Week 3, natural-language action descriptions are converted into 384-dimensional semantic text embeddings.

The project therefore prepares the following representations for Week 4:

```text
Visual feature dimension: 512
Text feature dimension:   384
```

These visual and semantic features will be combined during Week 4 to build the vision-language fusion model.




## Week 3: NLP-Based Action Description Encoding

### Objective

Week 3 adds semantic language information to the action-recognition project. Natural-language descriptions of each action class are converted into numerical text embeddings using Sentence-BERT.

These embeddings will be combined with the visual video features during Week 4.

### Action descriptions

The project currently contains 9 action classes:

* Archery
* Basketball
* Biking
* Bowling
* Drumming
* JavelinThrow
* PlayingGuitar
* RopeClimbing
* Typing

Each class has 5 natural-language descriptions stored in:

```text
metadata/action_descriptions.json
```

The dataset therefore contains:

```text
9 classes × 5 descriptions = 45 descriptions
```

Example descriptions for `Biking`:

```text
A person is riding a bicycle.
A person is moving forward while pedaling a bike.
The action shows someone cycling.
A cyclist is pedaling a bicycle along a path.
Someone is traveling by riding a bike.
```

### Text embedding model

The project uses:

```text
all-MiniLM-L6-v2
```

This is a Sentence-BERT model provided through the `sentence-transformers` library.

Each action description is converted into a:

```text
384-dimensional text embedding
```

The five description embeddings for each class are averaged and normalized to produce one final class embedding.

### Generated embedding shapes

```text
Description embeddings: (45, 384)
Class embeddings:       (9, 384)
Data type:              float32
```

All generated embeddings are normalized to have a vector length of approximately `1.0`.

### Week 3 scripts

```text
scripts/encode_action_descriptions.py
scripts/validate_text_embeddings.py
scripts/analyze_text_similarity.py
scripts/plot_text_similarity.py
scripts/inspect_text_embeddings.py
```

#### `encode_action_descriptions.py`

Loads the action descriptions, generates Sentence-BERT embeddings, averages the descriptions for each class, and saves the final arrays.

#### `validate_text_embeddings.py`

Checks:

* embedding shapes
* data types
* invalid values
* normalization
* descriptions per class
* duplicate descriptions
* metadata consistency

#### `analyze_text_similarity.py`

Calculates cosine similarity between every action-class embedding and saves the semantic similarity results.

#### `plot_text_similarity.py`

Creates a heatmap showing semantic similarity between all action classes.

#### `inspect_text_embeddings.py`

Displays the descriptions, embedding statistics, and nearest semantic classes for a selected action.

### Generated embedding files

```text
data/text_embeddings/
├── class_embeddings.npy
├── class_names.json
├── description_embeddings.npy
├── description_metadata.json
└── embedding_metadata.json
```

### Generated reports and plots

```text
results/reports/text_similarity_matrix.csv
results/reports/text_similarity_matrix.npy
results/reports/text_similarity_report.txt
results/plots/text_similarity_heatmap.png
```

### Semantic similarity results

The most semantically similar action pairs were:

```text
Drumming       <-> PlayingGuitar : 0.5231
Basketball     <-> Bowling       : 0.4992
Archery        <-> JavelinThrow  : 0.4892
Archery        <-> Bowling       : 0.4820
Bowling        <-> JavelinThrow  : 0.4382
```

The least semantically similar action pairs included:

```text
JavelinThrow   <-> Typing        : 0.1454
Bowling        <-> Typing        : 0.1727
Archery        <-> Typing        : 0.2103
Drumming       <-> JavelinThrow  : 0.2105
Bowling        <-> Drumming      : 0.2195
```

These results are reasonable because actions involving related concepts receive higher similarity scores. For example, `Drumming` and `PlayingGuitar` both involve playing musical instruments, while `Typing` and `JavelinThrow` share very little semantic meaning.

### Run the Week 3 pipeline

Generate the embeddings:

```powershell
python scripts/encode_action_descriptions.py
```

Validate the generated files:

```powershell
python scripts/validate_text_embeddings.py
```

Calculate semantic similarity:

```powershell
python scripts/analyze_text_similarity.py
```

Generate the similarity heatmap:

```powershell
python scripts/plot_text_similarity.py
```

Inspect one class:

```powershell
python scripts/inspect_text_embeddings.py --class-name Archery --top-k 5
```

Another example:

```powershell
python scripts/inspect_text_embeddings.py --class-name PlayingGuitar --top-k 3
```

### Week 3 outcome

Week 3 successfully produced a semantic representation for every action class.

The project now contains:

```text
Visual feature dimension: 512
Text feature dimension:   384
```

These visual and text features will be combined during Week 4 to develop the vision-language fusion model.



---

# Week 4: Vision-Language Fusion

## Objective

Week 4 explores how semantic language information can be combined with visual features for human action recognition.

The primary goals are to:

- Combine CNN visual features with Sentence-BERT text embeddings.
- Explore simple multimodal fusion techniques.
- Demonstrate why naïve text fusion introduces label leakage.
- Develop a leakage-safe visual-to-semantic projection model.
- Compare all approaches using a comprehensive ablation study.

This week bridges computer vision and natural language processing while emphasizing proper experimental design.

---

## Week 4 Workflow

```text
                 ORACLE EXPERIMENTS (Label Leakage)

Visual Feature (512)
        │
        ├─────────────┐
        │             │
        ▼             ▼
Ground-truth     Sentence-BERT
Class Label      Text Embedding (384)
        │             │
        └──────┬──────┘
               ▼
     Concatenated Feature (896)
               │
               ▼
 Logistic Regression Classifier


         LEAKAGE-SAFE EXPERIMENT

Video
   │
   ▼
ResNet18
   │
   ▼
Visual Feature (512)
   │
   ▼
Projection Network
   │
   ▼
Predicted Semantic Embedding (384)
   │
   ▼
Cosine Similarity
   │
   ▼
Class Text Embeddings
   │
   ▼
Predicted Action
```

---

# Oracle Fusion Experiments

Two oracle experiments were implemented to understand the effect of adding semantic information.

### Text-only Oracle Baseline

The classifier receives only the Sentence-BERT class embedding.

Pipeline:

```text
Ground-truth Label
      │
Sentence-BERT
      │
384-d embedding
      │
Logistic Regression
```

Since the correct class embedding is directly provided, this experiment contains complete label leakage.

---

### Oracle Concatenation

Visual features and the ground-truth text embedding are concatenated.

```text
512-d Visual Feature
          +
384-d Text Embedding
          │
          ▼
896-d Fusion Feature
          │
          ▼
Logistic Regression
```

This experiment also contains label leakage because the true class embedding is supplied during inference.

---

# Leakage-Safe Semantic Projection

To remove label leakage, a projection network was trained.

Instead of using the true class embedding during inference, the network predicts the semantic embedding directly from visual features.

Pipeline:

```text
Video
   │
ResNet18
   │
512-d Visual Feature
   │
Projection Network
   │
384-d Predicted Semantic Embedding
   │
Cosine Similarity
   │
Sentence-BERT Class Embeddings
   │
Predicted Action
```

During inference the model receives only visual information.

No ground-truth labels or text embeddings are provided.

---

# Experimental Results

| Model | Feature Dimension | Label Leakage | Test Accuracy | Macro F1 |
|------|------------------:|:-------------:|-------------:|----------:|
| Video-only Baseline | 512 | No | **0.9340** | **0.9277** |
| Text-only Oracle | 384 | Yes | **1.0000** | **1.0000** |
| Oracle Concatenation | 896 | Yes | **1.0000** | **1.0000** |
| Leakage-safe Semantic Projection | 384 | No | **0.8782** | **0.8639** |

---

# Ablation Study

The oracle experiments achieved perfect performance because they used the correct class embedding during inference.

These results should **not** be interpreted as genuine improvements.

The leakage-safe semantic projection represents the valid deployment scenario because it predicts semantic information from visual features alone.

Although the projection model did not outperform the original visual baseline, it successfully demonstrated semantic alignment without introducing label leakage.

---

# Class-wise Analysis

Comparison between the Video-only Baseline and the Leakage-safe Semantic Projection:

| Class | Result |
|------|--------|
| Bowling | Unchanged |
| PlayingGuitar | Unchanged |
| Archery | Lower F1 |
| Basketball | Lower F1 |
| Biking | Lower F1 |
| Drumming | Lower F1 |
| JavelinThrow | Lower F1 |
| RopeClimbing | Lower F1 |
| Typing | Lower F1 |

Summary:

- Improved classes: **0**
- Unchanged classes: **2**
- Lower-performing classes: **7**

The largest performance decreases were observed for:

- Basketball
- Archery
- Typing

---

# Semantic Prediction Pipeline

A complete inference pipeline was implemented.

The system accepts either:

- a raw video file (`.avi`, `.mp4`, `.mov`, `.mkv`)
- or an extracted frame directory.

Prediction workflow:

```text
Input Video
      │
Frame Sampling
      │
ResNet18
      │
Visual Feature
      │
Projection Network
      │
Semantic Embedding
      │
Cosine Similarity
      │
Top-K Predictions
```

The prediction script outputs:

- Predicted action
- Top-K predicted classes
- Cosine similarity scores

No ground-truth labels are used during inference.

---

# Scripts Added

```text
scripts/
│
├── build_fusion_dataset.py
├── validate_fusion_dataset.py
├── train_text_baseline.py
├── evaluate_text_baseline.py
├── train_fusion_classifier.py
├── evaluate_fusion_classifier.py
├── train_semantic_projection.py
├── evaluate_semantic_projection.py
├── predict_video_semantic.py
├── compare_models.py
└── analyze_classwise_results.py
```

---

# Generated Outputs

```text
data/
└── fusion_features/

models/
├── oracle_fusion_logistic_regression.joblib
├── semantic_projection_model.pth
└── semantic_projection_scaler.joblib

results/
├── reports/
│   ├── oracle_fusion_test_report.txt
│   ├── semantic_projection_test_report.txt
│   ├── week4_ablation_report.txt
│   └── week4_classwise_comparison.txt
│
└── plots/
    ├── oracle_fusion_test_confusion_matrix.png
    └── semantic_projection_test_confusion_matrix.png
```

---

# Key Learnings

- Simple concatenation of visual and text features can introduce severe label leakage.
- Oracle experiments are useful for understanding upper-bound performance but cannot be deployed.
- Leakage-safe semantic projection provides a valid multimodal inference strategy.
- Proper experimental design is more important than achieving artificially high accuracy.
- Vision-language integration forms the foundation for more advanced multimodal action recognition systems.

---

## Week 4 Status

**Status:** ✅ Completed

Major accomplishments:

- Oracle fusion experiments completed.
- Leakage-safe semantic projection implemented.
- End-to-end semantic prediction pipeline created.
- Ablation study completed.
- Class-wise analysis completed.
- Raw video and frame-folder inference verified successfully.