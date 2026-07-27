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


---


# Week 5: Explainable AI for Semantic Action Recognition

## Objective

The previous weeks focused on building an accurate semantic action recognition system.

- **Week 1** prepared the dataset and extracted video frames.
- **Week 2** built a visual-only action recognition baseline.
- **Week 3** introduced semantic action representations using Sentence-BERT.
- **Week 4** combined visual and semantic information through a leakage-safe projection model.

Although the system could now recognize human actions with high accuracy, it still behaved like a **black box**.

For example, if the model predicted:

```text
Drumming
```

there was no information explaining:

- Why was Drumming selected?
- Which other actions looked similar?
- How confident was the prediction?
- Was the decision obvious or ambiguous?
- Which frames were actually used?

Week 5 addresses this limitation by transforming the prediction pipeline into an **Explainable AI (XAI)** system.

Instead of producing only an action label, the system now generates an interpretable prediction report that includes:

- predicted action
- semantic similarity score
- confidence level
- score separation
- representative natural-language description
- explanation of the prediction
- reliability note
- Top-K alternative predictions
- sampled inference frames
- contact sheet visualization
- machine-readable JSON report

The goal of Week 5 is not to improve classification accuracy.

Instead, the objective is to improve the **interpretability, transparency, and usability** of the semantic action recognition system.

---

# Why Explainability Matters

Traditional machine learning systems usually produce outputs similar to this:

```text
Prediction:
Drumming
```

Although this prediction may be correct, it does not answer an important question:

> **Why did the model reach this decision?**

This lack of transparency creates several problems.

For example:

- incorrect predictions become difficult to analyze
- users cannot estimate prediction reliability
- similar action classes remain unexplained
- debugging the model becomes harder
- demonstrating the project becomes less convincing

Consider two different outputs.

Without explainability:

```text
Prediction:
PlayingGuitar
```

With explainability:

```text
Prediction:
PlayingGuitar

Similarity:
0.84

Confidence:
High

Second Best:
Drumming (0.42)

Explanation:

The semantic representation extracted from the video is much closer to
PlayingGuitar than every other action class.

The score difference between the first and second prediction indicates
that the decision is reliable.
```

The second output immediately provides useful information.

A human observer can understand:

- what the model predicted
- how confident it is
- which alternative actions were considered
- whether the prediction should be trusted

This is the primary objective of Explainable AI.

---

# Explainable AI (XAI)

Explainable AI refers to machine learning systems that provide understandable reasons behind their predictions.

Instead of behaving like a black box,

```text
Input
↓

Model

↓

Prediction
```

an explainable model exposes part of its reasoning process.

```text
Input
↓

Model

↓

Prediction

↓

Explanation

↓

Confidence

↓

Alternative Predictions
```

Modern AI applications increasingly require explainability.

Examples include:

- healthcare diagnosis
- autonomous driving
- finance
- security systems
- surveillance
- recommendation systems

Human action recognition is no exception.

If a surveillance system predicts:

```text
Running
```

the user naturally wants to know:

- Was the prediction reliable?
- Which other actions looked similar?
- Was the model uncertain?

Providing this information greatly increases user trust.

---

# Explainability in This Project

This project uses a semantic prediction model rather than a traditional classifier.

The model does **not** output probabilities directly.

Instead, it predicts a semantic embedding.

That semantic embedding is compared against every action class embedding using cosine similarity.

```text
Video

↓

Visual Features

↓

Semantic Projection Model

↓

Predicted Semantic Embedding

↓

Cosine Similarity

↓

All Class Embeddings

↓

Best Matching Action
```

Because similarity scores are available for every class, they naturally provide an explanation of the decision.

Rather than hiding intermediate information, Week 5 exposes it to the user.

---

# Week 5 Workflow

```text
                    INPUT VIDEO
                         │
                         ▼
                 Sample Video Frames
                         │
                         ▼
               ResNet18 Feature Extractor
                         │
                         ▼
                Mean Video Representation
                         │
                         ▼
          Semantic Projection Model (Week 4)
                         │
                         ▼
          Predicted Semantic Embedding (384)
                         │
                         ▼
      Cosine Similarity Against Every Action Class
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
  Top-K Predictions   Confidence     Score Gap
         │               │                │
         └───────────────┼────────────────┘
                         │
                         ▼
              Explanation Generation Engine
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
 Representative     Reliability      Natural Language
 Description            Note          Explanation
         │
         ▼
     Prediction Report
         │
         ▼
 JSON Report + Text Report + Contact Sheet
```

---

# Overall Week 5 Architecture

The explainability module is built on top of the semantic projection model developed during Week 4.

Unlike previous weeks, no new classifier is trained.

Instead, Week 5 enhances the **inference stage**.

```text
                   TRAINING
──────────────────────────────────────────────

Visual Features
        │
        ▼
Semantic Projection Model
        │
        ▼
Semantic Embedding

──────────────────────────────────────────────

                  INFERENCE

Input Video
      │
      ▼
Frame Sampling
      │
      ▼
Visual Feature Extraction
      │
      ▼
Semantic Projection Model
      │
      ▼
Similarity Scores
      │
      ▼
Explainability Engine
      │
      ▼
Prediction Report
```

This design has an important advantage.

The explainability module is completely independent from the training process.

If the semantic projection model is replaced in the future with a larger architecture (such as Video Swin Transformer or CLIP), the explanation engine can still be reused with minimal modifications.

---

# Components Added During Week 5

The following major components were implemented during this week.

| Component | Purpose |
|-----------|---------|
| Explanation Engine | Generates human-readable explanations from similarity scores |
| Explainable Prediction Script | Performs inference while generating explanations |
| Confidence Estimator | Converts similarity values into confidence levels |
| Score Separation Analyzer | Measures how distinct the best prediction is |
| Top-K Ranking Module | Displays the most similar action classes |
| Prediction Report Generator | Creates text and JSON reports |
| Frame Saving Utility | Stores the exact frames used during inference |
| Contact Sheet Generator | Creates a single image containing all sampled frames |
| Visual Evidence Metadata | Records saved frame locations inside the JSON report |

Together, these components transform the semantic action recognition model into an interpretable prediction system suitable for demonstrations, debugging, and future deployment.

---

# Learning Outcomes

After completing Week 5, the project is capable of answering questions such as:

- What action was predicted?
- Why was this action selected?
- Which other actions were considered?
- How similar were those actions?
- Is the prediction reliable?
- Was the decision ambiguous?
- Which frames influenced the prediction?
- Can another program read the prediction report?

These capabilities significantly improve the interpretability of the project while keeping the underlying semantic recognition model unchanged.

Week 5 therefore focuses on **understanding model decisions**, rather than improving classification accuracy.


# Explanation Engine

The central component introduced during Week 5 is the **Explanation Engine**.

The semantic projection model developed during Week 4 is capable of predicting a semantic embedding for an input video.

However, the projection model itself only produces numerical values.

These numerical values are meaningful for machine learning algorithms but are not easy for humans to interpret.

The role of the Explanation Engine is to convert these numerical outputs into understandable natural-language explanations.

Instead of returning only:

```text
Prediction:
Drumming
```

the system now produces information such as:

```text
Prediction:
Drumming

Similarity:
0.8776

Confidence:
High

Top Alternative:
Biking

Explanation:

The projected semantic representation of the video was most
similar to the Drumming class embedding. Its similarity score
was clearly higher than the next-best class, indicating a
confident prediction.

Reliability Note:

These values are cosine similarity scores rather than calibrated
probabilities.
```

This additional information allows users to understand not only
**what** the model predicted but also **why** it reached that decision.

---

# Why an Explanation Engine is Needed

Machine learning models generally produce numerical outputs.

For example, a classifier may internally calculate:

```text
Drumming        0.8776
Biking          0.3373
PlayingGuitar   0.3058
RopeClimbing    0.2127
Basketball      0.1657
```

While these numbers are useful to a computer, they do not immediately answer questions such as:

- Why is Drumming the prediction?
- Is the model confident?
- Is another class almost equally likely?
- Should this prediction be trusted?

The Explanation Engine interprets these values and converts them into meaningful sentences.

Its objective is to bridge the gap between numerical model outputs and human understanding.

---

# Position of the Explanation Engine

The explanation module is placed **after semantic prediction**.

```text
Input Video
      │
      ▼
Visual Features
      │
      ▼
Semantic Projection Model
      │
      ▼
Predicted Semantic Embedding
      │
      ▼
Cosine Similarity Scores
      │
      ▼
Explanation Engine
      │
      ▼
Prediction Report
```

Notice that the Explanation Engine does **not** affect the prediction itself.

Instead, it interprets the prediction after inference has already completed.

This separation keeps the prediction pipeline simple while making the output much easier to understand.

---

# Cosine Similarity

The semantic projection model predicts a **384-dimensional semantic embedding**.

Each action class already has its own semantic embedding generated during Week 3 using Sentence-BERT.

For every prediction, the model compares the projected video embedding with every class embedding using cosine similarity.

```text
Projected Video Embedding

        │

        ├────────► Archery
        │
        ├────────► Basketball
        │
        ├────────► Biking
        │
        ├────────► Bowling
        │
        ├────────► Drumming
        │
        ├────────► JavelinThrow
        │
        ├────────► PlayingGuitar
        │
        ├────────► RopeClimbing
        │
        └────────► Typing
```

Each comparison produces one similarity score.

Example:

```text
Archery          0.104
Basketball       0.166
Biking           0.337
Bowling          0.081
Drumming         0.878
PlayingGuitar    0.306
Typing           0.049
```

The class with the highest similarity becomes the prediction.

---

# Why Cosine Similarity?

Cosine similarity measures the angle between two vectors.

Unlike Euclidean distance, cosine similarity focuses on the direction of the vectors rather than their magnitude.

```text
          Vector A

             /

            /

           /

----------/----------------

         /

        /

Vector B
```

If two vectors point in almost the same direction,

their cosine similarity approaches:

```text
1.0
```

If they point in unrelated directions,

their similarity approaches:

```text
0.0
```

If they point in opposite directions,

their similarity becomes:

```text
-1.0
```

Because both the projected embeddings and class embeddings are L2-normalized,

their cosine similarity directly represents semantic closeness.

---

# Why Similarity is NOT a Probability

One of the most common misconceptions is treating cosine similarity as a probability.

For example:

```text
Similarity

0.91
```

does **not** mean

```text
91% confidence
```

These values represent geometric similarity in embedding space.

They are **not calibrated probabilities**.

Consider this example:

```text
Drumming          0.88
Biking            0.34
PlayingGuitar     0.31
```

The model is saying:

> "The semantic embedding of this video is closest to the
> Drumming embedding."

It is **not** saying:

> "There is an 88% chance that this video contains drumming."

This distinction is extremely important.

For that reason, every prediction report includes a reliability note reminding the user that cosine similarity values should not be interpreted as probabilities.

---

# Confidence Estimation

Although cosine similarity is not a probability,

it still provides useful information.

Larger similarity values generally indicate that the projected semantic embedding is closer to one particular class.

Week 5 therefore converts similarity values into simple confidence levels.

Example:

```text
Similarity

0.90

↓

Confidence

High
```

Another example:

```text
Similarity

0.54

↓

Confidence

Moderate
```

And:

```text
Similarity

0.18

↓

Confidence

Low
```

These confidence labels make the prediction easier for non-technical users to interpret.

---

# Why Confidence Levels are Useful

Most users are unfamiliar with cosine similarity.

Compare these two outputs.

Without interpretation:

```text
Similarity

0.81
```

With interpretation:

```text
Similarity

0.81

Confidence

High
```

The second output communicates the prediction quality much more clearly.

Confidence levels therefore improve the readability of prediction reports without changing the underlying model.

---

# Score Separation

Similarity alone is not always sufficient.

Consider two predictions.

Example 1:

```text
Drumming

0.88
```

Second place:

```text
Biking

0.34
```

The difference is:

```text
0.54
```

This is a strong separation.

Now consider another prediction.

```text
PlayingGuitar

0.61
```

Second place:

```text
Drumming

0.60
```

Difference:

```text
0.01
```

Although PlayingGuitar still has the highest similarity,

the model is clearly uncertain because another class is almost identical.

Week 5 therefore calculates the gap between the highest similarity score and the second-highest similarity score.

This value is called the **score separation**.

---

# Separation Levels

The score gap is translated into qualitative descriptions.

Examples include:

```text
Large gap

↓

Clear separation
```

```text
Medium gap

↓

Moderate separation
```

```text
Small gap

↓

Ambiguous prediction
```

These descriptions help users understand whether the model made an obvious decision or whether multiple actions looked equally similar.

---

# Reliability Notes

Every prediction report also includes a reliability note.

Example:

```text
These values are cosine similarity scores,
not calibrated probabilities.

The top result has a reasonably distinct
semantic advantage over the next candidate.
```

The purpose of this section is to prevent incorrect interpretation of the similarity values.

Instead of simply presenting numbers,

the report explains what those numbers actually mean.

---

# Natural-Language Explanation Generation

The Explanation Engine automatically generates complete English sentences.

Instead of displaying only numerical statistics,

it produces explanations such as:

```text
The projected semantic representation of the
video was most similar to the Drumming class
embedding.

Its similarity score of 0.8776 was clearly
higher than the next-best class, Biking,
which scored 0.3373.
```

These explanations are dynamically generated using the prediction results.

Different sentence templates are selected depending on the confidence level and score separation.

As a result, each prediction receives a customized explanation rather than a fixed text block.

---

# Decision Flow of the Explanation Engine

The explanation generation process follows the workflow below.

```text
Similarity Scores
        │
        ▼
Rank Classes
        │
        ▼
Calculate Top Score
        │
        ▼
Calculate Score Gap
        │
        ▼
Determine Confidence
        │
        ▼
Determine Separation Level
        │
        ▼
Generate Natural-Language Explanation
        │
        ▼
Generate Reliability Note
        │
        ▼
Create Prediction Report
```

This workflow transforms raw numerical outputs into a structured explanation that is easy for both technical and non-technical users to understand.

---

# Summary

The Explanation Engine is the core component introduced during Week 5.

Rather than modifying the semantic prediction model itself, it enhances the inference process by translating similarity scores into meaningful human-readable explanations.

By combining similarity interpretation, confidence estimation, score separation analysis, and automatically generated explanations, the project evolves from a traditional prediction system into an explainable AI application.

This makes the semantic action recognition model easier to debug, easier to demonstrate, and significantly more transparent for end users.


# Explainable Prediction Pipeline

After building the Explanation Engine, the next step is to integrate it into the complete inference pipeline.

Instead of producing only a predicted action class, the prediction script now generates a complete explainable prediction report.

The inference process begins with either a raw video file or a folder containing extracted image frames.

The video is first converted into a fixed number of representative frames.

These sampled frames are processed by the ResNet18 feature extractor to generate visual features.

The visual features are then projected into the semantic embedding space using the leakage-safe semantic projection model developed during Week 4.

Finally, cosine similarity is calculated between the projected video embedding and every action class embedding.

The Explanation Engine interprets these similarity scores and generates a complete prediction report.

---

# Explainable Prediction Workflow

```text
Input Video / Frame Folder
            │
            ▼
     Frame Sampling
            │
            ▼
 ResNet18 Feature Extractor
            │
            ▼
 Mean Video Feature
            │
            ▼
 Semantic Projection Model
            │
            ▼
 Projected Semantic Embedding
            │
            ▼
 Cosine Similarity
            │
            ▼
 Rank All Classes
            │
            ▼
 Explanation Engine
            │
            ▼
 Prediction Report
            │
            ├────────► Console Output
            ├────────► Text Report
            ├────────► JSON Report
            ├────────► Saved Frames
            └────────► Contact Sheet
```

---

# Explainable Prediction Outputs

Every prediction now produces significantly more information than previous weeks.

The output includes:

- Predicted action class
- Top-K ranked predictions
- Cosine similarity scores
- Confidence level
- Score separation
- Representative action description
- Natural-language explanation
- Reliability note
- Sampled inference frames
- Contact sheet visualization
- Machine-readable JSON report

Together, these outputs make the prediction process much easier to understand and analyze.

---

# Prediction Reports

Week 5 automatically generates two prediction reports.

## Text Report

The text report is designed for human reading.

It summarizes the prediction in a clean and organized format.

Example:

```text
Predicted Action

Drumming

Similarity

0.8776

Confidence

High

Explanation

The projected semantic representation of the video
was most similar to the Drumming class embedding.

Reliability

These values are cosine similarity scores rather
than calibrated probabilities.
```

---

## JSON Report

The JSON report stores the same information in a structured format.

Unlike the text report, it is intended for software applications.

The JSON report contains information such as:

```text
Predicted class

Top predictions

Similarity scores

Confidence level

Score gap

Representative description

Explanation

Reliability note

Visual evidence
```

This makes it possible for future applications such as the Streamlit interface to display prediction results without parsing text files.

---

# Top-K Predictions

Instead of displaying only the best prediction, the system also displays the Top-K most similar action classes.

Example:

```text
1. Drumming         0.8776
2. Biking           0.3373
3. PlayingGuitar    0.3058
4. RopeClimbing     0.2127
5. Basketball       0.1657
```

Displaying alternative predictions helps users understand which actions appeared visually or semantically similar.

---

# Representative Action Description

Each action class already contains multiple natural-language descriptions created during Week 3.

Week 5 includes one representative description inside the prediction report.

Example:

```text
Predicted Action

Drumming

Representative Description

A person is playing a drum using drumsticks.
```

This connects the semantic prediction with a human-readable explanation.

---

# Visual Evidence

Understanding why a prediction was made is easier when users can inspect the actual frames used during inference.

For this reason, Week 5 can optionally save all sampled frames.

Example:

```text
results/
└── predictions/
    └── difficult_drumming_test_frames/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        ├── ...
        └── frame_0016.jpg
```

These saved images represent the visual evidence used by the model.

---

# Contact Sheet Generation

Inspecting many individual frame images can be inconvenient.

Week 5 therefore generates a contact sheet.

A contact sheet combines all sampled frames into a single image.

Example:

```text
+---------+---------+---------+---------+
| Frame 1 | Frame 2 | Frame 3 | Frame 4 |
+---------+---------+---------+---------+
| Frame 5 | Frame 6 | Frame 7 | Frame 8 |
+---------+---------+---------+---------+
| Frame 9 | Frame10 | Frame11 | Frame12 |
+---------+---------+---------+---------+
| Frame13 | Frame14 | Frame15 | Frame16 |
+---------+---------+---------+---------+
```

This visualization provides a quick overview of the entire inference process.

---

# Files Generated During Week 5

Typical prediction outputs include:

```text
results/
└── predictions/
    ├── difficult_drumming_prediction.txt
    ├── difficult_drumming_prediction.json
    ├── difficult_drumming_contact_sheet.jpg
    └── difficult_drumming_frames/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        ├── ...
        └── frame_0016.jpg
```

These files provide both human-readable and machine-readable prediction results.

---

# Grad-CAM

The original roadmap suggested Grad-CAM as an optional explainability technique.

Grad-CAM highlights image regions that contribute most to a CNN prediction.

However, the final semantic projection model performs prediction in a semantic embedding space rather than directly through a CNN classifier.

Since the objective of this project is semantic explainability rather than spatial attention visualization, Grad-CAM was not included in the final implementation.

Instead, explainability is provided through semantic similarity analysis, confidence estimation, Top-K predictions, representative descriptions, and visual evidence.

---

# Folder Structure After Week 5

```text
results/
│
├── predictions/
│   ├── *.txt
│   ├── *.json
│   ├── *_contact_sheet.jpg
│   └── *_frames/
│
├── reports/
│
└── plots/
```

This organization separates prediction outputs from evaluation reports and visualization results.

---

# Week 5 Summary

Week 5 transforms the semantic action recognition model into an explainable AI system.

Instead of returning only an action label, the system now explains its prediction using semantic similarity information, confidence estimation, score separation, representative descriptions, Top-K predictions, and visual evidence.

The explainability module operates entirely during inference and does not modify the underlying semantic projection model.

As a result, the prediction process becomes significantly more transparent, easier to interpret, and more suitable for demonstrations and future deployment.

---

# Learning Outcomes

After completing Week 5, the project is capable of:

- Predicting actions from videos or frame folders.
- Generating natural-language explanations for predictions.
- Displaying Top-K alternative action classes.
- Estimating confidence using semantic similarity.
- Measuring prediction ambiguity using score separation.
- Saving structured JSON and text reports.
- Preserving sampled inference frames as visual evidence.
- Generating contact sheets for qualitative analysis.
- Producing an explainable prediction pipeline suitable for integration into a user interface.

With the completion of Week 5, the project has evolved from a semantic action recognition model into a fully explainable vision-language recognition system.

The remaining work in Week 6 focuses on deployment through a Streamlit application, final documentation, presentation preparation, and GitHub packaging.
```


# Week 6: Streamlit Deployment, Documentation and Final Project

## Objective

The final week focuses on transforming the complete machine learning pipeline into a polished, user-friendly application. Instead of executing multiple Python scripts manually, the entire workflow is integrated into a single interactive interface using **Streamlit**.

By the end of this week, the project becomes a complete demonstration of an end-to-end Vision-Language Action Recognition system capable of:

- Accepting a new video or extracted frame folder as input.
- Running the complete semantic action recognition pipeline.
- Displaying the predicted action.
- Showing semantic similarity scores.
- Generating a natural-language explanation.
- Presenting confidence and reliability information.
- Saving prediction reports.
- Organizing the project for GitHub publication.

This week transforms the project from a collection of scripts into a deployable application that can easily be demonstrated, shared, or extended in future research.

---

# Week 6 Workflow

```text
                    USER

                      │
                      ▼

              Upload Video / Frames

                      │
                      ▼

           Streamlit User Interface

                      │
                      ▼

        Frame Extraction (if required)

                      │
                      ▼

       ResNet18 Feature Extraction (512-D)

                      │
                      ▼

      Semantic Projection Model (512 → 384)

                      │
                      ▼

      Cosine Similarity with Class Embeddings

                      │
                      ▼

        Rank All Candidate Action Classes

                      │
                      ▼

         Explainability Engine (Week 5)

                      │
        ┌─────────────┼──────────────┐
        │             │              │
        ▼             ▼              ▼

 Predicted Class   Top-k Results   Confidence

        │             │              │
        └─────────────┼──────────────┘
                      │
                      ▼

      Natural Language Explanation

                      │
                      ▼

      Save Reports + Display Results
```

---

# Why Streamlit?

During previous weeks every task required executing individual Python scripts from the command line.

For example:

```text
python train_projection_model.py

python predict_video_semantic.py

python predict_video_explainable.py
```

Although this workflow is ideal during development, it is not convenient for demonstrations or non-technical users.

Streamlit provides a lightweight web application framework that allows the entire inference pipeline to be executed through a browser.

Instead of typing commands, the user only needs to:

1. Open the application.
2. Upload a video.
3. Click Predict.
4. View the results.

No knowledge of Python is required.

---

# Complete Project Architecture

```text
                    INPUT VIDEO
                          │
                          ▼
                Frame Sampling Module
                          │
                          ▼
            ResNet18 Feature Extraction
                          │
                  512-D Visual Feature
                          │
                          ▼
          Semantic Projection Network
                 (Week 4 Model)
                          │
                 384-D Semantic Vector
                          │
                          ▼
             Sentence-BERT Class Space
                          │
                          ▼
             Cosine Similarity Matching
                          │
                          ▼
              Explainability Module
                    (Week 5)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 Predicted Action   Top-k Predictions   Confidence
                          │
                          ▼
              Streamlit User Interface
                          │
                          ▼
          Reports + Visual Output + JSON
```

---

# Streamlit Application

The Streamlit application acts as the front-end of the entire project.

It does **not** retrain any models.

Instead, it loads the trained models produced during previous weeks:

- ResNet18 feature extractor
- Semantic Projection model
- StandardScaler
- Sentence-BERT class embeddings
- Action descriptions
- Explainability engine

The application simply performs inference on unseen videos.

---

# User Workflow

The complete interaction follows a simple sequence.

```text
Launch Application

        │

        ▼

Upload Video

        │

        ▼

Extract Frames

        │

        ▼

Generate Visual Features

        │

        ▼

Project into Semantic Space

        │

        ▼

Compare with Text Embeddings

        │

        ▼

Generate Prediction

        │

        ▼

Generate Explanation

        │

        ▼

Display Results
```

Every prediction follows exactly the same inference pipeline developed throughout Weeks 2–5.

---

# Streamlit Interface

The application provides a clean and intuitive interface.

Typical layout:

```text
-------------------------------------------------------

 Semantic Prompt-Guided Human Action Recognition

-------------------------------------------------------

Upload Video

[ Choose File ]

--------------------------------------------

Video Preview

--------------------------------------------

Prediction

PlayingGuitar

Semantic Similarity

0.5142

Confidence

Moderate

Representative Description

"A person holds a guitar and strums the strings."

Top Predictions

1. PlayingGuitar
2. Drumming
3. RopeClimbing

Explanation

The projected semantic representation was most similar
to the PlayingGuitar class embedding.

Reliability Note

These values are cosine similarities and not calibrated
probabilities.

-------------------------------------------------------
```

The interface is intentionally minimal so that the focus remains on model predictions and explainability.

---

# Inference Pipeline

The deployed application performs exactly the same operations as the command-line inference scripts.

```text
Video

↓

Frame Sampling

↓

Image Preprocessing

↓

ResNet18 Feature Extraction

↓

Mean Pooling

↓

512-D Video Feature

↓

StandardScaler

↓

Projection Network

↓

384-D Semantic Embedding

↓

Cosine Similarity

↓

Rank Predictions

↓

Explanation Generation

↓

Display Results
```

This ensures that the Streamlit application produces the same predictions as the standalone inference scripts developed in Weeks 4 and 5.

---

# Components Reused from Previous Weeks

Week 6 introduces very little new machine learning code.

Instead, it integrates components developed earlier.

| Week | Component |
|------|-----------|
| Week 2 | ResNet18 visual feature extraction |
| Week 3 | Sentence-BERT text embeddings |
| Week 4 | Semantic Projection Model |
| Week 4 | Cosine Similarity Classification |
| Week 5 | Explainability Engine |
| Week 5 | Prediction Reports |
| Week 5 | Confidence Interpretation |
| Week 5 | Reliability Notes |

This demonstrates an important principle of software engineering:

> A well-designed machine learning system should be modular, allowing previously developed components to be reused without modification.

---

# Advantages of the Deployment

The Streamlit application offers several practical benefits.

- Interactive user interface.
- No command-line knowledge required.
- Supports inference on unseen videos.
- Generates explainable predictions.
- Produces reusable prediction reports.
- Demonstrates the complete end-to-end pipeline.
- Makes the project suitable for academic presentations.
- Provides a foundation for future deployment on cloud platforms.

The deployment stage transforms the project from a development prototype into a complete application ready for demonstration.

---


# Project Organization

A well-organized project structure is essential for reproducibility and long-term maintenance. Throughout the six-week implementation, the repository was organized into separate directories for datasets, scripts, trained models, reports, visualizations, and deployment resources.

A simplified project structure is shown below.

```text
action-recognition-nlp/
│
├── app/
│
├── data/
│   ├── raw_videos/
│   ├── frames/
│   ├── features/
│   ├── text_embeddings/
│   └── fusion_features/
│
├── metadata/
│
├── models/
│
├── results/
│   ├── reports/
│   ├── plots/
│   └── predictions/
│
├── scripts/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Each directory has a clearly defined purpose, making the project easier to understand, maintain, and extend.

---

# Saved Models

By the end of the project, all trained models and supporting files are stored for future inference.

The repository includes:

- ResNet18 feature extractor configuration
- StandardScaler
- Logistic Regression baseline
- Label Encoder
- Semantic Projection Model
- Sentence-BERT class embeddings
- Action descriptions
- Prediction metadata

Saving these files allows inference without retraining the models, making deployment faster and more efficient.

---

# Reproducibility

One of the primary goals of the project is reproducibility.

A new user should be able to clone the repository, install the required dependencies, and run inference using the pretrained models.

Typical setup:

```bash
git clone <repository-url>

cd action-recognition-nlp

python -m venv .venv

pip install -r requirements.txt
```

Once the environment is configured, the Streamlit application can be launched directly.

```bash
streamlit run app/app.py
```

This ensures that the project can be reproduced on different systems with minimal effort.

---

# Documentation

Good documentation is as important as writing good code.

Throughout the project, documentation was created for:

- Dataset preparation
- Feature extraction
- Baseline model
- Text embedding generation
- Vision-language fusion
- Explainability module
- Streamlit deployment

The README provides installation instructions, workflow diagrams, project architecture, usage examples, and explanations of every major component.

This makes the repository suitable for both learning and demonstration.

---

# Project Deliverables

At the completion of Week 6, the project includes the following deliverables.

- Complete source code
- Organized dataset structure
- Trained machine learning models
- Sentence-BERT text embeddings
- Vision-language semantic projection model
- Explainable prediction pipeline
- Streamlit web application
- Prediction reports
- Evaluation reports
- Project documentation
- GitHub-ready repository

Together, these deliverables form a complete end-to-end machine learning application.

---

# Six-Week Project Timeline

```text
Week 1
Dataset Preparation
│
├── Dataset organization
├── Frame extraction
├── Train/Validation/Test split
└── Action descriptions

        │
        ▼

Week 2
Visual Baseline
│
├── ResNet18 features
├── Mean pooling
├── Logistic Regression
└── Performance evaluation

        │
        ▼

Week 3
Natural Language Processing
│
├── Sentence-BERT
├── Text embeddings
├── Semantic similarity
└── Embedding analysis

        │
        ▼

Week 4
Vision-Language Fusion
│
├── Semantic projection
├── Cosine similarity
├── Leakage-safe inference
└── Ablation study

        │
        ▼

Week 5
Explainable AI
│
├── Confidence estimation
├── Top-K predictions
├── Natural-language explanation
├── Visual evidence
└── Prediction reports

        │
        ▼

Week 6
Deployment
│
├── Streamlit interface
├── Documentation
├── GitHub repository
└── Final presentation
```

---

# Learning Outcomes

This six-week project demonstrates the complete workflow involved in developing a modern machine learning application.

After completing the project, the following concepts have been implemented and understood:

- Video preprocessing using OpenCV.
- Feature extraction using pretrained CNN models.
- Building a baseline machine learning classifier.
- Generating semantic embeddings using Sentence-BERT.
- Combining computer vision and NLP through vision-language fusion.
- Designing a leakage-safe semantic projection model.
- Measuring similarity using cosine similarity.
- Building an explainable AI inference pipeline.
- Creating structured prediction reports.
- Deploying a machine learning application using Streamlit.
- Organizing code for reproducibility and open-source publication.

The project integrates concepts from Computer Vision, Natural Language Processing, Machine Learning, Explainable AI, and Software Engineering into a single end-to-end application.

---

# Limitations

Although the project successfully demonstrates semantic action recognition, several limitations remain.

- The dataset contains only a subset of action classes.
- Temporal information is summarized using mean pooling instead of sequence models.
- Semantic projection uses a simple linear mapping.
- Explanations are generated from semantic similarity rather than visual attention maps.
- Real-time video streaming is not included.

These limitations provide opportunities for future improvements and research.

---

# Future Work

Several extensions can further improve the project.

- Support larger datasets such as UCF101, HMDB51, or Something-Something V2.
- Replace mean pooling with LSTM, GRU, or Transformer-based temporal modeling.
- Explore advanced multimodal fusion techniques.
- Integrate Vision Transformers or Video Swin Transformers.
- Add Grad-CAM or attention visualization.
- Support real-time webcam inference.
- Deploy the application on cloud platforms.
- Extend the explainability module using Large Language Models.

These enhancements would improve both recognition performance and interpretability.

---

# Project Summary

This project began as a traditional video classification task and gradually evolved into a semantic vision-language recognition system.

Starting from raw videos, visual features were extracted using a pretrained ResNet18 network. Sentence-BERT was then used to encode semantic action descriptions into a shared embedding space. A leakage-safe projection model learned to map visual representations into this semantic space, enabling action recognition through cosine similarity instead of direct class prediction.

To improve interpretability, an explainability module was introduced that reports confidence levels, Top-K predictions, representative action descriptions, and natural-language explanations. Finally, the complete inference pipeline was deployed through a Streamlit application, allowing users to upload new videos and obtain explainable predictions through an interactive interface.

The final outcome is an end-to-end **Semantic Prompt-Guided Human Action Recognition using Vision-Language Fusion** system that combines Computer Vision, Natural Language Processing, Explainable AI, and modern software engineering practices into a single deployable application.

---

# Conclusion

Over six weeks, the project progressed from dataset preparation to a fully deployable explainable action recognition system.

Each stage built upon the previous one, gradually introducing feature extraction, semantic embeddings, multimodal learning, explainability, and deployment.

The completed system demonstrates that semantic language information can effectively complement visual features, resulting in an interpretable vision-language model capable of recognizing human actions while providing meaningful explanations for its predictions.

This project establishes a strong foundation for future work in multimodal learning, explainable artificial intelligence, and research-oriented human action recognition.