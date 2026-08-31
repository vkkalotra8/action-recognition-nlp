SemHAR-Gate Block A

Main notebook:
  SemHAR_Gate_Block_A_UCF101_HMDB51.ipynb

What it covers:
  - Full UCF101 and HMDB51
  - Official split parsing
  - Leakage-aware validation construction
  - Dataset/video audit
  - Temporal segment sampling
  - Clip-consistent spatial preprocessing
  - ImageNet normalization
  - Frozen ResNet18 512-D frame feature caching
  - BiGRU temporal modeling
  - Temporal attention
  - Validation-based checkpointing
  - Test metrics and confusion matrix
  - Exported visual embeddings/logits/probabilities/attention for future SemHAR-Gate blocks

First-run action:
  1) Open the notebook.
  2) In the Config cell, set auto_download=True only if the datasets are not already extracted locally.
  3) If datasets are already local, keep auto_download=False and edit DATASET_PATHS.
  4) For HMDB51 automatic extraction, ensure 7-Zip/unrar is available on PATH.
  5) Keep quick_mode=False for actual research runs.

The notebook is designed so later semantic/gating blocks reuse the same manifests and cached outputs.
