| Experiment | Feature dimension | Leakage | Valid inference | Test accuracy | Macro F1 | Weighted F1 |
|---|---:|---|---|---:|---:|---:|
| Video-only baseline | 512 | No | Yes | 0.9340 | 0.9277 | 0.9341 |
| Text-only oracle | 384 | Yes | No | 1.0000 | 1.0000 | 1.0000 |
| Oracle concatenation | 896 | Yes | No | 1.0000 | 1.0000 | 1.0000 |
| Leakage-safe semantic projection | 384 | No | Yes | 0.8782 | 0.8639 | 0.8734 |