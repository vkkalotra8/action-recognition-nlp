# Presentation Source Traceability Summary

**Presentation File:** [`reports/presentation/Action_Recognition_Final_Presentation.pptx`](file:///d:/Projects/ML_project/action-recognition-nlp/reports/presentation/Action_Recognition_Final_Presentation.pptx)  
**Project:** Semantic Prompt-Guided Human Action Recognition using Vision-Language Fusion  
**Generated Date:** August 2026  
**Format:** 16:9 Widescreen (10 Slides with Embedded Figures & Comprehensive Speaker Notes)

This document establishes 100% evidence traceability for every numerical metric, dataset statistic, model parameter, experimental result, and visual figure used across the 10-slide final project presentation.

---

## Slide 1: Title & Administrative Metadata

* **Project Title & Subtitle:** Derived from repository [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L1-L4).
* **Candidate & Institution Placeholders:** Standard university viva defense format. Editable fields for student name, roll number, project guide, department, institution, and academic session.
* **Speaker Notes:** Introductory remarks and presentation agenda.

---

## Slide 2: Problem Statement & Project Objectives

* **Traditional HAR Limitations:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L84-L93) and [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L1300-L1375).
* **Vision-Language Motivation:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L993-L1008).
* **4 Core Project Objectives:** Derived from the 6-week project plan and verified repository milestones in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L3067-L3135).

---

## Slide 3: Dataset & Data Engineering Pipeline

* **Action Classes (9):** [`metadata/classes.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/metadata/classes.txt)
  * `Archery`, `Basketball`, `Biking`, `Bowling`, `Drumming`, `JavelinThrow`, `PlayingGuitar`, `RopeClimbing`, `Typing`
* **Video Splits (1,261 total):** [`metadata/video_splits.csv`](file:///d:/Projects/ML_project/action-recognition-nlp/metadata/video_splits.csv) & [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L108-L116)
  * Training videos: **867** (68.75%)
  * Validation videos: **197** (15.62%)
  * Test videos: **197** (15.62%)
* **Extracted Frame Statistics (20,176 total):** [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L124-L134)
  * 16 frames per video, resized to 224 × 224 RGB.
  * Training frames: **13,872** | Validation frames: **3,152** | Test frames: **3,152**
* **Group-Safe Split Policy:** Defined and validated in [`scripts/create_splits.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/create_splits.py) and [`scripts/check_split_leakage.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/check_split_leakage.py).

---

## Slide 4: Complete System Architecture

* **Visual Pipeline (ResNet18, 512-D):** [`scripts/extract_cnn_features.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/extract_cnn_features.py) & [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L175-L204).
* **Language Pipeline (Sentence-BERT, 384-D):** [`scripts/encode_action_descriptions.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/encode_action_descriptions.py) & [`metadata/action_descriptions.json`](file:///d:/Projects/ML_project/action-recognition-nlp/metadata/action_descriptions.json) (45 prompts, 5 per class).
* **Semantic Projection Network (512 → 384-D):** [`scripts/train_semantic_projection.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/train_semantic_projection.py) & [`models/semantic_projection_metadata.json`](file:///d:/Projects/ML_project/action-recognition-nlp/models/semantic_projection_metadata.json).
* **Cosine Matching & Explainability Engine:** [`scripts/explanation_engine.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/explanation_engine.py) & [`scripts/predict_video_explainable.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/predict_video_explainable.py).

---

## Slide 5: Visual Baseline & Semantic Representation

* **Visual Baseline Metrics (Video-Only Logistic Regression):**
  * Overall Test Metrics: [`results/reports/test_metrics.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/test_metrics.json)
    * Accuracy: **0.9340** (93.40%)
    * Macro Precision: **0.9289** (92.89%)
    * Macro Recall: **0.9284** (92.84%)
    * Macro F1-Score: **0.9277** (92.77%)
    * Weighted F1-Score: **0.9341** (93.41%)
  * Class-wise Breakdown & Misclassifications: [`results/reports/test_classification_report.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/test_classification_report.txt) & [`results/reports/test_misclassifications.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/test_misclassifications.json).
* **Semantic Space & Heatmap:**
  * Action Descriptions: [`metadata/action_descriptions.json`](file:///d:/Projects/ML_project/action-recognition-nlp/metadata/action_descriptions.json)
  * Semantic Similarity Report: [`results/reports/text_similarity_report.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/text_similarity_report.txt)
  * Semantic Similarity Matrix: [`results/reports/text_similarity_matrix.csv`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/text_similarity_matrix.csv)
  * **Embedded Heatmap Figure:** [`results/plots/text_similarity_heatmap.png`](file:///d:/Projects/ML_project/action-recognition-nlp/results/plots/text_similarity_heatmap.png)

---

## Slide 6: Vision-Language Fusion & Label Leakage Dilemma

* **Oracle Experiment Findings:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L1060-L1100), [`results/reports/oracle_fusion_test_metrics.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/oracle_fusion_test_metrics.json), and [`results/reports/text_only_test_metrics.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/text_only_test_metrics.json).
* **Leakage-Safe Architecture Formulation:** [`scripts/train_semantic_projection.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/train_semantic_projection.py) & [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L1104-L1134).
* **Methodological Integrity Rationale:** [`results/reports/week4_ablation_report.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/week4_ablation_report.txt).

---

## Slide 7: Experimental Results & Ablation Analysis

* **Ablation Results Summary:** [`results/reports/week4_ablation_results.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/week4_ablation_results.json) & [`results/reports/week4_ablation_results.csv`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/week4_ablation_results.csv):
  1. *Video-only baseline (512-D, No leakage):* Test Acc: **0.9340** | Macro F1: **0.9277**
  2. *Text-only oracle (384-D, Leakage = True):* Test Acc: **1.0000** | Macro F1: **1.0000**
  3. *Oracle concatenation (896-D, Leakage = True):* Test Acc: **1.0000** | Macro F1: **1.0000**
  4. *Leakage-safe semantic projection (384-D, No leakage):* Test Acc: **0.8782** | Macro F1: **0.8639**
* **Class-wise Comparison:** [`results/reports/week4_classwise_comparison.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/week4_classwise_comparison.json) & [`results/reports/week4_classwise_comparison.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/results/reports/week4_classwise_comparison.txt).
* **Explicit Note on Unmeasured Metrics:** `Top-3 Accuracy`, `Top-5 Accuracy`, and `Inference Latency` are explicitly marked as `[RESULT NOT AVAILABLE]` because they are not present in the existing project evaluation artifacts.

---

## Slide 8: Explainable AI & Qualitative Prediction Case Study

* **Verified Held-Out Test Video:** `v_PlayingGuitar_g10_c01.avi`
* **JSON Prediction Report:** [`results/predictions/playing_guitar_visual_example_prediction.json`](file:///d:/Projects/ML_project/action-recognition-nlp/results/predictions/playing_guitar_visual_example_prediction.json)
* **Text Prediction Report:** [`results/predictions/playing_guitar_visual_example_prediction.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/results/predictions/playing_guitar_visual_example_prediction.txt)
* **Embedded Visual Contact Sheet:** [`results/predictions/playing_guitar_contact_test_contact_sheet.jpg`](file:///d:/Projects/ML_project/action-recognition-nlp/results/predictions/playing_guitar_contact_test_contact_sheet.jpg)
* **Prediction Values:**
  * Predicted Class: `PlayingGuitar`
  * Cosine Similarity: `0.5230`
  * Confidence Level: `Moderate`
  * Score Gap: `+0.4719` (Clear separation over `Drumming` at `0.0511`)
  * Generated Natural Language Explanation: *"The projected semantic representation of the video was most similar to the PlayingGuitar class embedding. Its similarity score of 0.5230 was clearly higher than the next-best class, Drumming, which scored 0.0511."*

---

## Slide 9: Interactive Deployment — Streamlit Web Application

* **Streamlit Application Source:** [`app/streamlit_app.py`](file:///d:/Projects/ML_project/action-recognition-nlp/app/streamlit_app.py)
* **Supported Video Input Formats:** `.avi`, `.mp4`, `.mov`, `.mkv` (lines 31–36 of `streamlit_app.py`)
* **Inference Pipeline Integration:** Lines 169–217 of `streamlit_app.py` calling [`scripts/predict_video_explainable.py`](file:///d:/Projects/ML_project/action-recognition-nlp/scripts/predict_video_explainable.py).
* **Technology Stack Dependencies:** Verified against [`requirements.txt`](file:///d:/Projects/ML_project/action-recognition-nlp/requirements.txt).

---

## Slide 10: Conclusion, Limitations & Future Roadmap

* **Contributions & Conclusions:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L3137-L3210).
* **Technical Limitations:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L633-L679) & [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L3159-L3170).
* **Future Work Directions:** Documented in [`README.md`](file:///d:/Projects/ML_project/action-recognition-nlp/README.md#L3173-L3187).

---

## Verification & Integrity Statement

All statements, numbers, and diagrams presented in the final presentation are directly derived from the verified source code, generated metrics JSON files, and saved evaluation plots. No synthetic results or unverified claims were introduced.
