# 01 Dataset Summary

## Context within THMS Paper
This section provides rigorous reproducibility details for the two public datasets (WESAD, CASE) used in this study. We explicitly define the "baseline" and "active" states for each to ensure transparency. 

## Datasets

### 1. WESAD (Wearable Stress and Affect Detection)
- **Subjects**: 15 participants.
- **Task Formulations**:
  - `Baseline`: Resting condition (approx. 20 minutes).
  - `Active`: Acute stress condition induced by the Trier Social Stress Test (TSST) (approx. 10 minutes).
- **Sensors**: Resampled to 100 Hz dense extraction (BVP, EDA, TEMP, etc.).

### 2. CASE (Continuously Annotated Signals of Emotion)
- **Subjects**: 30 participants.
- **Task Formulations**:
  - `Baseline`: Neutral video watching.
  - `Active`: High-arousal + negative-valence video segments (fear/amusement control).
- **Sensors**: Resampled to 100 Hz dense extraction.

## Critical Interpretive Stance
We **do not** claim that "WESAD Stress" and "CASE High Arousal/Negative Valence" are perfectly identical psychological states. Rather, they are *related subjective-state formulations*. We utilize them to evaluate how normalization choices affect state separability within datasets, and use cross-dataset evaluation purely as a "stress test of label-physiology invariance", avoiding claims of universal emotion recognition.

## Reporting Requirements
The final manuscript must include a comprehensive Table 1 detailing:
- Number of subjects included after exclusion criteria.
- Total sample counts.
- Class balance (train vs test split).
- Missing data handling strategy.
