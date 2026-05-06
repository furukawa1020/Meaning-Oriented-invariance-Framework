# 09 Known Reviewer Attacks & Rebuttals

## Context
This document anticipates the most likely attacks from THMS reviewers and defines how our manuscript structure pre-emptively neutralizes them.

## Attack 1: "This is just a preprocessing comparison, not a THMS paper."
- **Rebuttal Strategy**: We explicitly shift the focus in the Introduction and Discussion from *signal processing accuracy* to *system interpretability and validity*. We emphasize Pipeline Disagreement—showing that the machine's understanding of the human user changes based on hidden preprocessing math, which is a core Cognitive Ergonomics and System Evaluation issue.

## Attack 2: "WESAD (Stress) and CASE (Arousal/Valence) labels are not equivalent."
- **Rebuttal Strategy**: We explicitly state in Section IV (Datasets) that these are "related but non-identical subjective-state formulations." We frame cross-dataset evaluation strictly as a *stress test of label-physiology invariance*, NOT a claim of universal emotion mapping.

## Attack 3: "The observed effects are specific to Logistic Regression."
- **Rebuttal Strategy**: We use fixed Logistic Regression in the main text specifically to isolate the causal effect of normalization (interpretability). However, we include a robust Supplementary section demonstrating that the normalization effects persist across Random Forest and Linear SVM classifiers.

## Attack 4: "Rolling Z-score was implemented poorly (e.g., window too short)."
- **Rebuttal Strategy**: We ablate rolling windows across 30s, 60s, 120s, and 300s, and include EMA (Exponential Moving Average) and Robust (Median/IQR) rolling in the supplementary materials to prove the effect is inherent to temporal windowing, not a specific arbitrary parameter.

## Attack 5: "Subject-level results are underpowered (only 15 WESAD subjects)."
- **Rebuttal Strategy**: We use Mixed-Effects models and subject-level paired plots to transparently show variance. We do not claim population-level universality; rather, we highlight the *existence of subject-level heterogeneity* as a reason why population-wide HMS models are fragile.

## Attack 6: "Physiological signals CAN infer subjective states; your framing is too pessimistic."
- **Rebuttal Strategy**: We do not claim physiological signals are useless. We claim they should not be treated as "preprocessing-invariant direct labels." We advocate for context-aware, pipeline-transparent modeling, which strengthens, rather than destroys, the field of affective computing.
