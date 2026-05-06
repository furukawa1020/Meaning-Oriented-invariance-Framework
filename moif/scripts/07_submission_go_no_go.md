# 07 Submission Go / No-Go Gates

## Context
This document defines the strict, numeric empirical thresholds required to submit this manuscript to IEEE THMS. If the results fail these gates, the paper will be demoted to a different journal or completely restructured.

## Gate 1: THMS Submission (GO)
To submit to THMS, **ALL** of the following must be true:
- **Normalization Effect**: $\Delta$AUROC $\ge$ 0.03, $\Delta$AUPRC $\ge$ 0.03, or $\Delta$MCC $\ge$ 0.03 across methods.
- **Pipeline Disagreement**: Mean prediction disagreement rate $\ge$ 5%, Cohen's $\kappa \le 0.90$ for at least one critical method pair (e.g., Rolling vs Baseline-only).
- **Subject-level Heterogeneity**: At least 20-30% of subjects show meaningful method-dependent changes (e.g., $|\Delta$AUROC$| \ge 0.05$ or prediction disagreement $\ge 10\%$).
- **Deployment Feasibility**: Calibration length ablation (30s to 600s) shows a measurable stabilization curve, proving practical implications.

## Gate 2: Alternate Submission (Biomedical Signal Processing and Control / HIS)
If the results show poor pipeline disagreement but confirm rolling normalization failure:
- **Condition**: Disagreement rate $< 5\%$, but Rolling Z-score F1/AUROC is significantly worse than static methods.
- **Action**: Abandon THMS. Submit as a technical note to BSPC or domestic HIS, focusing purely on "The danger of short-window rolling normalization in affective computing."

## Gate 3: Alternate Submission (Affective Computing / Biosignal Modeling)
If the main results are flat, but cross-dataset transfer yields strong results:
- **Condition**: Normalization effects are minimal, but WESAD $\rightarrow$ CASE transfer shows strong physiological feature correlations under specific scalers.
- **Action**: Abandon THMS. Pivot to a cross-dataset emotion recognition paper targeting IEEE TAFFC or similar.

## Gate 4: No-Go (Do Not Submit / Negative Result)
If all gates fail:
- **Condition**: All methods perform identically, no pipeline disagreement, no subject-level clusters, and cross-dataset transfer is random.
- **Action**: Abort submission. Re-evaluate the entire experimental premise or publish as a brief negative result.
