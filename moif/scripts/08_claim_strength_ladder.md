# 08 Claim Strength Ladder

## Context
To prevent the claims from overrunning the evidence (as occurred in the previous manuscript), this ladder maps the eventual empirical results strictly to the permitted assertions in the abstract and discussion.

## Level 4: Strong Result (Target for THMS)
- **Empirical Trigger**: Normalization substantially changes predictions ($\ge 10\%$ disagreement) AND produces large subject-level clusters of variability.
- **Permitted Claim**: "In human-machine systems, normalization is not a technical preprocessing detail; it is actively part of the user-state model. The apparent relationship between physiological responses and subjective-state labels is heavily pipeline-dependent."

## Level 3: Moderate Result
- **Empirical Trigger**: Normalization affects some secondary metrics (AUROC/AUPRC) and specific subgroups, but average predictions remain largely stable ($\kappa > 0.85$).
- **Permitted Claim**: "Preprocessing choices affect the stability and confidence of subjective-state models, particularly for susceptible subgroups. Human-machine systems should report and justify normalization strategies to ensure reproducibility."

## Level 2: Weak Result
- **Empirical Trigger**: Normalization effects are observed *only* when comparing extreme cases (e.g., 30s rolling window vs full train-block).
- **Permitted Claim**: "Short-window rolling normalization can suppress sustained offsets, degrading separability. Resting-baseline calibration offers a more stable alternative under deployment constraints." (Broader systemic claims are NOT supported).

## Level 1: Null Result
- **Empirical Trigger**: All methods yield statistically indistinguishable performance and identical predictions.
- **Permitted Claim**: "For binary acute stress and arousal detection, the choice of normalization strategy does not meaningfully alter the physiological feature space or classifier decisions." (THMS submission aborted).
