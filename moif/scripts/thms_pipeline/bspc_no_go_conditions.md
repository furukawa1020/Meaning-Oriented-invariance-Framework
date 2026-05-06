# BSPC Claim Ladder: Evidence-Based Claims

> [!WARNING]
> The Phase 2G audit reveals that the effect is strictly dataset-dependent (CASE > WESAD) and statistically borderline (p=0.053 for RF, p=0.15 for LR).

### Level 1: Defensible Claim (The "Safe" Zone)
- "Baseline covariance calibration provides a statistically detectable performance boost (+0.03 to +0.06 AUROC) in the CASE dataset under window-level evaluation, but yields no detectable benefit in the WESAD dataset due to ceiling effects and inherent class separability."
- **Use for**: Technical Note, Data-specific benchmark.

### Level 2: Moderate Claim (The "BSPC" Target)
- "The utility of resting-baseline covariance calibration is context-dependent: it significantly improves classification in datasets with lower inherent separability (CASE), while providing redundant information in high-SNR environments (WESAD)."
- **Use for**: Original Research focusing on "Context-dependent preprocessing."

### Level 3: Prohibited Claim (The "Rejection" Zone)
- "Covariance calibration robustly enhances physiological state classification."
- "Normalization unlocks model capacity across wearable datasets."
- **Reason**: **REJECTED.** WESAD results (16.6% support, 0.0 delta on non-ceiling) directly contradict this.

---

# BSPC No-Go Conditions

1. **Gate G1: Cross-Dataset Robustness (FAIL)**
   - WESAD non-ceiling delta is 0.000. 
   - **Status**: **FAILED.** We cannot claim generalizability across wearable datasets.

2. **Gate G2: Statistical Significance (FAIL/BORDERLINE)**
   - RF p-value = 0.053 (Non-significant at alpha=0.05).
   - LR p-value = 0.15.
   - **Status**: **FAILED.** The evidence package is statistically insufficient for a high-impact technical development claim.

3. **Gate G3: Ceiling Neutrality (PASS)**
   - 54% of WESAD/CASE slots are at ceiling (AUROC >= 0.98).
   - **Status**: **PASS (with heavy caveat).** The task itself is too easy for the proposed method to demonstrate value in "clean" laboratory datasets like WESAD.

## Final Recommendation
**HALT BSPC FULL PAPER PREPARATION.**
The current results are too dataset-specific and statistically weak for a 10-page international journal submission. 

**Alternative Path**:
- Document as a **Technical Report** or **Open Dataset Audit**.
- Redesign the evaluation with a "Noisy" or "Real-world" dataset where AUROC is not at ceiling (0.6 - 0.8 range), as the method only shows value when the task is difficult.
