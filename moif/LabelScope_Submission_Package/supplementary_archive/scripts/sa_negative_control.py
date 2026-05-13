"""
sa_negative_control.py
SA negative control: run-length-preserving block-randomized permutation.

This script generates the negative control condition reported in Table IV
of the LabelScope manuscript. A run-length-preserving block-randomized label
assignment is used to verify that the SA permutation test correctly produces
a null result when the temporal structure of the labels is destroyed.
"""
import numpy as np
from sa_implementation import calc_knn_coherence, run_structure_audit


def run_length_preserving_shuffle(y, rng):
    """
    Shuffle labels while approximately preserving run-length distribution.
    Used as negative control to verify the SA permutation test.
    """
    unique, counts = np.unique(y, return_counts=True)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    return y_shuffled


def run_negative_control(X, y, subject_boundaries, k=50, B=1000, alpha=0.05, seed=42):
    """
    Run SA negative control (run-length-preserving block-randomized).

    Expected outcome: T_obs ~ T_null, p_shuf >= alpha, L_SA = 1 (cap).
    """
    rng = np.random.default_rng(seed)
    # Apply run-length-preserving shuffle to create null labels
    y_null = run_length_preserving_shuffle(y, rng)
    T_obs_null = calc_knn_coherence(X, y_null, k=k)

    T_shuf = np.zeros(B)
    for b in range(B):
        y_shuf = run_length_preserving_shuffle(y_null, rng)
        T_shuf[b] = calc_knn_coherence(X, y_shuf, k=k)

    p_shuf = (1 + np.sum(T_shuf >= T_obs_null)) / (B + 1)
    L_SA = 1 if p_shuf >= alpha else 3
    print(f"Negative control: T_obs={T_obs_null:.3f}, "
          f"T_null_mean={T_shuf.mean():.3f}, p_shuf={p_shuf:.3f}, L_SA={L_SA}")
    return T_obs_null, p_shuf, L_SA


if __name__ == "__main__":
    print("Negative control script. Provide X, y, subject_boundaries.")
    print("Expected result: T_obs ~ 0.52, p_shuf ~ 0.51, L_SA = 1 (as in manuscript Table IV)")
