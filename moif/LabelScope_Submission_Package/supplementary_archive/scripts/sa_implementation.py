"""
sa_implementation.py
Structure Audit (SA) core implementation for LabelScope.

Evaluates k-NN label coherence in physiological feature space with
block-shuffle permutation test.

Reference: LabelScope manuscript, Section III-C
"""
import numpy as np
from scipy.spatial.distance import cdist
import json
from pathlib import Path


def calc_knn_coherence(X, y, k=50, temporal_exclusion_idx=None):
    """
    Compute k-NN label coherence T_obs.

    Args:
        X: feature matrix (N, D), subject-wise z-scored
        y: label array (N,)
        k: number of neighbors
        temporal_exclusion_idx: set of (i, j) index pairs to exclude
                                from neighborhood (for temporal restriction)

    Returns:
        T_obs: scalar coherence value
    """
    N = len(y)
    distances = cdist(X, X, metric='euclidean')
    # Exclude self
    np.fill_diagonal(distances, np.inf)
    # Exclude temporal neighbors if specified
    if temporal_exclusion_idx is not None:
        for i, j in temporal_exclusion_idx:
            distances[i, j] = np.inf
            distances[j, i] = np.inf

    coherence = 0.0
    for i in range(N):
        neighbor_indices = np.argsort(distances[i])[:k]
        neighbor_labels = y[neighbor_indices]
        mode_label = np.bincount(neighbor_labels.astype(int)).argmax()
        if y[i] == mode_label:
            coherence += 1.0
    return coherence / N


def block_shuffle(y, block_length, subject_boundaries, rng):
    """
    Block-shuffle label array within subject boundaries.

    Args:
        y: label array
        block_length: length of each block in samples
        subject_boundaries: list of (start, end) tuples per subject
        rng: numpy random generator

    Returns:
        y_shuffled: shuffled label array
    """
    y_shuffled = y.copy()
    for start, end in subject_boundaries:
        y_sub = y[start:end]
        n_blocks = len(y_sub) // block_length
        block_indices = np.arange(n_blocks)
        rng.shuffle(block_indices)
        y_shuffled_sub = np.concatenate([
            y_sub[b * block_length:(b + 1) * block_length]
            for b in block_indices
        ])
        y_shuffled[start:start + len(y_shuffled_sub)] = y_shuffled_sub
    return y_shuffled


def run_structure_audit(X, y, subject_boundaries, k=50, B=1000, alpha=0.05,
                        block_length=None, temporal_exclusion_sec=60,
                        window_step_sec=30, seed=42):
    """
    Run full Structure Audit.

    Args:
        X: feature matrix
        y: label array
        subject_boundaries: list of (start, end) tuples per subject
        k: number of neighbors for coherence
        B: number of permutation iterations
        alpha: significance threshold
        block_length: block shuffle length in samples (default: inferred)
        temporal_exclusion_sec: seconds to exclude around each query window
        window_step_sec: step between windows in seconds
        seed: random seed

    Returns:
        T_obs, p_shuf, L_SA
    """
    rng = np.random.default_rng(seed)

    # Build temporal exclusion set
    exclusion_radius = int(temporal_exclusion_sec / window_step_sec)
    N = len(y)
    temporal_exclusion_idx = set()
    for i in range(N):
        for delta in range(-exclusion_radius, exclusion_radius + 1):
            j = i + delta
            if 0 <= j < N and i != j:
                temporal_exclusion_idx.add((i, j))

    T_obs = calc_knn_coherence(X, y, k=k,
                               temporal_exclusion_idx=temporal_exclusion_idx)

    if block_length is None:
        block_length = exclusion_radius * 2

    # Permutation test
    T_shuf = np.zeros(B)
    for b in range(B):
        y_shuf = block_shuffle(y, block_length, subject_boundaries, rng)
        T_shuf[b] = calc_knn_coherence(X, y_shuf, k=k,
                                       temporal_exclusion_idx=temporal_exclusion_idx)

    p_shuf = (1 + np.sum(T_shuf >= T_obs)) / (B + 1)
    L_SA = 1 if p_shuf >= alpha else 3
    return T_obs, p_shuf, L_SA


if __name__ == "__main__":
    # Load config
    with open("configs/audit_config.json") as f:
        cfg = json.load(f)["sa"]
    with open("configs/random_seeds.json") as f:
        seeds = json.load(f)

    print("SA implementation loaded. Provide X, y, subject_boundaries to run.")
    print(f"Default params: k={cfg['k_neighbors']}, B={seeds['n_permutations']}, "
          f"alpha={cfg['alpha']}")
