import numpy as np
from moif.invariance.divergence import compute_jsd, get_distribution
from statsmodels.stats.multitest import multipletests

def permutation_test(labels_c1: np.ndarray, labels_c2: np.ndarray, all_classes: list, n_perm: int = 1000, seed: int = 42):
    """
    Performs permutation test on JS divergence between two conditions.
    Returns (obs_jsd, p_value, effect_z)
    """
    rng = np.random.default_rng(seed)
    n1 = len(labels_c1)
    n2 = len(labels_c2)
    
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, 0.0
        
    dist1 = get_distribution(labels_c1, all_classes)
    dist2 = get_distribution(labels_c2, all_classes)
    obs_jsd = compute_jsd(dist1, dist2)
    
    pooled = np.concatenate([labels_c1, labels_c2])
    null_jsds = np.zeros(n_perm)
    
    for i in range(n_perm):
        rng.shuffle(pooled)
        pd1 = get_distribution(pooled[:n1], all_classes)
        pd2 = get_distribution(pooled[n1:], all_classes)
        null_jsds[i] = compute_jsd(pd1, pd2)
        
    p_value = np.mean(null_jsds >= obs_jsd)
    
    mean_null = np.mean(null_jsds)
    std_null = np.std(null_jsds)
    effect_z = 0.0
    if std_null > 0:
        effect_z = (obs_jsd - mean_null) / std_null
        
    return float(obs_jsd), float(p_value), float(effect_z)

def apply_fdr(p_values: list[float], alpha: float = 0.05) -> list[float]:
    if not p_values:
        return []
    reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return list(q_values)
