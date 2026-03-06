import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_jsd(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Computes Jensen-Shannon Divergence (base 2) between two probability distributions.
    """
    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0
    # jensenshannon returns distance. Divergence is distance^2.
    js_dist = jensenshannon(dist1, dist2, base=2.0)
    if np.isnan(js_dist):
        return 0.0
    return float(js_dist ** 2)

def get_distribution(labels: list | np.ndarray, all_classes: list) -> np.ndarray:
    """
    Calculates probability distribution among all given classes.
    """
    if len(labels) == 0:
        return np.ones(len(all_classes)) / len(all_classes)
        
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    probs = np.array([counts.get(c, 0) for c in all_classes], dtype=float)
    return probs / probs.sum()
