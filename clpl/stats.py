"""Small statistics helpers: Wilson interval, mean/std, paired bootstrap."""
import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score 95% interval for k successes in n trials. Returns (low, high)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def paired_bootstrap_diff(a, b, n_boot=10000, seed=0):
    """a, b: per-episode success (0..1, e.g. averaged over training seeds) for the SAME eval episodes.
    Returns (mean diff a-b, 95% percentile interval) from resampling episodes with replacement."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
