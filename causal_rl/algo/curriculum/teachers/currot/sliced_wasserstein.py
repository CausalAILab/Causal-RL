# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/spl/sliced_wasserstein.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np


def sliced_wasserstein(source, target, n=100, grad=False):
    """Sliced Wasserstein distance between two equally sized particle sets (and optionally the
    per-particle movement of `source` towards `target`)."""
    # Generate random projection vectors
    dim = source.shape[1]
    directions = np.random.normal(0, 1, (n, dim))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    # Compute the projected assignments
    source_projections = np.einsum("nd,md->nm", directions, source)
    target_projections = np.einsum("nd,md->nm", directions, target)

    sorted_source = np.argsort(source_projections, axis=-1)
    reverse_source = np.zeros_like(sorted_source)
    reverse_source[np.arange(0, n)[:, None], sorted_source] = np.arange(0, source.shape[0])[None, :]
    sorted_target = np.argsort(target_projections, axis=-1)

    proj_diffs = target_projections[np.arange(n)[:, None], sorted_target[np.arange(n)[:, None], reverse_source]] - \
                 source_projections
    swd = np.mean(np.square(proj_diffs))

    if grad:
        return swd, np.einsum("ij,id->jd", proj_diffs, directions) / n
    else:
        return swd
