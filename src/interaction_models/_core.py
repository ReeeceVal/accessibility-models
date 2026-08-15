"""Segmented numpy primitives.

No domain concepts live here. A *segment* is a contiguous run of rows sharing the same
group code, which is what the ``(demand_code, cost, supply_code)`` sort in
:func:`interaction_models.prepare.prepare` guarantees for every demand node.

All reductions are ``O(n_pairs)`` with no pandas and no per-group Python loops.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_sum",
    "segment_first_rows",
    "segment_lengths",
    "segment_max",
    "segment_position",
    "segment_rank_desc",
    "segment_starts",
    "segment_sum",
]


def segment_starts(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Offsets delimiting each group's contiguous run of rows.

    Parameters
    ----------
    codes : ndarray of int
        Group code per row, sorted ascending. Values must lie in ``[0, n_groups)``.
    n_groups : int
        Total number of groups, including groups with no rows.

    Returns
    -------
    ndarray of int64, shape (n_groups + 1,)
        ``seg[g]:seg[g + 1]`` is the row slice of group ``g``. Empty groups give an
        empty slice rather than being omitted.
    """
    counts = np.bincount(codes, minlength=n_groups)
    seg = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(counts, out=seg[1:])
    return seg


def segment_lengths(seg: np.ndarray) -> np.ndarray:
    """Row count per group, shape ``(n_groups,)``."""
    return np.diff(seg)


def segment_sum(x: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Sum ``x`` within each segment.

    Accumulates in ``float64`` regardless of the input dtype. Empty segments sum to
    ``0.0`` — the reason this wraps :func:`numpy.add.reduceat`, which would otherwise
    emit the element at the boundary index.

    Parameters
    ----------
    x : ndarray, shape (n_rows,)
    seg : ndarray of int64, shape (n_groups + 1,)

    Returns
    -------
    ndarray of float64, shape (n_groups,)
    """
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    out = np.zeros(seg.size - 1, dtype=np.float64)
    nonempty = segment_lengths(seg) > 0
    if nonempty.any():
        out[nonempty] = np.add.reduceat(x, seg[:-1][nonempty])
    return out


def segment_max(x: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Maximum of ``x`` within each segment.

    The counterpart of :func:`segment_sum`, wrapping :func:`numpy.maximum.reduceat` for
    the same reason: an empty segment would otherwise emit the element at the boundary
    index. Empty segments give ``0.0``, matching the convention that a node reaching
    nobody contributes nothing.

    Parameters
    ----------
    x : ndarray, shape (n_rows,)
    seg : ndarray of int64, shape (n_groups + 1,)

    Returns
    -------
    ndarray of float64, shape (n_groups,)
    """
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    out = np.zeros(seg.size - 1, dtype=np.float64)
    nonempty = segment_lengths(seg) > 0
    if nonempty.any():
        out[nonempty] = np.maximum.reduceat(x, seg[:-1][nonempty])
    return out


def segment_position(seg: np.ndarray, n_rows: int) -> np.ndarray:
    """0-based position of each row within its own segment.

    Free given the sort: this is the rank by ``cost_default`` ascending, and therefore
    the rank by ``f_multi`` descending whenever impedance is monotone in that cost
    (the kappa-only fast path).
    """
    return np.arange(n_rows, dtype=np.int64) - np.repeat(seg[:-1], segment_lengths(seg))


def segment_first_rows(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row index of the first row of every non-empty segment.

    Returns
    -------
    groups : ndarray of int64
        Codes of the non-empty groups.
    rows : ndarray of int64
        Index of each group's first row. Given the cost-ascending sort this is the
        argmin over the segment, ties broken by lowest supply code.
    """
    nonempty = np.flatnonzero(segment_lengths(seg))
    return nonempty, seg[nonempty]


def segment_rank_desc(values: np.ndarray, codes: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Rank of each row within its segment by ``values`` descending.

    The general path, used when ``f_multi`` is not monotone in ``cost_default``. Ties
    keep the incoming row order (cost ascending, then supply code), so the result is
    deterministic.

    Returns
    -------
    ndarray of int64, shape (n_rows,)
        0 is the largest value in the segment.
    """
    order = np.lexsort((-values, codes))
    rank = np.empty(values.size, dtype=np.int64)
    rank[order] = segment_position(seg, values.size)
    return rank


def group_sum(codes: np.ndarray, weights: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum ``weights`` by group code, for codes in arbitrary order.

    The scatter counterpart of :func:`segment_sum`; used for per-supply reductions,
    whose codes are not sorted.

    Returns
    -------
    ndarray of float64, shape (n_groups,)
    """
    return np.bincount(codes, weights=weights, minlength=n_groups)
