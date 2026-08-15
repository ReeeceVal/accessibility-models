"""Optional statistic groups.

Requested by group name — ``stats=["coverage", "inequality"]`` — and returned as one flat
``dict[str, float]``. A group is computed only when asked for, so the default cost is
nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _core

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .prepare import Prepared

__all__ = ["GROUPS", "compute_stats", "gini"]

GROUPS = ("coverage", "distribution", "inequality", "choice_set")

_MOMENTS = {
    "mean": np.mean,
    "median": np.median,
    "p10": lambda x: np.percentile(x, 10),
    "p90": lambda x: np.percentile(x, 90),
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Gini coefficient, optionally frequency-weighted.

    Parameters
    ----------
    values : ndarray
        Non-negative quantities.
    weights : ndarray, optional
        Population weights. Unweighted when omitted.

    Returns
    -------
    float
        0.0 for a perfectly equal distribution, approaching 1.0 as it concentrates.
        ``NaN`` when the total is zero or there is nothing to measure.

    Notes
    -----
    Computed from the sorted Lorenz curve rather than the O(n^2) mean-difference form::

        G = 1 - sum_k w_k (S_{k-1} + S_k) / (S_n * sum_k w_k),  S_k = cumsum(w * x)

    For ``1..n`` unweighted this reproduces the closed form ``(n - 1) / (3n)``.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cumulative = np.cumsum(w * v)
    total, total_w = cumulative[-1], w.sum()
    if total <= 0 or total_w <= 0:
        return float("nan")
    previous = np.concatenate(([0.0], cumulative[:-1]))
    return float(1.0 - np.sum(w * (previous + cumulative)) / (total * total_w))


def _coverage(prep: Prepared, n_pairs_used: int, E_j: np.ndarray, A_i: np.ndarray) -> dict:
    total_demand = float(prep.P.sum())
    sum_E_j = float(E_j.sum())
    return {
        "n_demand": float(prep.n_demand),
        "n_supply": float(prep.n_supply),
        "n_pairs_input": float(prep.n_pairs),
        "n_pairs_used": float(n_pairs_used),
        "total_demand": total_demand,
        "total_capacity": float(prep.S.sum()) if prep.S is not None else float("nan"),
        "sum_E_j": sum_E_j,
        "demand_capture_rate": sum_E_j / total_demand if total_demand else float("nan"),
        "n_supply_zero_exposure": float((E_j == 0).sum()),
        "n_demand_zero_access": float((A_i == 0).sum()),
    }


def _distribution(prep: Prepared, E_j: np.ndarray, A_i: np.ndarray) -> dict:
    out = {}
    for name, values in (("A_i", A_i), ("E_j", E_j)):
        for moment, fn in _MOMENTS.items():
            out[f"{moment}_{name}"] = float(fn(values)) if values.size else float("nan")
    total_demand = prep.P.sum()
    out["weighted_mean_A_i"] = (
        float((prep.P * A_i).sum() / total_demand) if total_demand else float("nan")
    )
    return out


def _inequality(prep: Prepared, E_j: np.ndarray, A_i: np.ndarray) -> dict:
    p10, p90 = np.percentile(A_i, [10, 90]) if A_i.size else (np.nan, np.nan)
    return {
        "gini_A_i": gini(A_i, prep.P),
        "gini_E_j": gini(E_j),
        "p90_p10_ratio_A_i": float(p90 / p10) if p10 else float("inf"),
    }


def _choice_set(G: np.ndarray, seg: np.ndarray, Q: int | None) -> dict:
    sizes = _core.segment_lengths(seg).astype(np.float64)
    herfindahl = _core.segment_sum(G * G, seg)
    N_eff = np.divide(
        1.0, herfindahl, out=np.full(sizes.size, np.nan), where=herfindahl > 0
    )[sizes > 0]
    return {
        "mean_N_eff_i": float(np.mean(N_eff)) if N_eff.size else float("nan"),
        "median_N_eff_i": float(np.median(N_eff)) if N_eff.size else float("nan"),
        "mean_choice_set_size": float(sizes.mean()) if sizes.size else float("nan"),
        "median_choice_set_size": float(np.median(sizes)) if sizes.size else float("nan"),
        "frac_demand_q_binding": float((sizes == Q).mean()) if sizes.size else float("nan"),
    }


def compute_stats(
    groups: Sequence[str],
    *,
    prep: Prepared,
    n_pairs_used: int,
    E_j: np.ndarray,
    A_i: np.ndarray,
    seg: np.ndarray,
    Q: int | None = None,
    G: np.ndarray | None = None,
    family: str = "",
) -> dict[str, float]:
    """Compute the requested statistic groups.

    Parameters
    ----------
    groups : sequence of str
        Any of ``"coverage"``, ``"distribution"``, ``"inequality"``, ``"choice_set"``.
    prep : Prepared
    n_pairs_used : int
        Pairs surviving every filter.
    E_j, A_i : ndarray
        The model's outputs, before frame assembly.
    seg : ndarray
        Segment offsets over the surviving pairs.
    Q : int, optional
    G : ndarray, optional
        Selection probabilities. Required by ``"choice_set"``, which is why that group is
        only available from Q-restricted families.
    family : str
        Used only to phrase the error when ``"choice_set"`` is not applicable.

    Returns
    -------
    dict of str to float

    Raises
    ------
    ValueError
        On an unknown group name, or on ``"choice_set"`` from a family without a choice
        set.
    """
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown stat group(s) {unknown}; available: {list(GROUPS)}")

    out: dict[str, float] = {}
    for group in groups:
        if group == "coverage":
            out.update(_coverage(prep, n_pairs_used, E_j, A_i))
        elif group == "distribution":
            out.update(_distribution(prep, E_j, A_i))
        elif group == "inequality":
            out.update(_inequality(prep, E_j, A_i))
        elif group == "choice_set":
            if G is None:
                raise ValueError(
                    f"stat group 'choice_set' needs a choice set; {family} has none "
                    "(use sfca or sfca_e, or ifca with Q set)"
                )
            out.update(_choice_set(G, seg, Q))
    return out
