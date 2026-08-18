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

__all__ = ["GROUPS", "compute_stats", "gini", "needs_A_i", "needs_E_j"]

GROUPS = ("coverage", "distribution", "inequality", "exposure", "choice_set")

#: Which model outputs each group reads. A caller that gates ``E_j`` and ``A_i`` on these
#: skips the work no requested group will look at -- ``A_i`` in particular costs an
#: O(n_pairs) pass, and its weighted Gini an O(n_demand log n_demand) sort.
_NEEDS_E_J = frozenset({"coverage", "distribution", "inequality", "exposure"})
_NEEDS_A_I = frozenset({"coverage", "distribution", "inequality"})

_MOMENTS = {
    "mean": np.mean,
    "median": np.median,
    "p10": lambda x: np.percentile(x, 10),
    "p90": lambda x: np.percentile(x, 90),
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def needs_E_j(groups: Sequence[str] | None) -> bool:
    """Whether any requested group reads ``E_j``. False for no groups at all."""
    return bool(groups) and not _NEEDS_E_J.isdisjoint(groups)


def needs_A_i(groups: Sequence[str] | None) -> bool:
    """Whether any requested group reads ``A_i``. False for no groups at all."""
    return bool(groups) and not _NEEDS_A_I.isdisjoint(groups)


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


def _moments(name: str, values: np.ndarray) -> dict:
    return {
        f"{moment}_{name}": float(fn(values)) if values.size else float("nan")
        for moment, fn in _MOMENTS.items()
    }


def _supply_ginis(prep: Prepared, E_j: np.ndarray, open_set: np.ndarray | None) -> dict:
    """The two supply-side concentration measures, over the **open** sites.

    ``gini_E_j`` asks how unevenly demand lands across sites; ``gini_L_j`` asks how
    unevenly it lands per unit of capacity, which is the operational reading — a site with
    five units of capacity is expected to absorb more than one with a single unit, and only ``L_j``
    accounts for that. ``G_ij`` is capacity-blind, so ``gini_E_j`` alone cannot see it.

    Closed sites are excluded rather than counted as zero. A Gini is a concentration
    measure, so padding it with zeros inflates it directly: counting the sites a network
    does *not* have would make the coefficient partly a function of how many candidates
    were declined, and declining to build would read as a fairness cost. Excluding them
    makes a masked run report what a ``prepare()`` over the open sites alone reports.
    ``gini_L_j`` drops ``S_j == 0`` for the same reason — the load is undefined there,
    which is also how the supply frame reports it.

    Shared by ``exposure`` and ``inequality`` so the two cannot drift apart.
    """
    S = prep.S
    if open_set is not None:
        E_j = E_j[open_set]
        S = None if S is None else S[open_set]
    if S is None:
        gini_L_j = float("nan")  # no capacity column; `exposure` tolerates that
    else:
        have = S > 0
        gini_L_j = gini(E_j[have] / S[have]) if have.any() else float("nan")
    return {"gini_E_j": gini(E_j), "gini_L_j": gini_L_j}


def _exposure(prep: Prepared, E_j: np.ndarray, open_set: np.ndarray | None) -> dict:
    total_demand = float(prep.P.sum())
    sum_E_j = float(E_j.sum())
    return {
        "total_demand": total_demand,
        "sum_E_j": sum_E_j,
        "demand_capture_rate": sum_E_j / total_demand if total_demand else float("nan"),
        "n_supply_zero_exposure": float((E_j == 0).sum()),
        **_supply_ginis(prep, E_j, open_set),
        **_moments("E_j", E_j),
    }


def _distribution(prep: Prepared, E_j: np.ndarray, A_i: np.ndarray) -> dict:
    out = _moments("A_i", A_i) | _moments("E_j", E_j)
    total_demand = prep.P.sum()
    out["weighted_mean_A_i"] = (
        float((prep.P * A_i).sum() / total_demand) if total_demand else float("nan")
    )
    return out


def _inequality(
    prep: Prepared, E_j: np.ndarray, A_i: np.ndarray, open_set: np.ndarray | None
) -> dict:
    p10, p90 = np.percentile(A_i, [10, 90]) if A_i.size else (np.nan, np.nan)
    return {
        "gini_A_i": gini(A_i, prep.P),
        **_supply_ginis(prep, E_j, open_set),
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
    E_j: np.ndarray | None,
    A_i: np.ndarray | None,
    seg: np.ndarray,
    Q: int | None = None,
    G: np.ndarray | None = None,
    open_set: np.ndarray | None = None,
    family: str = "",
) -> dict[str, float]:
    """Compute the requested statistic groups.

    Parameters
    ----------
    groups : sequence of str
        Any of ``"coverage"``, ``"distribution"``, ``"inequality"``, ``"exposure"``,
        ``"choice_set"``.
    prep : Prepared
    n_pairs_used : int
        Pairs surviving every filter.
    E_j, A_i : ndarray, optional
        The model's outputs, before frame assembly. May be ``None`` only when no
        requested group reads them — the caller is expected to have gated their
        computation on :func:`needs_E_j` and :func:`needs_A_i`, which is the point of
        those predicates. ``"choice_set"`` alone needs neither.
    seg : ndarray
        Segment offsets over the surviving pairs.
    Q : int, optional
    G : ndarray, optional
        Selection probabilities. Required by ``"choice_set"``, which is why that group is
        only available from Q-restricted families.
    open_set : ndarray of bool, optional
        The resolved open sites. ``"exposure"`` and ``"inequality"`` restrict their Gini
        coefficients to these; omitted, every site counts.
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
            out.update(_inequality(prep, E_j, A_i, open_set))
        elif group == "exposure":
            out.update(_exposure(prep, E_j, open_set))
        elif group == "choice_set":
            if G is None:
                raise ValueError(
                    f"stat group 'choice_set' needs a choice set; {family} has none "
                    "(use sfca or sfca_e, or ifca with Q set)"
                )
            out.update(_choice_set(G, seg, Q))
    return out
