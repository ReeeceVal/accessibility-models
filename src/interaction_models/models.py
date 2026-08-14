"""The four model families and the ``Result`` they return.

Each family is a single model. Its optional terms — impedance, mode split, a bounded
choice set — switch on when the corresponding parameter is given, so the configuration
is fully described by ``res.params``.

``E_j`` is the primary supply-side output in every family; ``R_j = S_j / E_j`` is a
diagnostic only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import _core
from .modes import impedance, is_kappa_only, reach_weight
from .prepare import Prepared, prepare
from .stats import compute_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .modes import Mode

__all__ = ["Result", "catchment", "ifca", "sfca", "voronoi"]

_DEFAULT_OUTPUT = ("demand", "supply")


@dataclass
class Result:
    """What every model function returns.

    Attributes
    ----------
    params : dict
        The resolved configuration: ``D_max``, ``Q``, ``tau``, ``n_modes``. Keys that
        the family does not use are ``None``.
    demand : DataFrame or None
        ``demand_id | demand | A_i | SPAR | n_supply_i | <passthrough>``, plus ``r_i``
        for the iFCA family. ``None`` when ``"demand"`` is not in ``output``.
    supply : DataFrame or None
        ``supply_id | capacity | E_j | R_j | n_demand_j | <passthrough>``. ``None`` when
        ``"supply"`` is not in ``output``.
    stats : dict of str to float or None
        Flat statistics, ``None`` unless ``stats=[...]`` was requested.
    """

    params: dict
    demand: pd.DataFrame | None = None
    supply: pd.DataFrame | None = None
    stats: dict[str, float] | None = None


@dataclass(frozen=True)
class _Selection:
    """The pair set surviving stages 1-6, plus its impedance."""

    rows: np.ndarray
    f: np.ndarray
    seg: np.ndarray
    demand_code: np.ndarray
    supply_code: np.ndarray

    @property
    def n_pairs(self) -> int:
        return self.rows.size


def _as_prepared(
    first: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None,
    cost_df: pd.DataFrame | None,
    modes: Sequence[Mode] | None,
    validate: bool,
) -> Prepared:
    if isinstance(first, Prepared):
        if supply_df is not None or cost_df is not None:
            raise TypeError("pass either a Prepared or three DataFrames, not both")
        return first
    if supply_df is None or cost_df is None:
        raise TypeError("pass a Prepared, or all three of demand_df, supply_df, cost_df")
    return prepare(first, supply_df, cost_df, modes=modes, validate=validate)


def _select(
    prep: Prepared,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    Q: int | None,
    use_impedance: bool,
) -> _Selection:
    """Pipeline stages 1-6, returning the surviving pairs and their ``f_multi``."""
    rows = np.flatnonzero(prep.cost <= D_max)

    if use_impedance:
        f = impedance(prep, modes, rows)
        keep = f >= tau
        rows, f = rows[keep], f[keep]
    else:
        f = np.ones(rows.size, dtype=np.float64)

    demand_code = prep.demand_code[rows]
    seg = _core.segment_starts(demand_code, prep.n_demand)

    if Q is not None and np.isfinite(Q):
        kappa_only = modes is None or is_kappa_only(modes)
        if prep.has_rank and not kappa_only:
            warnings.warn(
                "cost_df.rank is ignored: with a mode on its own cost column, rank by "
                "cost is not rank by f_multi",
                UserWarning,
                stacklevel=3,
            )
        if kappa_only:
            rank = _core.segment_position(seg, rows.size)
        else:
            rank = _core.segment_rank_desc(f, demand_code, seg)
        keep = rank < Q
        rows, f, demand_code = rows[keep], f[keep], demand_code[keep]
        seg = _core.segment_starts(demand_code, prep.n_demand)

    return _Selection(
        rows=rows,
        f=f,
        seg=seg,
        demand_code=demand_code,
        supply_code=prep.supply_code[rows],
    )


def _require_capacity(prep: Prepared, why: str) -> np.ndarray:
    if prep.S is None:
        raise ValueError(f"supply_df needs a 'capacity' column to compute {why}")
    return prep.S


def _spar(A_i: np.ndarray) -> np.ndarray:
    mean = A_i.mean() if A_i.size else np.nan
    return A_i / mean if mean else np.full(A_i.size, np.nan)


def _frame(base: pd.DataFrame, lead: list[str], computed: dict[str, np.ndarray]) -> pd.DataFrame:
    """Lead columns, then computed columns, then every other input column untouched."""
    base = base.reset_index(drop=True)
    out = pd.DataFrame({name: base[name] for name in lead if name in base.columns})
    for name, values in computed.items():
        out[name] = values
    passthrough = [c for c in base.columns if c not in out.columns]
    return pd.concat([out, base[passthrough]], axis=1) if passthrough else out


def _assemble(
    prep: Prepared,
    params: dict,
    output: Sequence[str],
    demand_cols: dict[str, np.ndarray] | None,
    supply_cols: dict[str, np.ndarray] | None,
) -> Result:
    return Result(
        params=params,
        demand=(
            _frame(prep.demand_df, ["demand_id", "demand"], demand_cols)
            if "demand" in output and demand_cols is not None
            else None
        ),
        supply=(
            _frame(prep.supply_df, ["supply_id", "capacity"], supply_cols)
            if "supply" in output and supply_cols is not None
            else None
        ),
    )


def _finish(
    prep: Prepared,
    params: dict,
    output: Sequence[str],
    demand_cols: dict[str, np.ndarray] | None,
    supply_cols: dict[str, np.ndarray] | None,
    *,
    stats: Sequence[str] | None,
    n_pairs_used: int,
    seg: np.ndarray,
    family: str,
    G: np.ndarray | None = None,
) -> Result:
    """Assemble the frames and, if asked, the statistics."""
    result = _assemble(prep, params, output, demand_cols, supply_cols)
    if stats:
        result.stats = compute_stats(
            stats,
            prep=prep,
            n_pairs_used=n_pairs_used,
            E_j=supply_cols["E_j"],
            A_i=demand_cols["A_i"],
            seg=seg,
            Q=params["Q"],
            G=G,
            family=family,
        )
    return result


def _n_supply_i(sel: _Selection) -> np.ndarray:
    return _core.segment_lengths(sel.seg).astype(np.int64)


def _n_demand_j(sel: _Selection, n_supply: int) -> np.ndarray:
    return np.bincount(sel.supply_code, minlength=n_supply).astype(np.int64)


# --- Catchment ------------------------------------------------------------------


def catchment(
    prep: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    stats: Sequence[str] | None = None,
    output: Sequence[str] = _DEFAULT_OUTPUT,
    validate: bool = True,
) -> Result:
    """Cumulative-opportunity catchment, with no competition between sites.

    Every site accumulates the demand of every node that can reach it within ``D_max``.
    One node's demand may be counted by several sites at once.

    Formulae, over the pairs surviving ``D_max`` and ``tau``::

        E_j = sum_i P_i * f_multi(i, j)
        A_i = sum_j S_j * f_multi(i, j)

    Without a ``decay`` on any mode there is no impedance, ``f_multi = 1``, and the sums
    reduce to plain counts of demand and capacity within reach.

    Parameters
    ----------
    prep : Prepared or DataFrame
        A :class:`~interaction_models.prepare.Prepared`, or ``demand_df`` followed by
        ``supply_df`` and ``cost_df``.
    modes : sequence of Mode, optional
        Omit, or pass modes with no ``decay``, for the impedance-free form.
    D_max : float, default inf
        Maximum ``cost_default``, defining ``C_D(i)``.
    tau : float, default 0.0
        Impedance floor on ``f_multi``. Unused when there is no impedance.
    output : sequence of str, default ("demand", "supply")
        Which frames to assemble.
    validate : bool, default True
        Only consulted when raw frames are passed.

    Returns
    -------
    Result

    Raises
    ------
    ValueError
        If ``A_i`` is requested and ``supply_df`` has no ``capacity`` column.

    See Also
    --------
    docs/families/catchment.md
    """
    prep = _as_prepared(prep, supply_df, cost_df, modes, validate)
    use_impedance = bool(modes) and any(mode.decay is not None for mode in modes)

    sel = _select(prep, modes, D_max, tau, None, use_impedance)
    params = {
        "D_max": D_max,
        "Q": None,
        "tau": tau if use_impedance else None,
        "n_modes": len(modes) if modes else 0,
    }

    demand_cols = supply_cols = None
    if "supply" in output or stats:
        E_j = _core.group_sum(sel.supply_code, prep.P[sel.demand_code] * sel.f, prep.n_supply)
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(prep.S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or stats:
        S = _require_capacity(prep, "A_i")
        A_i = _core.segment_sum(S[sel.supply_code] * sel.f, sel.seg)
        demand_cols = {"A_i": A_i, "SPAR": _spar(A_i), "n_supply_i": _n_supply_i(sel)}

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="catchment",
    )


def _R_j(S: np.ndarray | None, E_j: np.ndarray) -> np.ndarray:
    """Diagnostic supply-to-exposure ratio; NaN where ``E_j == 0`` or capacity is absent."""
    if S is None:
        return np.full(E_j.size, np.nan)
    return np.divide(S, E_j, out=np.full(E_j.size, np.nan), where=E_j > 0)


# --- Voronoi --------------------------------------------------------------------


def voronoi(
    prep: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    stats: Sequence[str] | None = None,
    output: Sequence[str] = _DEFAULT_OUTPUT,
    validate: bool = True,
) -> Result:
    """Closest-facility assignment: all of a node's demand goes to its nearest site.

    ``j*(i) = argmin_{j in C_D(i)} cost_default``. Ties break on the lowest supply code,
    deterministically.

    Formulae::

        E_j = sum_{i: j = j*(i)} P_i * w_i
        w_i = sum_m pi_m * 1[kappa_m * cost_m(i, j*) <= D_max]     (1 without modes)
        A_i = S_{j*} / E_{j*}

    Decay is ignored — this family has no impedance, so mode split enters only as a
    reach gate and can never change ``j*``. The gate does **not** renormalise: a share
    whose mode cannot reach the assigned site is lost, which is the modelled penalty.

    Parameters
    ----------
    prep : Prepared or DataFrame
    modes : sequence of Mode, optional
        Only ``share``, ``cost`` and ``kappa`` are read; ``decay`` is ignored.
    D_max : float, default inf
    output : sequence of str, default ("demand", "supply")
    validate : bool, default True

    Returns
    -------
    Result
        ``n_supply_i`` is 1 for an assigned node and 0 for an unreachable one.

    See Also
    --------
    docs/families/voronoi.md
    """
    prep = _as_prepared(prep, supply_df, cost_df, modes, validate)

    rows = np.flatnonzero(prep.cost <= D_max)
    seg = _core.segment_starts(prep.demand_code[rows], prep.n_demand)
    assigned, first = _core.segment_first_rows(seg)
    star_rows = rows[first]
    star_supply = prep.supply_code[star_rows]

    w = (
        reach_weight(prep, modes, star_rows, D_max)
        if modes
        else np.ones(star_rows.size, dtype=np.float64)
    )
    E_j = _core.group_sum(star_supply, prep.P[assigned] * w, prep.n_supply)
    params = {"D_max": D_max, "Q": None, "tau": None, "n_modes": len(modes) if modes else 0}

    demand_cols = supply_cols = None
    if "supply" in output or stats:
        n_demand_j = np.bincount(star_supply, minlength=prep.n_supply).astype(np.int64)
        supply_cols = {"E_j": E_j, "R_j": _R_j(prep.S, E_j), "n_demand_j": n_demand_j}
    if "demand" in output or stats:
        S = _require_capacity(prep, "A_i")
        A_i = np.zeros(prep.n_demand, dtype=np.float64)
        exposure = E_j[star_supply]
        A_i[assigned] = np.divide(
            S[star_supply], exposure, out=np.zeros(star_rows.size), where=exposure > 0
        )
        n_supply_i = np.zeros(prep.n_demand, dtype=np.int64)
        n_supply_i[assigned] = 1
        demand_cols = {"A_i": A_i, "SPAR": _spar(A_i), "n_supply_i": n_supply_i}

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=star_rows.size,
        seg=_core.segment_starts(assigned, prep.n_demand), family="voronoi",
    )


# --- iFCA -----------------------------------------------------------------------


def _q_is_set(Q: float | None) -> bool:
    return Q is not None and np.isfinite(Q)


def _wants_choice_set(stats: Sequence[str] | None) -> bool:
    return bool(stats) and "choice_set" in stats


def ifca(
    prep: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode],
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int | None = None,
    stats: Sequence[str] | None = None,
    output: Sequence[str] = _DEFAULT_OUTPUT,
    validate: bool = True,
) -> Result:
    """Inverted floating catchment area (Wang 2018), with competition for demand.

    A node first works out how thinly its demand is spread across the capacity it can
    reach (``r_i``), then each site accumulates the resulting per-unit demand.

    Formulae, with denominators running over ``C_D(i)``, or ``C_Q(i)`` when ``Q`` is
    set::

        r_i = P_i / sum_j S_j * f_ij
        C_j = sum_i r_i * f_ij
        E_j = S_j * C_j
        A_i = 1 / r_i = sum_j S_j * f_ij / P_i

    Parameters
    ----------
    prep : Prepared or DataFrame
    modes : sequence of Mode
        Required — this family is undefined without impedance.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor, applied before the choice set is formed.
    Q : int, optional
        Choice-set size. ``None`` or ``inf`` leaves the denominator over ``C_D(i)``.
    output : sequence of str, default ("demand", "supply")
    validate : bool, default True

    Returns
    -------
    Result
        The demand frame carries ``r_i`` alongside ``A_i``.

    Raises
    ------
    ValueError
        If ``modes`` is empty, or ``supply_df`` has no ``capacity`` column — ``S_j``
        appears in the denominator, so capacity is always required here.

    Notes
    -----
    ``r_i``'s denominator runs over ``C_D(i)``: ``D_max`` is a common bound on every
    sum in the package, this one included.

    Nodes whose denominator is zero get ``r_i = inf``, ``A_i = 0``, and contribute
    nothing to any ``C_j``.

    See Also
    --------
    docs/families/ifca.md
    """
    prep = _as_prepared(prep, supply_df, cost_df, modes, validate)
    if not modes:
        raise ValueError("ifca requires modes=[...]; it is undefined without impedance")

    S = _require_capacity(prep, "the iFCA family (S_j is in the denominator)")
    sel = _select(prep, modes, D_max, tau, Q, True)
    params = {
        "D_max": D_max,
        "Q": Q if _q_is_set(Q) else None,
        "tau": tau,
        "n_modes": len(modes),
    }

    supply_within_reach = _core.segment_sum(S[sel.supply_code] * sel.f, sel.seg)
    reached = supply_within_reach > 0

    demand_cols = supply_cols = None
    if "supply" in output or stats:
        # r_i is 0, not inf, for unreached nodes: they contribute nothing to any C_j.
        r_contrib = np.divide(prep.P, supply_within_reach, out=np.zeros(prep.n_demand), where=reached)
        C_j = _core.group_sum(sel.supply_code, r_contrib[sel.demand_code] * sel.f, prep.n_supply)
        E_j = S * C_j
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or stats:
        A_i = np.divide(supply_within_reach, prep.P, out=np.zeros(prep.n_demand), where=reached)
        r_i = np.divide(
            prep.P, supply_within_reach, out=np.full(prep.n_demand, np.inf), where=reached
        )
        demand_cols = {
            "A_i": A_i,
            "SPAR": _spar(A_i),
            "r_i": r_i,
            "n_supply_i": _n_supply_i(sel),
        }

    G = _selection_probability(sel) if _wants_choice_set(stats) and _q_is_set(Q) else None
    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="ifca", G=G,
    )


# --- 3SFCA ----------------------------------------------------------------------


def _selection_probability(sel: _Selection) -> np.ndarray:
    """``G_ij = f_ij / sum_{k in C_Q(i)} f_ik``.

    Sums to 1.0 within every non-empty choice set, and to 0.0 within a set whose
    impedances are all zero.
    """
    denom = _core.segment_sum(sel.f, sel.seg)[sel.demand_code]
    return np.divide(sel.f, denom, out=np.zeros(sel.n_pairs), where=denom > 0)


def sfca(
    prep: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode],
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int,
    stats: Sequence[str] | None = None,
    output: Sequence[str] = _DEFAULT_OUTPUT,
    validate: bool = True,
) -> Result:
    """Three-step floating catchment area, with a demand-side selection step.

    Each node distributes its demand across its choice set in proportion to relative
    impedance, so the selection weight already yields an expected count and there is no
    further multiplication by ``S_j``.

    Formulae::

        G_ij = f_ij / sum_{k in C_Q(i)} f_ik
        E_j  = sum_{i: j in C_Q(i)} P_i * G_ij * f_ij
        R_j  = S_j / E_j                                  (diagnostic)
        A_i  = sum_j R_j * G_ij * f_ij

    Parameters
    ----------
    prep : Prepared or DataFrame
    modes : sequence of Mode
        Required.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor, applied before ``G_ij`` is normalised.
    Q : int
        Choice-set size. **Required** by this family — the selection probability is
        defined over ``C_Q(i)``.
    output : sequence of str, default ("demand", "supply")
    validate : bool, default True

    Returns
    -------
    Result

    Raises
    ------
    ValueError
        If ``modes`` is empty, ``Q`` is not a finite positive integer, or ``A_i`` is
        requested without a ``capacity`` column.

    See Also
    --------
    docs/families/sfca.md
    """
    prep = _as_prepared(prep, supply_df, cost_df, modes, validate)
    if not modes:
        raise ValueError("sfca requires modes=[...]")
    if not _q_is_set(Q) or Q < 1:
        raise ValueError("sfca requires a finite Q >= 1; G_ij is defined over C_Q(i)")

    sel = _select(prep, modes, D_max, tau, Q, True)
    G = _selection_probability(sel)
    drawn = prep.P[sel.demand_code] * G * sel.f
    E_j = _core.group_sum(sel.supply_code, drawn, prep.n_supply)
    params = {"D_max": D_max, "Q": Q, "tau": tau, "n_modes": len(modes)}

    demand_cols = supply_cols = None
    if "supply" in output or stats:
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(prep.S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or stats:
        S = _require_capacity(prep, "A_i")
        # R_j is NaN where E_j == 0, but such a site draws no pair, so 0.0 is safe here.
        R_finite = np.divide(S, E_j, out=np.zeros(prep.n_supply), where=E_j > 0)
        A_i = _core.segment_sum(R_finite[sel.supply_code] * G * sel.f, sel.seg)
        demand_cols = {"A_i": A_i, "SPAR": _spar(A_i), "n_supply_i": _n_supply_i(sel)}

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="sfca", G=G,
    )
