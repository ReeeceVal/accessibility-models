"""The five model families and the ``Result`` they return.

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
from .compiled import Compiled, _open_rows, _validate_mask
from .modes import impedance, is_kappa_only, reach_weight
from .prepare import Prepared, prepare
from .stats import compute_stats, needs_A_i, needs_E_j

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .modes import Mode

__all__ = ["Result", "catchment", "ifca", "sfca", "sfca_e", "voronoi"]

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
        for the iFCA family and ``Phi_i`` for MAC-3SFCA-E. ``None`` when ``"demand"`` is
        not in ``output``.
    supply : DataFrame or None
        ``supply_id | capacity | E_j | R_j | n_demand_j | <passthrough>``, plus ``L_j``
        for MAC-3SFCA-E. ``None`` when ``"supply"`` is not in ``output``.
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


def _open_set(prep: Prepared, open_mask: np.ndarray | None) -> np.ndarray | None:
    """The resolved open sites, or ``None`` when every site is open."""
    return _validate_mask(open_mask, prep.n_supply, "open_mask")


def _n_open(open_set: np.ndarray | None) -> int | None:
    return None if open_set is None else int(np.count_nonzero(open_set))


def _reject_baked(modes: Sequence[Mode] | None, D_max: float, tau: float) -> None:
    """A Compiled already fixes stages 0-5; repeating them here would be ignored."""
    baked = [
        name
        for name, given in (("modes", modes is not None), ("D_max", D_max != np.inf), ("tau", tau != 0.0))
        if given
    ]
    if baked:
        raise ValueError(
            f"{', '.join(baked)} {'is' if len(baked) == 1 else 'are'} baked into the "
            "Compiled passed here; set them on compile_f() instead"
        )


def _select_compiled(comp: Compiled, Q: int | None, open_mask: np.ndarray | None) -> _Selection:
    """Stages 0 and 6 over an already-compiled ``f_multi``.

    No impedance and no sort: the compiled rows are already ``f``-descending within each
    segment, and masking preserves relative order, so the rank is a prefix truncation
    whatever the mode set.
    """
    mask = _validate_mask(open_mask, comp.prep.n_supply, "open_mask")
    if mask is None:
        sel = np.arange(comp.n_pairs, dtype=np.int64)
    else:
        if comp.sites is not None and (mask & ~comp.sites).any():
            raise ValueError(
                "open_mask selects sites excluded by compile_f(sites=...); it must be a "
                "subset of the sites the Compiled was built over"
            )
        sel = np.flatnonzero(mask[comp.supply_code])

    demand_code = comp.demand_code[sel]
    seg = _core.segment_starts(demand_code, comp.prep.n_demand)

    if Q is not None and np.isfinite(Q):
        keep = _core.segment_position(seg, sel.size) < Q
        sel, demand_code = sel[keep], demand_code[keep]
        seg = _core.segment_starts(demand_code, comp.prep.n_demand)

    return _Selection(
        rows=sel,
        f=comp.f[sel],
        seg=seg,
        demand_code=demand_code,
        supply_code=comp.supply_code[sel],
    )


def _resolve(
    first: Prepared | Compiled | pd.DataFrame,
    supply_df: pd.DataFrame | None,
    cost_df: pd.DataFrame | None,
    *,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    Q: int | None,
    use_impedance: bool,
    open_mask: np.ndarray | None,
    validate: bool,
) -> tuple[Prepared, _Selection, float, float, int, int | None, np.ndarray | None]:
    """Run stages 0-6 from whichever of the three entry points the caller used.

    Returns the prepared inputs, the surviving pairs, the resolved ``D_max``, ``tau``,
    ``n_modes`` and ``n_open`` that belong in ``params``, and the open set those last two
    were read off — which the ``exposure`` and ``inequality`` statistics need in order to
    ignore closed sites.
    """
    if isinstance(first, Compiled):
        if supply_df is not None or cost_df is not None:
            raise TypeError("pass either a Compiled or three DataFrames, not both")
        _reject_baked(modes, D_max, tau)
        # A Compiled built with sites=... is already restricted, so that is the open set
        # whenever the call does not narrow it further.
        open_set = _open_set(first.prep, open_mask)
        if open_set is None:
            open_set = first.sites
        sel = _select_compiled(first, Q, open_mask)
        return (first.prep, sel, first.D_max, first.tau, first.n_modes,
                _n_open(open_set), open_set)

    prep = _as_prepared(first, supply_df, cost_df, modes, validate)
    sel = _select(prep, modes, D_max, tau, Q, use_impedance, open_mask)
    open_set = _open_set(prep, open_mask)
    return (prep, sel, D_max, tau, (len(modes) if modes else 0),
            _n_open(open_set), open_set)


def _select(
    prep: Prepared,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    Q: int | None,
    use_impedance: bool,
    open_mask: np.ndarray | None = None,
) -> _Selection:
    """Pipeline stages 0-6, returning the surviving pairs and their ``f_multi``."""
    rows = _open_rows(prep, np.flatnonzero(prep.cost <= D_max), open_mask)

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
    open_set: np.ndarray | None = None,
) -> Result:
    """Assemble the frames and, if asked, the statistics."""
    result = _assemble(prep, params, output, demand_cols, supply_cols)
    if stats:
        result.stats = compute_stats(
            stats,
            prep=prep,
            n_pairs_used=n_pairs_used,
            E_j=supply_cols["E_j"] if supply_cols is not None else None,
            A_i=demand_cols["A_i"] if demand_cols is not None else None,
            seg=seg,
            Q=params["Q"],
            G=G,
            open_set=open_set,
            family=family,
        )
    return result


def _n_supply_i(sel: _Selection) -> np.ndarray:
    return _core.segment_lengths(sel.seg).astype(np.int64)


def _n_demand_j(sel: _Selection, n_supply: int) -> np.ndarray:
    return np.bincount(sel.supply_code, minlength=n_supply).astype(np.int64)


# --- Catchment ------------------------------------------------------------------


def catchment(
    prep: Prepared | Compiled | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    open_mask: np.ndarray | None = None,
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
    prep : Prepared, Compiled or DataFrame
        A :class:`~interaction_models.prepare.Prepared`, a
        :class:`~interaction_models.compiled.Compiled` from
        :func:`~interaction_models.compile_f`, or ``demand_df`` followed by ``supply_df``
        and ``cost_df``. Given a ``Compiled``, ``modes``, ``D_max`` and ``tau`` are
        already fixed and passing them here raises.
    modes : sequence of Mode, optional
        Omit, or pass modes with no ``decay``, for the impedance-free form.
    D_max : float, default inf
        Maximum ``cost_default``, defining ``C_D(i)``.
    tau : float, default 0.0
        Impedance floor on ``f_multi``. Unused when there is no impedance.
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``. Closed sites are dropped before every other
        stage, so results match those of re-running on a ``cost_df`` restricted to the
        open sites. Output frames keep their full length, with zeros at closed sites.
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
    use_impedance = isinstance(prep, Compiled) or (
        bool(modes) and any(mode.decay is not None for mode in modes)
    )
    prep, sel, D_max, tau, n_modes, n_open, open_set = _resolve(
        prep, supply_df, cost_df, modes=modes, D_max=D_max, tau=tau, Q=None,
        use_impedance=use_impedance, open_mask=open_mask, validate=validate,
    )
    params = {
        "D_max": D_max,
        "Q": None,
        "tau": tau if use_impedance else None,
        "n_modes": n_modes,
        "n_open": n_open,
    }

    demand_cols = supply_cols = None
    if "supply" in output or needs_E_j(stats):
        E_j = _core.group_sum(sel.supply_code, prep.P[sel.demand_code] * sel.f, prep.n_supply)
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(prep.S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or needs_A_i(stats):
        S = _require_capacity(prep, "A_i")
        A_i = _core.segment_sum(S[sel.supply_code] * sel.f, sel.seg)
        demand_cols = {"A_i": A_i, "SPAR": _spar(A_i), "n_supply_i": _n_supply_i(sel)}

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="catchment", open_set=open_set,
    )


def _R_j(S: np.ndarray | None, E_j: np.ndarray) -> np.ndarray:
    """Diagnostic supply-to-exposure ratio; NaN where ``E_j == 0`` or capacity is absent."""
    if S is None:
        return np.full(E_j.size, np.nan)
    return np.divide(S, E_j, out=np.full(E_j.size, np.nan), where=E_j > 0)


def _L_j(S: np.ndarray | None, E_j: np.ndarray) -> np.ndarray:
    """Operational load per unit capacity; NaN where ``S_j == 0`` or capacity is absent."""
    if S is None:
        return np.full(E_j.size, np.nan)
    return np.divide(E_j, S, out=np.full(E_j.size, np.nan), where=S > 0)


# --- Voronoi --------------------------------------------------------------------


def voronoi(
    prep: Prepared | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    open_mask: np.ndarray | None = None,
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
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``. Closed sites cannot be assigned, so ``j*(i)``
        is the nearest **open** site.
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
    if isinstance(prep, Compiled):
        raise TypeError(
            "voronoi does not accept a Compiled: it has no impedance and assigns on "
            "cost_default order, not f_multi order. Pass the Prepared instead."
        )
    prep = _as_prepared(prep, supply_df, cost_df, modes, validate)
    open_set = _open_set(prep, open_mask)

    rows = _open_rows(prep, np.flatnonzero(prep.cost <= D_max), open_mask)
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
    params = {
        "D_max": D_max,
        "Q": None,
        "tau": None,
        "n_modes": len(modes) if modes else 0,
        "n_open": _n_open(open_set),
    }

    demand_cols = supply_cols = None
    if "supply" in output or needs_E_j(stats):
        n_demand_j = np.bincount(star_supply, minlength=prep.n_supply).astype(np.int64)
        supply_cols = {"E_j": E_j, "R_j": _R_j(prep.S, E_j), "n_demand_j": n_demand_j}
    if "demand" in output or needs_A_i(stats):
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
        seg=_core.segment_starts(assigned, prep.n_demand), family="voronoi", open_set=open_set,
    )


# --- iFCA -----------------------------------------------------------------------


def _q_is_set(Q: float | None) -> bool:
    return Q is not None and np.isfinite(Q)


def _wants_choice_set(stats: Sequence[str] | None) -> bool:
    return bool(stats) and "choice_set" in stats


def ifca(
    prep: Prepared | Compiled | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int | None = None,
    open_mask: np.ndarray | None = None,
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
    prep : Prepared, Compiled or DataFrame
        Given a :class:`~interaction_models.compiled.Compiled`, ``modes``, ``D_max`` and
        ``tau`` are already fixed and passing them here raises.
    modes : sequence of Mode
        Required — this family is undefined without impedance — unless a ``Compiled``
        was passed, which already carries them.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor, applied before the choice set is formed.
    Q : int, optional
        Choice-set size. ``None`` or ``inf`` leaves the denominator over ``C_D(i)``.
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``. Closed sites are dropped before ``r_i``'s
        denominator is formed, so they neither absorb demand nor dilute it.
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
    if not modes and not isinstance(prep, Compiled):
        raise ValueError("ifca requires modes=[...]; it is undefined without impedance")

    prep, sel, D_max, tau, n_modes, n_open, open_set = _resolve(
        prep, supply_df, cost_df, modes=modes, D_max=D_max, tau=tau, Q=Q,
        use_impedance=True, open_mask=open_mask, validate=validate,
    )
    S = _require_capacity(prep, "the iFCA family (S_j is in the denominator)")
    params = {
        "D_max": D_max,
        "Q": Q if _q_is_set(Q) else None,
        "tau": tau,
        "n_modes": n_modes,
        "n_open": n_open,
    }

    supply_within_reach = _core.segment_sum(S[sel.supply_code] * sel.f, sel.seg)
    reached = supply_within_reach > 0

    demand_cols = supply_cols = None
    if "supply" in output or needs_E_j(stats):
        # r_i is 0, not inf, for unreached nodes: they contribute nothing to any C_j.
        r_contrib = np.divide(prep.P, supply_within_reach, out=np.zeros(prep.n_demand), where=reached)
        C_j = _core.group_sum(sel.supply_code, r_contrib[sel.demand_code] * sel.f, prep.n_supply)
        E_j = S * C_j
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or needs_A_i(stats):
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
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="ifca", G=G, open_set=open_set,
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
    prep: Prepared | Compiled | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int,
    open_mask: np.ndarray | None = None,
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
    prep : Prepared, Compiled or DataFrame
        Given a :class:`~interaction_models.compiled.Compiled`, ``modes``, ``D_max`` and
        ``tau`` are already fixed and passing them here raises.
    modes : sequence of Mode
        Required, unless a ``Compiled`` was passed.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor, applied before ``G_ij`` is normalised.
    Q : int
        Choice-set size. **Required** by this family — the selection probability is
        defined over ``C_Q(i)``.
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``. Closed sites are dropped before the rank, so
        ``C_Q(i)`` is the top ``Q`` **open** options — not the open members of the
        top ``Q`` overall.
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
    if not modes and not isinstance(prep, Compiled):
        raise ValueError("sfca requires modes=[...]")
    if not _q_is_set(Q) or Q < 1:
        raise ValueError("sfca requires a finite Q >= 1; G_ij is defined over C_Q(i)")

    prep, sel, D_max, tau, n_modes, n_open, open_set = _resolve(
        prep, supply_df, cost_df, modes=modes, D_max=D_max, tau=tau, Q=Q,
        use_impedance=True, open_mask=open_mask, validate=validate,
    )
    G = _selection_probability(sel)
    drawn = prep.P[sel.demand_code] * G * sel.f
    E_j = _core.group_sum(sel.supply_code, drawn, prep.n_supply)
    params = {
        "D_max": D_max,
        "Q": Q,
        "tau": tau,
        "n_modes": n_modes,
        "n_open": n_open,
    }

    demand_cols = supply_cols = None
    if "supply" in output or needs_E_j(stats):
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(prep.S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or needs_A_i(stats):
        S = _require_capacity(prep, "A_i")
        # R_j is NaN where E_j == 0, but such a site draws no pair, so 0.0 is safe here.
        R_finite = np.divide(S, E_j, out=np.zeros(prep.n_supply), where=E_j > 0)
        A_i = _core.segment_sum(R_finite[sel.supply_code] * G * sel.f, sel.seg)
        demand_cols = {"A_i": A_i, "SPAR": _spar(A_i), "n_supply_i": _n_supply_i(sel)}

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="sfca", G=G, open_set=open_set,
    )


# --- MAC-3SFCA-E ----------------------------------------------------------------


def sfca_e(
    prep: Prepared | Compiled | pd.DataFrame,
    supply_df: pd.DataFrame | None = None,
    cost_df: pd.DataFrame | None = None,
    *,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int,
    open_mask: np.ndarray | None = None,
    stats: Sequence[str] | None = None,
    output: Sequence[str] = _DEFAULT_OUTPUT,
    validate: bool = True,
) -> Result:
    """Three-step FCA with an explicit participation step, monotone under site openings.

    3SFCA lets ``f_ij`` govern both *which* site a node selects and *whether* it travels
    at all, which makes ``Σ_j E_j`` fall when a below-average site opens. This family
    separates the two: ``Phi_i`` gates participation on the node's single best option,
    and ``G_ij`` — unchanged, and capacity-blind — allocates the participating demand.

    Formulae::

        Phi_i = max_{j in C_Q(i)} f_multi(i, j)
        G_ij  = f_ij / sum_{k in C_Q(i)} f_ik
        E_j   = sum_{i: j in C_Q(i)} P_i * Phi_i * G_ij
        L_j   = E_j / S_j                                 (operational load per unit capacity)
        R_j   = S_j / E_j                                 (diagnostic)
        A_i   = sum_j R_j * G_ij * f_ij

    Since ``sum_j G_ij = 1``, the network total collapses to
    ``Σ_j E_j = Σ_i P_i max_j f_ij`` — the classical facility-location function, which is
    monotone and submodular in the site set. Capacity never enters selection; it appears
    only in ``L_j`` and ``R_j``.

    Parameters
    ----------
    prep : Prepared, Compiled or DataFrame
        Given a :class:`~interaction_models.compiled.Compiled`, ``modes``, ``D_max`` and
        ``tau`` are already fixed and passing them here raises. That is the shape an
        optimisation loop wants: compile once, then vary ``open_mask``.
    modes : sequence of Mode
        Required, unless a ``Compiled`` was passed.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor, applied before ``C_Q(i)`` is formed.
    Q : int
        Choice-set size. **Required**, as for :func:`sfca` — ``G_ij`` is defined over
        ``C_Q(i)``. ``Phi_i`` is unaffected by it (see Notes).
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``, naming the sites that are open. Closed sites
        are dropped before the rank, so ``C_Q(i)`` is the top ``Q`` **open** options —
        not the open members of the top ``Q`` overall. This is the site-selection lever:
        ``Σ_j E_j`` is monotone in it, and results are identical to re-running on a
        ``cost_df`` restricted to the open sites, at a fraction of the cost.
    output : sequence of str, default ("demand", "supply")
    validate : bool, default True

    Returns
    -------
    Result
        The demand frame carries ``Phi_i`` alongside ``A_i``; the supply frame carries
        ``L_j`` alongside ``R_j``.

    Raises
    ------
    ValueError
        If ``modes`` is empty, ``Q`` is not a finite positive integer, or ``A_i`` is
        requested without a ``capacity`` column.

    Notes
    -----
    ``Phi_i`` is invariant to ``Q``: the maximising pair is necessarily the highest-ranked
    member of ``C_Q(i)``, so it survives any ``Q >= 1``. ``Q`` therefore governs how
    exposure is distributed across sites, never how much of it exists.

    ``Phi_i`` is the maximum of the impedances in the choice set where :func:`sfca` uses
    their contraharmonic mean. A weighted average never exceeds a maximum, so
    ``Σ_j E_j`` here is bounded below by the 3SFCA total, with equality when every
    impedance in the choice set is equal.

    See Also
    --------
    docs/families/sfca-e.md
    """
    if not modes and not isinstance(prep, Compiled):
        raise ValueError("sfca_e requires modes=[...]")
    if not _q_is_set(Q) or Q < 1:
        raise ValueError("sfca_e requires a finite Q >= 1; G_ij is defined over C_Q(i)")

    prep, sel, D_max, tau, n_modes, n_open, open_set = _resolve(
        prep, supply_df, cost_df, modes=modes, D_max=D_max, tau=tau, Q=Q,
        use_impedance=True, open_mask=open_mask, validate=validate,
    )
    G = _selection_probability(sel)
    Phi_i = _core.segment_max(sel.f, sel.seg)
    drawn = prep.P[sel.demand_code] * Phi_i[sel.demand_code] * G
    E_j = _core.group_sum(sel.supply_code, drawn, prep.n_supply)
    params = {
        "D_max": D_max,
        "Q": Q,
        "tau": tau,
        "n_modes": n_modes,
        "n_open": n_open,
    }

    demand_cols = supply_cols = None
    if "supply" in output or needs_E_j(stats):
        supply_cols = {
            "E_j": E_j,
            "R_j": _R_j(prep.S, E_j),
            "L_j": _L_j(prep.S, E_j),
            "n_demand_j": _n_demand_j(sel, prep.n_supply),
        }
    if "demand" in output or needs_A_i(stats):
        S = _require_capacity(prep, "A_i")
        # R_j is NaN where E_j == 0, but such a site draws no pair, so 0.0 is safe here.
        R_finite = np.divide(S, E_j, out=np.zeros(prep.n_supply), where=E_j > 0)
        A_i = _core.segment_sum(R_finite[sel.supply_code] * G * sel.f, sel.seg)
        demand_cols = {
            "A_i": A_i,
            "SPAR": _spar(A_i),
            "Phi_i": Phi_i,
            "n_supply_i": _n_supply_i(sel),
        }

    return _finish(
        prep, params, output, demand_cols, supply_cols,
        stats=stats, n_pairs_used=sel.n_pairs, seg=sel.seg, family="sfca_e", G=G, open_set=open_set,
    )
