"""Per-pair breakdown for one demand node or one supply point (or a short list of them).

Every model family sums per-pair terms into a demand-node or supply-point aggregate and
discards the individual values -- that is what keeps a call over 10-100M pairs cheap.
``explain()`` recovers them, scoped by exactly one of ``demand_id`` or ``supply_id``:
the "click a node, see its choice set" panel and the "click a site, see who it pulls"
view an interactive drill-down needs.

The two axes are not symmetric under the hood. ``Prepared``/``Compiled`` are sorted by
demand only, so a ``demand_id`` query is a bounded segment slice, while a ``supply_id``
query first pays a one-shot ``O(n_pairs)`` scan to find which demand nodes reach the
named site(s) -- then, for the competitive families, re-runs the same bounded per-node
computation a ``demand_id`` query would, since a pair's weight there depends on that
node's whole choice set, not just the one site being asked about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .compiled import Compiled, _validate_mask
from .models import _q_is_set, _reject_baked, _require_capacity
from .modes import impedance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .modes import Mode
    from .prepare import Prepared

__all__ = ["explain"]

_FAMILIES = ("catchment", "ifca", "sfca", "sfca_e", "voronoi")


def _base_of(prep: Prepared | Compiled) -> Prepared:
    return prep.prep if isinstance(prep, Compiled) else prep


def _resolve_demand_codes(base: Prepared, demand_id) -> tuple[np.ndarray, list]:
    ids = list(demand_id) if pd.api.types.is_list_like(demand_id) else [demand_id]
    if not ids:
        raise ValueError("demand_id must name at least one demand node")
    codes = pd.Index(base.demand_df["demand_id"]).get_indexer(ids)
    missing = [i for i, c in zip(ids, codes, strict=True) if c < 0]
    if missing:
        raise ValueError(f"demand_id not found in demand_df: {missing!r}")
    return codes.astype(np.int64), ids


def _resolve_supply_codes(base: Prepared, supply_id) -> tuple[np.ndarray, list]:
    ids = list(supply_id) if pd.api.types.is_list_like(supply_id) else [supply_id]
    if not ids:
        raise ValueError("supply_id must name at least one supply point")
    codes = pd.Index(base.supply_df["supply_id"]).get_indexer(ids)
    missing = [i for i, c in zip(ids, codes, strict=True) if c < 0]
    if missing:
        raise ValueError(f"supply_id not found in supply_df: {missing!r}")
    return codes.astype(np.int64), ids


def _demand_codes_reaching(
    prep: Prepared | Compiled,
    base: Prepared,
    supply_codes: np.ndarray,
    D_max: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Demand nodes with a surviving pair to any of ``supply_codes``.

    The one place a ``supply_id`` query pays for what ``demand_id`` gets for free:
    Prepared/Compiled are sorted by demand only, so finding the other direction costs a
    one-shot scan over every pair. Everything downstream of this stays bounded to each
    candidate node's own segment, exactly as a ``demand_id`` query already is.
    """
    if isinstance(prep, Compiled):
        rows = np.flatnonzero(np.isin(prep.supply_code, supply_codes))
        if mask is not None:
            rows = rows[mask[prep.supply_code[rows]]]
        return np.unique(prep.demand_code[rows])
    rows = np.flatnonzero(np.isin(base.supply_code, supply_codes))
    rows = rows[base.cost[rows] <= D_max]
    if mask is not None:
        rows = rows[mask[base.supply_code[rows]]]
    return np.unique(base.demand_code[rows])


_EMPTY_DTYPES = {
    "demand_id": object,
    "supply_id": object,
    "cost_default": np.float64,
    "f_multi": np.float64,
    "weight": np.float64,
}


def _explain_by_supply(
    prep: Prepared | Compiled,
    base: Prepared,
    family: str,
    supply_codes: np.ndarray,
    supply_ids: list,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    Q: int | None,
    mask: np.ndarray | None,
) -> pd.DataFrame:
    """Every surviving pair for one supply point (or a short list), across demand nodes.

    Loops the bounded per-node computation over each candidate node found by
    :func:`_demand_codes_reaching`, then keeps only the requested site(s)' own row from
    each -- the same per-node work a ``demand_id`` query does, just addressed by site
    instead of collected under it.
    """
    demand_codes = _demand_codes_reaching(prep, base, supply_codes, D_max, mask)
    if demand_codes.size == 0:
        return pd.DataFrame({c: pd.array([], dtype=dt) for c, dt in _EMPTY_DTYPES.items()})

    demand_id_col = base.demand_df["demand_id"].to_numpy()
    if family == "voronoi":
        frames = [
            _explain_voronoi_one(base, code, demand_id_col[code], D_max, mask)
            for code in demand_codes
        ]
    else:
        frames = [
            _explain_one(prep, base, family, code, demand_id_col[code], modes, D_max, tau, Q, mask)
            for code in demand_codes
        ]
    result = pd.concat(frames, ignore_index=True)
    result = result[result["supply_id"].isin(supply_ids)]
    result = result.sort_values(["weight", "cost_default"], ascending=[False, True], kind="stable")
    return result.reset_index(drop=True)


def _cost_lookup(base: Prepared, code: int, supply_code: np.ndarray) -> np.ndarray:
    """``cost_default`` for a Compiled's surviving pairs, read off the Prepared node slice.

    A ``Compiled`` keeps ``f_multi`` but not ``cost_default`` for the pairs it selects --
    carrying it would cost an extra ``n_pairs`` array at the scale ``Compiled`` is built
    for. Cheap here because it never leaves one node's own segment.
    """
    lo, hi = base.seg_start[code], base.seg_start[code + 1]
    node_supply = base.supply_code[lo:hi]
    node_cost = base.cost[lo:hi]
    order = np.argsort(node_supply, kind="stable")
    idx = np.searchsorted(node_supply[order], supply_code)
    return node_cost[order][idx]


def _node_rows_prepared(
    base: Prepared,
    code: int,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    use_impedance: bool,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """One demand node's surviving rows, ranked ``f`` descending.

    A plain slice of ``seg_start`` plus a local sort -- no segment machinery, since a
    single node's candidate list is small whatever the network size.
    """
    lo, hi = base.seg_start[code], base.seg_start[code + 1]
    rows = np.arange(lo, hi, dtype=np.int64)
    rows = rows[base.cost[rows] <= D_max]
    if mask is not None:
        rows = rows[mask[base.supply_code[rows]]]
    if use_impedance:
        f = impedance(base, modes, rows)
        keep = f >= tau
        rows, f = rows[keep], f[keep]
    else:
        f = np.ones(rows.size, dtype=np.float64)
    order = np.argsort(-f, kind="stable")
    return rows[order], f[order]


def _node_rows_compiled(comp: Compiled, code: int, mask: np.ndarray | None) -> np.ndarray:
    """One demand node's surviving rows from a Compiled -- already ``f``-descending."""
    lo, hi = comp.seg[code], comp.seg[code + 1]
    rows = np.arange(lo, hi, dtype=np.int64)
    if mask is not None:
        rows = rows[mask[comp.supply_code[rows]]]
    return rows


def _weight(
    family: str, base: Prepared, code: int, supply_code: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """The per-pair quantity each family sums away: ``P_i f_ij``, ``r_i f_ij`` or ``G_ij``."""
    if family == "catchment":
        return base.P[code] * f
    if family == "ifca":
        denom = float(np.sum(base.S[supply_code] * f))
        r_contrib = base.P[code] / denom if denom > 0 else 0.0
        return r_contrib * f
    denom = float(f.sum())  # sfca, sfca_e: G_ij
    return f / denom if denom > 0 else np.zeros_like(f)


def _explain_one(
    prep: Prepared | Compiled,
    base: Prepared,
    family: str,
    code: int,
    demand_id_value,
    modes: Sequence[Mode] | None,
    D_max: float,
    tau: float,
    Q: int | None,
    mask: np.ndarray | None,
) -> pd.DataFrame:
    if isinstance(prep, Compiled):
        rows = _node_rows_compiled(prep, code, mask)
        f = prep.f[rows]
        supply_code = prep.supply_code[rows]
    else:
        use_impedance = family != "catchment" or (
            bool(modes) and any(mode.decay is not None for mode in modes)
        )
        rows, f = _node_rows_prepared(base, code, modes, D_max, tau, use_impedance, mask)
        supply_code = base.supply_code[rows]

    if family in ("ifca", "sfca", "sfca_e") and _q_is_set(Q):
        n = int(np.ceil(Q))
        rows, f, supply_code = rows[:n], f[:n], supply_code[:n]

    cost_default = (
        _cost_lookup(base, code, supply_code) if isinstance(prep, Compiled) else base.cost[rows]
    )
    weight = _weight(family, base, code, supply_code, f)
    supply_ids = base.supply_df["supply_id"].to_numpy()[supply_code]
    return pd.DataFrame(
        {
            "demand_id": demand_id_value,
            "supply_id": supply_ids,
            "cost_default": cost_default,
            "f_multi": f,
            "weight": weight,
        }
    )


def _explain_voronoi_one(
    base: Prepared, code: int, demand_id_value, D_max: float, mask: np.ndarray | None
) -> pd.DataFrame:
    lo, hi = base.seg_start[code], base.seg_start[code + 1]
    rows = np.arange(lo, hi, dtype=np.int64)
    rows = rows[base.cost[rows] <= D_max]
    if mask is not None:
        rows = rows[mask[base.supply_code[rows]]]
    weight = np.zeros(rows.size, dtype=np.float64)
    if rows.size:
        weight[0] = 1.0  # rows are cost-ascending, supply_code-ascending tie-break -- j*(i)
    supply_ids = base.supply_df["supply_id"].to_numpy()[base.supply_code[rows]]
    return pd.DataFrame(
        {
            "demand_id": demand_id_value,
            "supply_id": supply_ids,
            "cost_default": base.cost[rows],
            "f_multi": np.ones(rows.size, dtype=np.float64),
            "weight": weight,
        }
    )


def explain(
    prep: Prepared | Compiled,
    *,
    family: str,
    demand_id=None,
    supply_id=None,
    modes: Sequence[Mode] | None = None,
    D_max: float = np.inf,
    tau: float = 0.0,
    Q: int | None = None,
    open_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """The per-pair terms a model call sums away, for one node scoped by either axis.

    Every family reduces ``C_D(i)`` (or ``C_Q(i)``) to a single ``A_i`` and ``E_j``.
    ``explain()`` stops one step earlier and returns the surviving pairs themselves --
    what an interactive drill-down needs for "this node reaches these sites, in this order, with this much
    weight on each" (``demand_id``) or "this site is reached by these nodes, this
    strongly" (``supply_id``). Exactly one of the two must be given.

    A ``demand_id`` query only ever reads the named node(s)' own segment, since
    Prepared/Compiled are already sorted that way. A ``supply_id`` query pays a one-shot
    scan over every pair to find which demand nodes reach the named site(s) -- and for
    the competitive families (``ifca``, ``sfca``, ``sfca_e``), a pair's weight depends on
    that node's *whole* choice set, not just the site asked about, so each candidate node
    is run through the same bounded per-node computation a ``demand_id`` query would do.

    Parameters
    ----------
    prep : Prepared or Compiled
        As passed to the matching model function. Given a
        :class:`~interaction_models.compiled.Compiled`, ``modes``, ``D_max`` and ``tau``
        are already baked in and passing them here raises, exactly as in the model
        functions.
    family : {"catchment", "ifca", "sfca", "sfca_e", "voronoi"}
        Which family's per-pair term to compute.
    demand_id : scalar or sequence, optional
        One demand node's id, or a short list of them. Mutually exclusive with
        ``supply_id``.
    supply_id : scalar or sequence, optional
        One supply point's id, or a short list of them. Mutually exclusive with
        ``demand_id``.
    modes : sequence of Mode, optional
        Required by ``ifca``, ``sfca`` and ``sfca_e`` unless a ``Compiled`` was passed.
        Unused by ``voronoi``, which has no impedance.
    D_max : float, default inf
    tau : float, default 0.0
        Impedance floor. Not accepted by ``voronoi``, which has no ``tau``.
    Q : int, optional
        Choice-set size. Required by ``sfca`` and ``sfca_e``, optional for ``ifca``, not
        accepted by ``catchment`` or ``voronoi`` -- exactly as in the model functions.
    open_mask : ndarray of bool, optional
        One flag per row of ``supply_df``. Closed sites are dropped before ranking, as in
        every model function.

    Returns
    -------
    DataFrame
        ``demand_id | supply_id | cost_default | f_multi | weight``, one row per
        surviving pair. A ``demand_id`` query ranks by ``f_multi`` descending
        (``cost_default`` ascending for ``voronoi``, which has no impedance) within each
        node; a ``supply_id`` query ranks by ``weight`` descending, ``cost_default``
        ascending, across whichever nodes reach the named site(s). ``weight`` is the
        per-pair term the family sums away before it ever reaches ``E_j``: ``P_i * f_ij``
        for ``catchment``, ``r_i * f_ij`` for ``ifca``, ``G_ij`` for ``sfca``/``sfca_e``,
        and ``1.0`` on the assigned pair (``0.0`` elsewhere) for ``voronoi``.

    Raises
    ------
    ValueError
        If neither or both of ``demand_id``/``supply_id`` are given, on an unknown
        ``family``, an id absent from ``demand_df``/``supply_df``, a parameter the
        family does not accept, or a missing requirement (``modes``, ``Q``,
        ``capacity``).
    TypeError
        If ``voronoi`` is asked to explain a ``Compiled``.

    See Also
    --------
    docs/explain.md
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
    if (demand_id is None) == (supply_id is None):
        raise ValueError("explain() takes exactly one of demand_id or supply_id")

    base = _base_of(prep)

    if family == "voronoi":
        if isinstance(prep, Compiled):
            raise TypeError("voronoi has no impedance; pass the Prepared, not a Compiled")
        if tau != 0.0 or Q is not None:
            raise ValueError("voronoi does not accept tau or Q")
        mask = _validate_mask(open_mask, base.n_supply, "open_mask")
        if demand_id is not None:
            codes, ids = _resolve_demand_codes(base, demand_id)
            frames = [
                _explain_voronoi_one(base, code, demand_id_value, D_max, mask)
                for code, demand_id_value in zip(codes, ids, strict=True)
            ]
            return pd.concat(frames, ignore_index=True)
        supply_codes, supply_ids = _resolve_supply_codes(base, supply_id)
        return _explain_by_supply(
            base, base, "voronoi", supply_codes, supply_ids, None, D_max, tau, Q, mask
        )

    if family in ("ifca", "sfca", "sfca_e") and not modes and not isinstance(prep, Compiled):
        raise ValueError(f"{family} requires modes=[...]; it is undefined without impedance")
    if family in ("sfca", "sfca_e") and (not _q_is_set(Q) or Q < 1):
        raise ValueError(f"{family} requires a finite Q >= 1; G_ij is defined over C_Q(i)")
    if family == "catchment" and Q is not None:
        raise ValueError("catchment does not accept Q; it has no choice set")
    if family == "ifca":
        _require_capacity(base, "the iFCA family (S_j is in the denominator)")

    if isinstance(prep, Compiled):
        _reject_baked(modes, D_max, tau)
        mask = _validate_mask(open_mask, base.n_supply, "open_mask")
        if mask is not None and prep.sites is not None and (mask & ~prep.sites).any():
            raise ValueError(
                "open_mask selects sites excluded by compile_f(sites=...); it must be a "
                "subset of the sites the Compiled was built over"
            )
    else:
        mask = _validate_mask(open_mask, base.n_supply, "open_mask")

    if demand_id is not None:
        codes, ids = _resolve_demand_codes(base, demand_id)
        frames = [
            _explain_one(prep, base, family, code, demand_id_value, modes, D_max, tau, Q, mask)
            for code, demand_id_value in zip(codes, ids, strict=True)
        ]
        return pd.concat(frames, ignore_index=True)

    supply_codes, supply_ids = _resolve_supply_codes(base, supply_id)
    return _explain_by_supply(
        prep, base, family, supply_codes, supply_ids, modes, D_max, tau, Q, mask
    )
