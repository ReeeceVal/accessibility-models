"""Input validation and the one-off factorisation step.

``prepare()`` is what makes a parameter sweep cheap: the string identifiers are
factorised to ``int32`` codes and the pair table is sorted once, then every model call
reuses the same arrays. Model functions perform **zero** validation — that is what keeps
the hot path clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import _core

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .modes import Mode

__all__ = ["Prepared", "prepare", "validate_inputs"]

SHARE_TOL = 1e-6

_DEMAND_ID = "demand_id"
_SUPPLY_ID = "supply_id"
_DEMAND = "demand"
_CAPACITY = "capacity"
_COST_DEFAULT = "cost_default"
_RANK = "rank"


@dataclass(frozen=True)
class Prepared:
    """Factorised, sorted inputs, reusable across an entire parameter sweep.

    Rows of every ``n_pairs`` array are sorted by ``(demand_code, cost_default,
    supply_code)``, so each demand node owns one contiguous segment delimited by
    ``seg_start`` and that segment is ordered by ascending reachability cost.

    Attributes
    ----------
    demand_df, supply_df : DataFrame
        The originals, held for passthrough of extra columns. Row order defines the
        codes: ``demand_code == i`` refers to ``demand_df.iloc[i]``.
    demand_code, supply_code : ndarray of int32, shape (n_pairs,)
    cost : ndarray of float32, shape (n_pairs,)
        ``cost_default`` — the reachability cost ``d_ij``.
    cost_cols : dict of str to ndarray
        Every numeric column of ``cost_df`` (including ``cost_default``) as float32,
        the candidate per-mode cost columns. ``NaN`` means no route.
    share_cols : dict of str to ndarray
        Every numeric column of ``demand_df`` as float64, the candidate mode shares.
    seg_start : ndarray of int64, shape (n_demand + 1,)
    P : ndarray of float64, shape (n_demand,)
        Demand ``P_i``.
    S : ndarray of float64 or None, shape (n_supply,)
        Capacity ``S_j``, or ``None`` when ``supply_df`` carries no ``capacity``
        column. Required by iFCA and by ``A_i`` in every family.
    """

    demand_df: pd.DataFrame
    supply_df: pd.DataFrame
    demand_code: np.ndarray
    supply_code: np.ndarray
    cost: np.ndarray
    cost_cols: dict[str, np.ndarray]
    share_cols: dict[str, np.ndarray]
    seg_start: np.ndarray
    P: np.ndarray
    S: np.ndarray | None
    has_rank: bool = False
    _validated_modes: bool = field(default=False, repr=False)

    @property
    def n_demand(self) -> int:
        return self.P.size

    @property
    def n_supply(self) -> int:
        return self.supply_df.shape[0]

    @property
    def n_pairs(self) -> int:
        return self.demand_code.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Prepared(n_demand={self.n_demand}, n_supply={self.n_supply}, "
            f"n_pairs={self.n_pairs}, cost_cols={sorted(self.cost_cols)}, "
            f"capacity={'yes' if self.S is not None else 'no'})"
        )


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool
    ]


def validate_inputs(
    demand_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    modes: Sequence[Mode] | None = None,
) -> None:
    """Check the three input frames against the data contract.

    Run inside :func:`prepare` by default and importable standalone. Every failure
    raises ``ValueError`` naming the offending rule.

    Parameters
    ----------
    demand_df, supply_df, cost_df : DataFrame
        See the data contract (``docs/data-contract.md``).
    modes : sequence of Mode, optional
        When given, the mode share and cost columns are checked too: each share column
        present and within ``[0, 1]``, the shares summing to 1.0 across modes to within
        ``1e-6``, and each cost column present on ``cost_df``.

    Raises
    ------
    ValueError
        On any contract violation.
    """
    _require_columns(demand_df, "demand_df", [_DEMAND_ID, _DEMAND])
    _require_columns(supply_df, "supply_df", [_SUPPLY_ID])
    _require_columns(cost_df, "cost_df", [_DEMAND_ID, _SUPPLY_ID, _COST_DEFAULT])

    if demand_df[_DEMAND_ID].duplicated().any():
        raise ValueError("demand_df.demand_id must be unique")
    if supply_df[_SUPPLY_ID].duplicated().any():
        raise ValueError("supply_df.supply_id must be unique")

    demand = demand_df[_DEMAND]
    if demand.isna().any() or not (demand > 0).all():
        raise ValueError("demand_df.demand must be non-null and > 0")

    if _CAPACITY in supply_df.columns:
        capacity = supply_df[_CAPACITY]
        if capacity.isna().any() or not (capacity >= 0).all():
            raise ValueError("supply_df.capacity must be non-null and >= 0")

    cost = cost_df[_COST_DEFAULT]
    if cost.isna().any() or not (cost >= 0).all():
        raise ValueError("cost_df.cost_default must be non-null and >= 0")

    missing_d = ~cost_df[_DEMAND_ID].isin(demand_df[_DEMAND_ID])
    if missing_d.any():
        example = cost_df.loc[missing_d, _DEMAND_ID].iloc[0]
        raise ValueError(
            f"cost_df.demand_id has {int(missing_d.sum())} value(s) absent from demand_df "
            f"(e.g. {example!r})"
        )
    missing_s = ~cost_df[_SUPPLY_ID].isin(supply_df[_SUPPLY_ID])
    if missing_s.any():
        example = cost_df.loc[missing_s, _SUPPLY_ID].iloc[0]
        raise ValueError(
            f"cost_df.supply_id has {int(missing_s.sum())} value(s) absent from supply_df "
            f"(e.g. {example!r})"
        )

    if cost_df.duplicated(subset=[_DEMAND_ID, _SUPPLY_ID]).any():
        raise ValueError("cost_df has duplicate (demand_id, supply_id) pairs")

    if modes:
        _validate_modes(demand_df, cost_df, modes)


def _validate_modes(
    demand_df: pd.DataFrame, cost_df: pd.DataFrame, modes: Sequence[Mode]
) -> None:
    total = np.zeros(demand_df.shape[0], dtype=np.float64)
    for mode in modes:
        if isinstance(mode.share, str):
            if mode.share not in demand_df.columns:
                raise ValueError(f"mode share column {mode.share!r} is not on demand_df")
            share = demand_df[mode.share].to_numpy(dtype=np.float64)
            if np.isnan(share).any() or share.min() < 0.0 or share.max() > 1.0:
                raise ValueError(f"demand_df.{mode.share} must be non-null and within [0, 1]")
        else:
            share = float(mode.share)
            if not 0.0 <= share <= 1.0:
                raise ValueError(f"mode share constant {share} must be within [0, 1]")
        total = total + share
        if mode.cost not in cost_df.columns:
            raise ValueError(f"mode cost column {mode.cost!r} is not on cost_df")

    if np.abs(total - 1.0).max() > SHARE_TOL:
        worst = float(np.abs(total - 1.0).max())
        raise ValueError(
            f"mode shares must sum to 1.0 across modes (worst deviation {worst:.3g} "
            f"> {SHARE_TOL:g})"
        )


def _require_columns(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def prepare(
    demand_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    modes: Sequence[Mode] | None = None,
    validate: bool = True,
) -> Prepared:
    """Validate, factorise and sort the inputs once for reuse across many model calls.

    Parameters
    ----------
    demand_df : DataFrame
        One row per demand node: ``demand_id``, ``demand`` (``P_i``, > 0), any mode
        share columns, plus arbitrary passthrough columns.
    supply_df : DataFrame
        One row per supply point: ``supply_id``, optional ``capacity`` (``S_j``, >= 0),
        plus passthrough columns.
    cost_df : DataFrame
        One row per reachable pair: ``demand_id``, ``supply_id``, ``cost_default``
        (``d_ij``, >= 0, no NaN), optional per-mode cost columns where ``NaN`` means no
        route, and an optional ``rank`` column. Pairs absent from this frame are
        unreachable.
    modes : sequence of Mode, optional
        Validated here if given, so that repeat model calls need no further checking.
    validate : bool, default True
        Set ``False`` to skip :func:`validate_inputs` on trusted repeat calls.

    Returns
    -------
    Prepared

    Notes
    -----
    Identifiers are held as ``str`` where they are ``str`` (zero-padded codes carry leading zeros)
    and exist only at this boundary; everything downstream is ``int32`` codes.
    """
    if validate:
        validate_inputs(demand_df, supply_df, cost_df, modes)

    demand_index = pd.Index(demand_df[_DEMAND_ID])
    supply_index = pd.Index(supply_df[_SUPPLY_ID])
    demand_code = demand_index.get_indexer(cost_df[_DEMAND_ID]).astype(np.int32)
    supply_code = supply_index.get_indexer(cost_df[_SUPPLY_ID]).astype(np.int32)
    if not validate and (demand_code.min(initial=0) < 0 or supply_code.min(initial=0) < 0):
        raise ValueError("cost_df references ids absent from demand_df/supply_df")

    cost = cost_df[_COST_DEFAULT].to_numpy(dtype=np.float32)
    order = np.lexsort((supply_code, cost, demand_code))

    demand_code = demand_code[order]
    supply_code = supply_code[order]
    cost = cost[order]

    cost_cols = {
        name: cost_df[name].to_numpy(dtype=np.float32)[order]
        for name in _numeric_columns(cost_df, {_DEMAND_ID, _SUPPLY_ID, _RANK})
    }
    cost_cols[_COST_DEFAULT] = cost

    share_cols = {
        name: demand_df[name].to_numpy(dtype=np.float64)
        for name in _numeric_columns(demand_df, {_DEMAND_ID})
    }

    S = (
        supply_df[_CAPACITY].to_numpy(dtype=np.float64)
        if _CAPACITY in supply_df.columns
        else None
    )

    return Prepared(
        demand_df=demand_df,
        supply_df=supply_df,
        demand_code=demand_code,
        supply_code=supply_code,
        cost=cost,
        cost_cols=cost_cols,
        share_cols=share_cols,
        seg_start=_core.segment_starts(demand_code, demand_df.shape[0]),
        P=demand_df[_DEMAND].to_numpy(dtype=np.float64),
        S=S,
        has_rank=_RANK in cost_df.columns,
        _validated_modes=bool(modes) and validate,
    )
