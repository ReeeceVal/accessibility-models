"""Pipeline stages 0-5, hoisted out of the model call.

``prepare()`` factorises and sorts what is independent of *every* parameter.
:func:`compile_f` goes one step further and resolves what is independent of the **open
set**: ``f_multi`` depends only on the pair and the impedance parameters, never on which
sites are open, so a site-selection loop can compute it once and reuse it for every
candidate network.

Two things are hoisted, and the second matters more than the first:

* the ``decay`` evaluations, one per mode per surviving pair;
* the **rank order**. Rows are stored ``f_multi``-descending within each demand node's
  segment. Masking preserves relative order, so ``C_Q(i)`` is a prefix truncation for
  every candidate network — even when a mode carries its own cost column, where
  :func:`~interaction_models.models.sfca_e` would otherwise pay a ``lexsort`` per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _core
from .modes import impedance, is_kappa_only

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .modes import Mode
    from .prepare import Prepared

__all__ = ["Compiled", "compile_f"]


@dataclass(frozen=True)
class Compiled:
    """``f_multi`` for the pairs surviving ``sites``, ``D_max`` and ``tau``.

    Rows are ordered by ``(demand_code, f descending)``, ties keeping the incoming
    ``(cost_default, supply_code)`` order — the same tie-break
    :func:`~interaction_models._core.segment_rank_desc` applies, so a compiled call
    selects the same choice set as an uncompiled one.

    Attributes
    ----------
    prep : Prepared
        Held by reference, for ``P``, ``S`` and output assembly.
    f : ndarray of float64, shape (n_pairs,)
        ``f_multi`` for each surviving pair.
    demand_code, supply_code : ndarray of int32, shape (n_pairs,)
    seg : ndarray of int64, shape (n_demand + 1,)
        Segment offsets over ``demand_code``, before any open-set mask.
    D_max, tau : float
    n_modes : int
    sites : ndarray of bool or None, shape (n_supply,)
        The permanent site restriction this was compiled under, if any. An ``open_mask``
        passed to a model must be a subset of it.
    """

    prep: Prepared
    f: np.ndarray
    demand_code: np.ndarray
    supply_code: np.ndarray
    seg: np.ndarray
    D_max: float
    tau: float
    n_modes: int
    sites: np.ndarray | None = None

    @property
    def n_pairs(self) -> int:
        return self.f.size

    @property
    def n_sites(self) -> int:
        """How many supply points this was compiled over."""
        return self.prep.n_supply if self.sites is None else int(np.count_nonzero(self.sites))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Compiled(n_pairs={self.n_pairs}, n_demand={self.prep.n_demand}, "
            f"n_sites={self.n_sites}, D_max={self.D_max}, tau={self.tau}, "
            f"n_modes={self.n_modes})"
        )


def _validate_mask(mask: np.ndarray | None, n_supply: int, name: str) -> np.ndarray | None:
    """Check a site mask against the supply table, or pass ``None`` through."""
    if mask is None:
        return None
    out = np.asarray(mask)
    if out.dtype != np.bool_ or out.shape != (n_supply,):
        raise ValueError(
            f"{name} must be a boolean array of shape ({n_supply},), got "
            f"dtype {out.dtype} shape {out.shape}"
        )
    return out


def _open_rows(prep: Prepared, rows: np.ndarray, open_mask: np.ndarray | None) -> np.ndarray:
    """Drop pairs whose supply point is closed (pipeline stage 0).

    Applied before impedance, so a closed site is indistinguishable from one absent
    from ``cost_df`` altogether: every later stage — ``tau``, the rank, ``C_Q(i)`` —
    sees only the open sites, and ``Q`` therefore means the top ``Q`` *open* options.
    """
    mask = _validate_mask(open_mask, prep.n_supply, "open_mask")
    if mask is None:
        return rows
    return rows[mask[prep.supply_code[rows]]]


def compile_f(
    prep: Prepared,
    *,
    modes: Sequence[Mode],
    D_max: float = np.inf,
    tau: float = 0.0,
    sites: np.ndarray | None = None,
) -> Compiled:
    """Resolve ``f_multi`` and the rank order once, for reuse across many open sets.

    Everything this computes is independent of which sites are open, so an optimisation
    loop that varies the network while holding the impedance parameters fixed should
    call this once and pass the result to the model in place of ``prep``.

    Parameters
    ----------
    prep : Prepared
    modes : sequence of Mode
        Required — there is no ``f_multi`` without them.
    D_max : float, default inf
        Maximum ``cost_default``. Baked in; the model call must not repeat it.
    tau : float, default 0.0
        Impedance floor on ``f_multi``. Baked in, as ``D_max`` is.
    sites : ndarray of bool, optional
        A **permanent** restriction to a subset of ``supply_df``, applied before the
        decay evaluations so the excluded pairs cost nothing to compile. Use this when
        the network is fixed and the parameters vary; use the model's ``open_mask``
        when the network is what varies. An ``open_mask`` must be a subset of it.

    Returns
    -------
    Compiled

    Raises
    ------
    ValueError
        If ``modes`` is empty, or ``sites`` is not a boolean array of one flag per row
        of ``supply_df``.

    Notes
    -----
    Accepted by :func:`~interaction_models.models.catchment`,
    :func:`~interaction_models.models.ifca`, :func:`~interaction_models.models.sfca` and
    :func:`~interaction_models.models.sfca_e`. **Not** by
    :func:`~interaction_models.models.voronoi`, which has no impedance and assigns on
    ``cost_default`` order rather than ``f_multi`` order.

    Results are identical to the uncompiled call, and bit-for-bit so on the kappa-only
    fast path. Where a mode carries its own cost column the compiled rows are physically
    reordered, so float accumulation runs in a different order and sums can differ in the
    last ulp.

    Examples
    --------
    >>> comp = im.compile_f(prep, modes=modes, D_max=45, tau=0.01)   # doctest: +SKIP
    >>> for mask in candidate_networks:                              # doctest: +SKIP
    ...     res = im.sfca_e(comp, Q=5, open_mask=mask)
    """
    if not modes:
        raise ValueError("compile_f requires modes=[...]; f_multi is undefined without them")

    site_mask = _validate_mask(sites, prep.n_supply, "sites")
    rows = np.flatnonzero(prep.cost <= D_max)
    if site_mask is not None:
        rows = rows[site_mask[prep.supply_code[rows]]]

    f = impedance(prep, modes, rows)
    keep = f >= tau
    rows, f = rows[keep], f[keep]
    demand_code = prep.demand_code[rows]

    # On the kappa-only fast path the incoming order is cost ascending within each
    # segment, which for a non-increasing decay already *is* f descending. Otherwise pay
    # the lexsort once here, so that no model call ever has to.
    if not is_kappa_only(modes):
        order = np.lexsort((-f, demand_code))
        rows, f, demand_code = rows[order], f[order], demand_code[order]

    return Compiled(
        prep=prep,
        f=f,
        demand_code=demand_code,
        supply_code=prep.supply_code[rows],
        seg=_core.segment_starts(demand_code, prep.n_demand),
        D_max=D_max,
        tau=tau,
        n_modes=len(modes),
        sites=site_mask,
    )
