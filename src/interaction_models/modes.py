"""Mode specification, decay callables and the ``f_multi`` assembly.

The combined impedance is a share-weighted sum over any number of modes::

    f_multi(i, j) = sum_m pi_m(i) * f_m(kappa_m * cost_m(i, j))

Mode shares live on ``demand_df``, per demand node, never on ``cost_df``: a per-pair
share column would cost hundreds of megabytes per mode at scale, and ``pi_m(i, j)`` is
not something the data can support anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .prepare import Prepared

__all__ = ["Mode", "gaussian", "impedance", "is_kappa_only", "reach_weight"]


@dataclass(frozen=True)
class Mode:
    """One travel mode's contribution to the combined impedance.

    Parameters
    ----------
    share : str or float, default 1.0
        Column on ``demand_df`` holding ``pi_m(i)``, or a constant applying to every
        node. Shares must sum to 1.0 across the modes of a call.
    cost : str, default "cost_default"
        Column on ``cost_df`` this mode travels on. ``NaN`` means no route exists for
        this mode on that pair, which makes the mode *unavailable* there.
    kappa : float, default 1.0
        Multiplier on that cost. A two-mode split typically leaves the reference mode at
        ``kappa=1`` and penalises the slower one with ``kappa>1``, both travelling on
        ``cost_default``.
    decay : callable or None, default None
        Vectorised ``ndarray -> ndarray`` impedance, applied to ``kappa * cost``.
        ``None`` means no impedance for this mode, i.e. ``f_m = 1``.

        Decay callables are assumed **non-increasing**. That assumption drives the
        prefix-truncation fast path described in ``docs/pipeline.md``; a non-monotone
        callable still gives correct results, but only if some mode carries its own
        cost column.

    Examples
    --------
    A two-mode split, ``f_multi = s*f(d) + (1-s)*f(kappa*d)``:

    >>> modes = [
    ...     Mode(share="pv_share", cost="cost_default", kappa=1.0, decay=gaussian(800)),
    ...     Mode(share="mbt_share", cost="cost_default", kappa=2.0, decay=gaussian(800)),
    ... ]

    A single-element list is the unimodal case: one share of 1.0, one decay, no split.
    """

    share: str | float = 1.0
    cost: str = "cost_default"
    kappa: float = 1.0
    decay: Callable[[np.ndarray], np.ndarray] | None = None


def gaussian(beta: float) -> Callable[[np.ndarray], np.ndarray]:
    """Gaussian distance decay ``f(d) = exp(-d**2 / beta)``.

    The only decay shipped with the package; any vectorised callable works in its place.

    Parameters
    ----------
    beta : float
        Bandwidth, in squared cost units. Larger is flatter.

    Returns
    -------
    callable
        ``ndarray -> ndarray``, carrying ``.beta`` and a readable ``__name__``.

    Examples
    --------
    >>> f = gaussian(800)
    >>> float(f(np.array([0.0])))
    1.0
    >>> f.__name__
    'gaussian(800)'
    """

    def decay(d: np.ndarray) -> np.ndarray:
        return np.exp(-(d**2) / beta)

    decay.__name__ = f"gaussian({beta:g})"
    decay.beta = beta
    return decay


def is_kappa_only(modes: Sequence[Mode]) -> bool:
    """True when every mode travels on ``cost_default``.

    In that case ``f_multi`` is a monotone function of ``cost_default``, so the
    cost-ascending segment order is already the ``f_multi``-descending rank order and
    every filter becomes a prefix truncation.
    """
    return all(mode.cost == "cost_default" for mode in modes)


def _share_values(prep: Prepared, mode: Mode, demand_code: np.ndarray) -> np.ndarray:
    if isinstance(mode.share, str):
        return prep.share_cols[mode.share][demand_code]
    return np.float64(mode.share)


def impedance(prep: Prepared, modes: Sequence[Mode], rows: np.ndarray) -> np.ndarray:
    """Combined impedance ``f_multi`` for the given pair rows (pipeline stages 2-4).

    Per mode, ``f_m = decay_m(kappa_m * cost_m)``; a ``NaN`` cost makes the mode
    unavailable on that pair and its share is redistributed **proportionally over the
    modes that remain**. Where every mode is unavailable the result is ``0.0``, which
    the ``tau`` filter then drops.

    Parameters
    ----------
    prep : Prepared
    modes : sequence of Mode
    rows : ndarray of int
        Indices into the prepared pair arrays.

    Returns
    -------
    ndarray of float64, shape ``rows.shape``

    Notes
    -----
    Renormalisation applies to **unavailability only**. A mode whose route exists but is
    too costly is a modelled penalty, not missing data, and is never renormalised — see
    ``docs/modes.md``.
    """
    demand_code = prep.demand_code[rows]
    weighted = np.zeros(rows.size, dtype=np.float64)
    available_share = np.zeros(rows.size, dtype=np.float64)

    for mode in modes:
        cost = prep.cost_cols[mode.cost][rows]
        available = ~np.isnan(cost)
        share = _share_values(prep, mode, demand_code)
        if mode.decay is None:
            f = np.ones(rows.size, dtype=np.float64)
        else:
            f = np.asarray(mode.decay(mode.kappa * cost), dtype=np.float64)
        weighted += np.where(available, share * f, 0.0)
        available_share += np.where(available, share, 0.0)

    return np.divide(
        weighted,
        available_share,
        out=np.zeros(rows.size, dtype=np.float64),
        where=available_share > 0.0,
    )


def reach_weight(
    prep: Prepared, modes: Sequence[Mode], rows: np.ndarray, D_max: float
) -> np.ndarray:
    """Mode-availability weight ``sum_m pi_m * 1[kappa_m * d <= D_max]``.

    The Voronoi family has no impedance, so mode split enters only as a reach gate. There
    is deliberately **no** renormalisation here: a share whose mode cannot reach the
    assigned site is simply lost, and that loss is the modelled penalty. Renormalising
    would erase it, leaving the gate with no effect at all.

    A ``NaN`` mode cost fails the gate, exactly as an over-budget one does.

    Returns
    -------
    ndarray of float64, shape ``rows.shape``
        Within ``[0, 1]``.
    """
    demand_code = prep.demand_code[rows]
    weight = np.zeros(rows.size, dtype=np.float64)
    for mode in modes:
        cost = prep.cost_cols[mode.cost][rows]
        with np.errstate(invalid="ignore"):
            reaches = mode.kappa * cost <= D_max
        weight += np.where(reaches, _share_values(prep, mode, demand_code), 0.0)
    return weight
