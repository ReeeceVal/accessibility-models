"""A minimal grid runner over one :class:`~interaction_models.prepare.Prepared`.

Cartesian product, one row per combination, labels carried through. Deliberately not a
harness: no dedup, no persistence, no resume, no evaluation against realised counts.
``prepare()`` is what makes an external sweep fast, so nothing is lost by leaving those
to the caller.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from .models import catchment, ifca, sfca, sfca_e, voronoi

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .modes import Mode
    from .prepare import Prepared

__all__ = ["MODELS", "sweep"]

MODELS = {
    "catchment": catchment,
    "voronoi": voronoi,
    "ifca": ifca,
    "sfca": sfca,
    "sfca_e": sfca_e,
}

#: Grid keys that go straight to the model function.
_PARAM_KEYS = ("D_max", "tau", "Q")
#: Grid keys that rewrite the mode template instead.
_MODE_KEYS = ("modes", "decay", "kappa")


def _labelled(value: object) -> tuple[object, object]:
    """Split a ``(label, value)`` tuple, or use the value as its own label."""
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return value, value


def _apply(key: str, value: object, modes: list[Mode] | None) -> list[Mode]:
    if modes is None and key in ("decay", "kappa"):
        raise ValueError(f"grid key {key!r} rewrites a mode template; pass modes=[...]")
    if key == "modes":
        return list(value)
    if key == "decay":
        return [replace(mode, decay=value) for mode in modes]
    # kappa applies to every mode but the first, which holds the reference kappa = 1.
    return [mode if i == 0 else replace(mode, kappa=value) for i, mode in enumerate(modes)]


def sweep(
    prep: Prepared,
    model: str | Callable,
    grid: dict[str, Sequence],
    *,
    modes: Sequence[Mode] | None = None,
    stats: Sequence[str] | None = None,
    **fixed,
) -> pd.DataFrame:
    """Run one model across the Cartesian product of a parameter grid.

    Parameters
    ----------
    prep : Prepared
        Built once, reused for every combination — this is the whole point.
    model : str or callable
        ``"catchment"``, ``"voronoi"``, ``"ifca"``, ``"sfca"``, ``"sfca_e"``, or a model
        function.
    grid : dict of str to sequence
        Keys may be ``"D_max"``, ``"tau"``, ``"Q"`` (passed to the model),
        ``"modes"`` (a whole mode list), ``"decay"`` (applied to **every** mode of the
        template) or ``"kappa"`` (applied to ``modes[1:]``, leaving mode 0 at the
        reference ``κ = 1``). Any other key raises.

        Each value is either a plain value or a ``(label, value)`` tuple; the label is
        what lands in the result column, which is how a decay callable becomes readable.
    modes : sequence of Mode, optional
        The mode template that ``"decay"`` and ``"kappa"`` rewrite. Required if either
        of those keys is in the grid.
    stats : sequence of str, optional
        Stat groups to compute per combination; their keys become result columns.
    **fixed
        Constant model arguments, e.g. ``D_max=45``.

    Returns
    -------
    DataFrame
        One row per combination: ``model``, one column per grid key, then the resolved
        ``D_max`` / ``Q`` / ``tau`` / ``n_modes``, then the stat keys.

    Notes
    -----
    Output frames are never assembled (``output=()``): a sweep row is a scalar summary,
    so per-node and per-site frames would be built and thrown away. Call the model
    directly for those.

    Examples
    --------
    >>> rows = im.sweep(prep, "sfca", modes=modes, D_max=45, tau=0.01, grid={
    ...     "decay": [("gauss800", gaussian(800)), ("gauss1200", gaussian(1200))],
    ...     "kappa": [1.5, 2.0],
    ...     "Q":     [5, 10, 20],
    ... }, stats=["coverage", "choice_set"])
    """
    fn = model if callable(model) else MODELS[model]
    name = model if isinstance(model, str) else fn.__name__
    unknown = [k for k in grid if k not in _PARAM_KEYS + _MODE_KEYS]
    if unknown:
        raise ValueError(
            f"unknown grid key(s) {unknown}; available: {list(_PARAM_KEYS + _MODE_KEYS)}"
        )

    keys = list(grid)
    rows = []
    for combination in itertools.product(*(grid[key] for key in keys)):
        labels: dict[str, object] = {}
        call_modes = list(modes) if modes is not None else None
        kwargs = dict(fixed)

        for key, entry in zip(keys, combination, strict=True):
            label, value = _labelled(entry)
            labels[key] = label
            if key in _PARAM_KEYS:
                kwargs[key] = value
            else:
                call_modes = _apply(key, value, call_modes)

        result = fn(prep, modes=call_modes, stats=stats, output=(), **kwargs)
        row: dict[str, object] = {"model": name, **labels}
        row.update({k: v for k, v in result.params.items() if k not in row})
        if result.stats:
            row.update(result.stats)
        rows.append(row)

    return pd.DataFrame(rows)
