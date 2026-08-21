"""Spatial interaction and accessibility estimators over a demand-to-supply cost matrix.

Five model families, computed from three pandas DataFrames::

    import interaction_models as im

    prep = im.prepare(demand_df, supply_df, cost_df)
    modes = [
        im.Mode(share="pv_share",  cost="cost_default", kappa=1.0, decay=im.gaussian(800)),
        im.Mode(share="mbt_share", cost="cost_default", kappa=2.0, decay=im.gaussian(800)),
    ]
    res = im.sfca(prep, modes=modes, D_max=45, Q=5, tau=0.01)

Each family is a single model whose optional terms -- impedance, mode split, a bounded
choice set -- switch on with the parameters that define them. The resolved configuration
comes back as ``res.params``. See ``docs/`` for the full reference.
"""

from . import stats
from .compiled import Compiled, compile_f
from .explain import explain
from .models import Result, catchment, ifca, sfca, sfca_e, voronoi
from .modes import Mode, gaussian
from .prepare import Prepared, prepare, validate_inputs
from .sweep import sweep

__all__ = [
    "Compiled",
    "Mode",
    "Prepared",
    "Result",
    "catchment",
    "compile_f",
    "explain",
    "gaussian",
    "ifca",
    "prepare",
    "sfca",
    "sfca_e",
    "stats",
    "sweep",
    "validate_inputs",
    "voronoi",
]
