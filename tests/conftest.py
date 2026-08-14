"""The toy network every golden-value test is computed against.

6 demand nodes x 4 supply points. Costs and shares are chosen so that every filter
actually bites at ``D_max=45``, ``tau=0.1``, ``Q=2``:

* ``s4`` is beyond ``D_max`` for everybody -> zero exposure, ``R_j`` NaN.
* ``d6`` reaches nobody within ``D_max`` -> ``A_i = 0``, an empty segment.
* ``d5`` is equidistant from ``s1`` and ``s2`` -> exercises the Voronoi tie-break.
* ``d5``-``s3`` falls below ``tau`` under ``gaussian(800)``.
* ``d1``, ``d2`` and ``d5`` keep 3 pairs after ``D_max``, so ``Q=2`` binds.
* ``w_cost`` is NaN everywhere except two pairs, exercising share renormalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

D_MAX = 45.0
TAU = 0.1
Q = 2
BETA = 800.0
KAPPA = 2.0

P = {"d1": 100.0, "d2": 200.0, "d3": 50.0, "d4": 400.0, "d5": 150.0, "d6": 300.0}
PV = {"d1": 1.0, "d2": 0.5, "d3": 0.0, "d4": 0.25, "d5": 0.8, "d6": 0.6}
S = {"s1": 10.0, "s2": 5.0, "s3": 20.0, "s4": 1.0}

# (demand_id, supply_id, cost_default, w_cost)
PAIRS = [
    ("d1", "s1", 5.0, 10.0),
    ("d1", "s2", 20.0, np.nan),
    ("d1", "s3", 40.0, np.nan),
    ("d1", "s4", 60.0, np.nan),
    ("d2", "s1", 15.0, np.nan),
    ("d2", "s2", 10.0, np.nan),
    ("d2", "s3", 35.0, np.nan),
    ("d2", "s4", 55.0, np.nan),
    ("d3", "s2", 8.0, 12.0),
    ("d3", "s3", 25.0, np.nan),
    ("d3", "s4", 50.0, np.nan),
    ("d4", "s1", 30.0, np.nan),
    ("d4", "s3", 12.0, np.nan),
    ("d4", "s4", 48.0, np.nan),
    ("d5", "s1", 22.0, np.nan),
    ("d5", "s2", 22.0, np.nan),
    ("d5", "s3", 44.0, np.nan),
    ("d6", "s4", 70.0, np.nan),
]


def gauss(d: float, beta: float = BETA) -> float:
    """Scalar Gaussian impedance, used to hand-compute expected values."""
    return float(np.exp(-(d**2) / beta))


@pytest.fixture
def demand_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "demand_id": list(P),
            "demand": [P[k] for k in P],
            "pv_share": [PV[k] for k in P],
            "mbt_share": [1.0 - PV[k] for k in P],
            "province": ["WC", "WC", "GP", "GP", "KZN", "KZN"],
        }
    )


@pytest.fixture
def supply_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "supply_id": list(S),
            "capacity": [S[k] for k in S],
            "site_type": ["mall", "rank", "mall", "rural"],
        }
    )


@pytest.fixture
def cost_df() -> pd.DataFrame:
    return pd.DataFrame(PAIRS, columns=["demand_id", "supply_id", "cost_default", "w_cost"])


@pytest.fixture
def frames(demand_df, supply_df, cost_df):
    return demand_df, supply_df, cost_df


@pytest.fixture
def prep(frames):
    from interaction_models import prepare

    return prepare(*frames)


def by_id(df: pd.DataFrame, id_col: str, value_col: str) -> dict:
    """Result column keyed by identifier, for readable assertions."""
    return dict(zip(df[id_col], df[value_col], strict=True))
