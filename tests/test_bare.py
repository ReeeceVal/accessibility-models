"""``bare=True`` against the ``E_j`` column it has to reproduce.

The bare path exists for a loop that calls a model tens of thousands of times and reads
one vector out of each result. It computes ``E_j`` by the same expression as the full
call and then returns, skipping ``R_j``, ``L_j``, the ``O(n_pairs)`` ``n_demand_j``
bincount, both frames, ``params`` and the ``Result``. Same expression, same accumulation
order, so every assertion here is exact equality rather than a tolerance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, S

from interaction_models import (
    Mode,
    Result,
    catchment,
    compile_f,
    gaussian,
    ifca,
    prepare,
    sfca,
    sfca_e,
    voronoi,
)

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]

SITES = list(S)
SUBSETS = [
    subset
    for size in range(1, len(SITES) + 1)
    for subset in itertools.combinations(SITES, size)
]

#: (label, model, kwargs) — every family, since every family produces an E_j.
BARE_CASES = [
    ("catchment", catchment, {"modes": MAC, "D_max": D_MAX, "tau": TAU}),
    ("voronoi", voronoi, {"modes": MAC, "D_max": D_MAX}),
    ("ifca", ifca, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca", sfca, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca_e", sfca_e, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
]


def mask_for(supply_df, keep) -> np.ndarray:
    return supply_df["supply_id"].isin(keep).to_numpy()


@pytest.mark.parametrize(
    ("label", "model", "kwargs"), BARE_CASES, ids=[case[0] for case in BARE_CASES]
)
def test_bare_returns_the_exposure_column_of_the_full_result(
    prep, supply_df, label, model, kwargs
):
    for keep in [None, *SUBSETS]:
        mask = None if keep is None else mask_for(supply_df, keep)
        np.testing.assert_array_equal(
            model(prep, bare=True, open_mask=mask, **kwargs),
            model(prep, open_mask=mask, **kwargs).supply["E_j"].to_numpy(),
            err_msg=f"{label} keep={keep}",
        )


@pytest.mark.parametrize(
    ("label", "model", "kwargs"), BARE_CASES, ids=[case[0] for case in BARE_CASES]
)
def test_bare_is_a_float64_vector_over_the_supply_rows(prep, label, model, kwargs):
    out = model(prep, bare=True, **kwargs)
    assert not isinstance(out, Result)
    assert out.dtype == np.float64
    assert out.shape == (len(SITES),)


def test_bare_from_a_compiled_matches_the_result(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    np.testing.assert_array_equal(
        sfca_e(comp, Q=2, bare=True),
        sfca_e(comp, Q=2).supply["E_j"].to_numpy(),
    )


def test_bare_from_a_width_compile_matches_the_unbounded_result(prep, supply_df):
    """The two features composed: a bounded read returning a bare vector."""
    bounded = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    baseline = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    for keep in [None, *SUBSETS]:
        mask = None if keep is None else mask_for(supply_df, keep)
        np.testing.assert_array_equal(
            sfca_e(bounded, Q=2, open_mask=mask, bare=True),
            sfca_e(baseline, Q=2, open_mask=mask).supply["E_j"].to_numpy(),
            err_msg=f"keep={keep}",
        )


@pytest.mark.parametrize(
    "conflict",
    [{"stats": ["coverage"]}, {"output": ("supply",)}, {"output": ()}],
)
@pytest.mark.parametrize(
    ("label", "model", "kwargs"), BARE_CASES, ids=[case[0] for case in BARE_CASES]
)
def test_bare_rejects_stats_and_output(prep, label, model, kwargs, conflict):
    """Silently ignoring a request the caller made is the failure mode to avoid."""
    with pytest.raises(ValueError, match="bare=True returns E_j only"):
        model(prep, bare=True, **kwargs, **conflict)


def test_an_explicit_default_output_is_not_a_conflict(prep):
    np.testing.assert_array_equal(
        sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, bare=True,
               output=("demand", "supply")),
        sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, bare=True),
    )


def test_bare_needs_no_capacity_column(frames):
    """E_j is capacity-free in four families, so bare works where A_i would raise."""
    demand_df, supply_df, cost_df = frames
    prep = prepare(demand_df, supply_df.drop(columns="capacity"), cost_df)
    for model, kwargs in (
        (catchment, {"modes": MAC, "D_max": D_MAX, "tau": TAU}),
        (voronoi, {"modes": MAC, "D_max": D_MAX}),
        (sfca, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
        (sfca_e, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ):
        assert np.isfinite(model(prep, bare=True, **kwargs)).all()
        with pytest.raises(ValueError, match="capacity"):
            model(prep, **kwargs)
    # iFCA is the exception: S_j sits inside E_j, so bare does not relax the requirement.
    with pytest.raises(ValueError, match="S_j is in the denominator"):
        ifca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, bare=True)
