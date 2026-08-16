"""``open_mask`` against the reference it has to match.

The reference is a full re-``prepare()`` on a ``cost_df`` restricted to the open sites —
what an optimisation loop would otherwise have to do per candidate network, and what
``tests/test_sfca_e.py`` already uses to probe monotonicity. Every assertion here is
exact equality, not a tolerance: the mask drops rows before impedance, so the surviving
pairs and their ``f_multi`` are bit-for-bit the ones the reference computes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, S

from interaction_models import Mode, catchment, gaussian, ifca, prepare, sfca, sfca_e, voronoi

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]
#: A mode on its own cost column: f_multi is no longer monotone in cost_default, so the
#: rank runs through the lexsort path rather than the prefix-truncation fast path.
OWN_COL = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", cost="w_cost", kappa=1.0, decay=gaussian(BETA)),
]

SITES = list(S)
#: Every non-empty subset of the four supply points.
SUBSETS = [
    subset
    for size in range(1, len(SITES) + 1)
    for subset in itertools.combinations(SITES, size)
]


def mask_for(supply_df, keep) -> np.ndarray:
    return supply_df["supply_id"].isin(keep).to_numpy()


def restricted(frames, keep):
    """The reference: the network with closed sites removed from the inputs outright."""
    demand_df, supply_df, cost_df = frames
    return prepare(
        demand_df,
        supply_df[supply_df["supply_id"].isin(keep)].reset_index(drop=True),
        cost_df[cost_df["supply_id"].isin(keep)].reset_index(drop=True),
    )


CASES = [
    ("catchment", catchment, {"modes": MAC, "D_max": D_MAX, "tau": TAU}),
    ("voronoi", voronoi, {"modes": MAC, "D_max": D_MAX}),
    ("ifca", ifca, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca", sfca, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca_e-single", sfca_e, {"modes": SINGLE, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca_e-mac", sfca_e, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca_e-owncol", sfca_e, {"modes": OWN_COL, "D_max": D_MAX, "tau": TAU, "Q": 2}),
    ("sfca_e-Q1", sfca_e, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 1}),
    ("sfca_e-Q4", sfca_e, {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 4}),
]


@pytest.mark.parametrize(("label", "model", "kwargs"), CASES, ids=[c[0] for c in CASES])
def test_open_mask_matches_a_restricted_reprepare(frames, prep, label, model, kwargs):
    """The whole contract: masking == removing the sites from cost_df, exactly."""
    supply_df = frames[1]
    for keep in SUBSETS:
        masked = model(prep, open_mask=mask_for(supply_df, keep), **kwargs)
        reference = model(restricted(frames, keep), **kwargs)

        # Supply frames differ in length by construction — the mask keeps closed sites
        # as zero rows, which is what makes the output alignable across candidate
        # networks. Compare on the open sites, then assert the closed ones are zero.
        got = dict(zip(masked.supply["supply_id"], masked.supply["E_j"], strict=True))
        want = dict(zip(reference.supply["supply_id"], reference.supply["E_j"], strict=True))
        for site in keep:
            assert got[site] == want[site], (label, keep, site)
        for site in set(SITES) - set(keep):
            assert got[site] == 0.0, (label, keep, site)

        np.testing.assert_array_equal(
            masked.demand["A_i"].to_numpy(), reference.demand["A_i"].to_numpy(),
            err_msg=f"{label} {keep}",
        )
        np.testing.assert_array_equal(
            masked.demand["n_supply_i"].to_numpy(),
            reference.demand["n_supply_i"].to_numpy(),
            err_msg=f"{label} {keep}",
        )


def test_q_takes_the_top_q_open_not_the_open_of_the_top_q(frames, prep):
    """The distinction the mask exists to get right.

    ``d1`` reaches s1 (5), s2 (20), s3 (40) within D_max, ranked in that order. Close
    s1 and s2: with Q=2 the choice set must fall back to s3, not come up empty.
    """
    supply_df = frames[1]
    res = sfca_e(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=2,
                 open_mask=mask_for(supply_df, ["s3", "s4"]))
    d1 = dict(zip(res.demand["demand_id"], res.demand["n_supply_i"], strict=True))
    assert d1["d1"] == 1
    exposure = dict(zip(res.supply["supply_id"], res.supply["E_j"], strict=True))
    assert exposure["s3"] > 0.0


def test_mask_is_reported_in_params(frames, prep):
    supply_df = frames[1]
    kwargs = {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}
    assert sfca_e(prep, **kwargs).params["n_open"] is None
    assert sfca_e(prep, open_mask=mask_for(supply_df, ["s1", "s3"]), **kwargs).params[
        "n_open"
    ] == 2


def test_all_closed_gives_an_empty_network(frames, prep):
    supply_df = frames[1]
    res = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2,
                 open_mask=mask_for(supply_df, []))
    assert res.supply["E_j"].to_numpy().tolist() == [0.0] * len(SITES)
    assert res.demand["Phi_i"].to_numpy().tolist() == [0.0] * res.demand.shape[0]
    assert res.demand["n_supply_i"].to_numpy().tolist() == [0] * res.demand.shape[0]


def test_opening_a_site_never_lowers_total_exposure_under_the_mask(frames, prep):
    """Monotonicity, the property the family exists for, driven through the mask."""
    supply_df = frames[1]
    kwargs = {"modes": MAC, "D_max": D_MAX, "tau": TAU, "Q": 2}

    def total(keep):
        res = sfca_e(prep, open_mask=mask_for(supply_df, keep), output=("supply",), **kwargs)
        return float(res.supply["E_j"].sum())

    for size in range(len(SITES)):
        for subset in itertools.combinations(SITES, size):
            for added in SITES:
                if added not in subset:
                    assert total((*subset, added)) >= total(subset) - 1e-12, (subset, added)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([True, False]),
        np.array([1, 0, 1, 0]),
        np.ones((4, 1), dtype=bool),
    ],
    ids=["wrong-length", "not-boolean", "wrong-shape"],
)
def test_a_malformed_mask_is_rejected(prep, bad):
    with pytest.raises(ValueError, match="open_mask must be a boolean array"):
        sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, open_mask=bad)
