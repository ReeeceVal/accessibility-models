"""Golden E_j, A_i and r_i for the iFCA family.

Expected values come from a direct transcription of the family's three equations over
plain dicts, driven by hand-derived survivor sets. Nothing here touches the segmented
core.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, PV, TAU, P, Q, S, by_id, gauss

from interaction_models import Mode, gaussian, ifca

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]

# Pairs surviving D_max = 45 and tau = 0.1 under the single-mode Gaussian.
# d5-s3 goes: f(44) = 0.0889. d6 reaches nobody at all.
SURVIVORS_SINGLE = {
    "d1": {"s1": 5.0, "s2": 20.0, "s3": 40.0},
    "d2": {"s1": 15.0, "s2": 10.0, "s3": 35.0},
    "d3": {"s2": 8.0, "s3": 25.0},
    "d4": {"s1": 30.0, "s3": 12.0},
    "d5": {"s1": 22.0, "s2": 22.0},
    "d6": {},
}

# Under the two-mode split the floor bites harder: d3-s3 (0.0439), d4-s1 (0.0895)
# and d5-s3 (0.0712) all fall below tau.
SURVIVORS_MAC = {
    "d1": {"s1": 5.0, "s2": 20.0, "s3": 40.0},
    "d2": {"s1": 15.0, "s2": 10.0, "s3": 35.0},
    "d3": {"s2": 8.0},
    "d4": {"s3": 12.0},
    "d5": {"s1": 22.0, "s2": 22.0},
    "d6": {},
}


def top_q(survivors, impedance, q=Q):
    """Keep the q best pairs per node by impedance, ties on ascending cost then site."""
    out = {}
    for i, row in survivors.items():
        ordered = sorted(row.items(), key=lambda kv: (-impedance(i, kv[1]), kv[1], kv[0]))
        out[i] = dict(ordered[:q])
    return out


def single(i, d):
    return gauss(d)


def mac(i, d):
    return PV[i] * gauss(d) + (1.0 - PV[i]) * gauss(KAPPA * d)


def ifca_reference(survivors, impedance):
    """r_i, C_j and E_j, transcribed literally from the family definition."""
    f = {i: {j: impedance(i, d) for j, d in row.items()} for i, row in survivors.items()}
    denom = {i: sum(S[j] * fij for j, fij in row.items()) for i, row in f.items()}
    r = {i: (P[i] / denom[i] if denom[i] > 0 else 0.0) for i in f}
    C = {j: sum(r[i] * f[i].get(j, 0.0) for i in f) for j in S}
    E = {j: S[j] * C[j] for j in S}
    A = {i: (denom[i] / P[i]) for i in f}
    return E, A


#: Every combination of the family's two optional terms: mode split and a bounded Q.
CASES = {
    "one-mode": (SURVIVORS_SINGLE, single, SINGLE, None),
    "one-mode+Q": (top_q(SURVIVORS_SINGLE, single), single, SINGLE, Q),
    "two-mode": (SURVIVORS_MAC, mac, MAC, None),
    "two-mode+Q": (top_q(SURVIVORS_MAC, mac), mac, MAC, Q),
}


@pytest.mark.parametrize("case", list(CASES))
def test_golden_exposure_and_accessibility(prep, case):
    survivors, impedance, modes, q = CASES[case]
    res = ifca(prep, modes=modes, D_max=D_MAX, tau=TAU, Q=q)
    assert res.params["n_modes"] == len(modes)
    assert res.params["Q"] == q

    expected_E, expected_A = ifca_reference(survivors, impedance)
    assert by_id(res.supply, "supply_id", "E_j") == pytest.approx(expected_E)
    assert by_id(res.demand, "demand_id", "A_i") == pytest.approx(expected_A)


@pytest.mark.parametrize("case", list(CASES))
def test_choice_set_sizes_match_the_hand_derived_survivors(prep, case):
    survivors, _, modes, q = CASES[case]
    res = ifca(prep, modes=modes, D_max=D_MAX, tau=TAU, Q=q)
    expected = {i: len(row) for i, row in survivors.items()}
    assert by_id(res.demand, "demand_id", "n_supply_i") == expected


def test_q_binds_only_where_more_than_q_pairs_survive(prep):
    assert by_id(
        ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q).demand,
        "demand_id",
        "n_supply_i",
    ) == {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2, "d6": 0}


def test_unreached_node_gets_infinite_r_and_zero_access(prep):
    res = ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    demand = res.demand.set_index("demand_id")
    assert np.isinf(demand.loc["d6", "r_i"])
    assert demand.loc["d6", "A_i"] == 0.0


def test_r_i_is_the_reciprocal_of_a_i(prep):
    res = ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    reached = res.demand["A_i"] > 0
    np.testing.assert_allclose(
        res.demand.loc[reached, "r_i"], 1.0 / res.demand.loc[reached, "A_i"]
    )


def test_exposure_of_an_unreached_site_is_zero(prep):
    res = ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    supply = res.supply.set_index("supply_id")
    assert supply.loc["s4", "E_j"] == 0.0
    assert np.isnan(supply.loc["s4", "R_j"])


def test_infinite_q_equals_no_q(prep):
    unrestricted = ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    infinite = ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=np.inf)
    assert infinite.params["Q"] is None
    np.testing.assert_array_equal(infinite.supply["E_j"], unrestricted.supply["E_j"])
    np.testing.assert_array_equal(infinite.demand["A_i"], unrestricted.demand["A_i"])


def test_modes_are_required(prep):
    with pytest.raises(ValueError, match="requires modes"):
        ifca(prep, modes=[], D_max=D_MAX)


def test_capacity_is_required(frames):
    demand_df, supply_df, cost_df = frames
    with pytest.raises(ValueError, match="S_j is in the denominator"):
        ifca(
            demand_df,
            supply_df.drop(columns="capacity"),
            cost_df,
            modes=SINGLE,
            D_max=D_MAX,
        )
