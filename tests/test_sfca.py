"""Golden E_j and A_i for the 3SFCA family."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import D_MAX, TAU, P, Q, S, by_id
from test_ifca import MAC, SINGLE, SURVIVORS_MAC, SURVIVORS_SINGLE, mac, single, top_q

from interaction_models import sfca
from interaction_models._core import segment_lengths, segment_sum
from interaction_models.models import _select, _selection_probability


def sfca_reference(survivors, impedance):
    """G_ij, E_j and the A_i dual, transcribed literally."""
    f = {i: {j: impedance(i, d) for j, d in row.items()} for i, row in survivors.items()}
    total = {i: sum(row.values()) for i, row in f.items()}
    G = {i: {j: fij / total[i] for j, fij in row.items()} for i, row in f.items() if total[i] > 0}
    E = {j: sum(P[i] * G[i].get(j, 0.0) * f[i].get(j, 0.0) for i in G) for j in S}
    R = {j: (S[j] / E[j] if E[j] > 0 else 0.0) for j in S}
    A = {
        i: (sum(R[j] * G[i][j] * f[i][j] for j in G[i]) if i in G else 0.0)
        for i in survivors
    }
    return E, A


CASES = {
    "one-mode": (top_q(SURVIVORS_SINGLE, single), single, SINGLE),
    "two-mode": (top_q(SURVIVORS_MAC, mac), mac, MAC),
}


@pytest.mark.parametrize("case", list(CASES))
def test_golden_exposure_and_accessibility(prep, case):
    survivors, impedance, modes = CASES[case]
    res = sfca(prep, modes=modes, D_max=D_MAX, tau=TAU, Q=Q)
    assert res.params["n_modes"] == len(modes)

    expected_E, expected_A = sfca_reference(survivors, impedance)
    assert by_id(res.supply, "supply_id", "E_j") == pytest.approx(expected_E)
    assert by_id(res.demand, "demand_id", "A_i") == pytest.approx(expected_A)


@pytest.mark.parametrize("modes", [SINGLE, MAC])
def test_selection_probabilities_sum_to_one(prep, modes):
    sel = _select(prep, modes, D_MAX, TAU, Q, True)
    totals = segment_sum(_selection_probability(sel), sel.seg)
    non_empty = segment_lengths(sel.seg) > 0
    np.testing.assert_allclose(totals[non_empty], 1.0)
    np.testing.assert_array_equal(totals[~non_empty], 0.0)


def test_exposure_ignores_capacity(frames):
    """E_j is already an expected demand count, so doubling capacity must not move it.

    A_i does double, because it is read off R_j = S_j / E_j.
    """
    demand_df, supply_df, cost_df = frames
    base = sfca(demand_df, supply_df, cost_df, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q)
    doubled = sfca(
        demand_df,
        supply_df.assign(capacity=supply_df["capacity"] * 2),
        cost_df,
        modes=SINGLE,
        D_max=D_MAX,
        tau=TAU,
        Q=Q,
    )
    np.testing.assert_allclose(base.supply["E_j"], doubled.supply["E_j"])
    np.testing.assert_allclose(base.demand["A_i"] * 2, doubled.demand["A_i"])


def test_q_is_required(prep):
    with pytest.raises(ValueError, match="requires a finite Q"):
        sfca(prep, modes=SINGLE, D_max=D_MAX, Q=None)
    with pytest.raises(ValueError, match="requires a finite Q"):
        sfca(prep, modes=SINGLE, D_max=D_MAX, Q=np.inf)


def test_unreached_site_and_node(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q)
    supply = res.supply.set_index("supply_id")
    demand = res.demand.set_index("demand_id")
    assert supply.loc["s4", "E_j"] == 0.0
    assert np.isnan(supply.loc["s4", "R_j"])
    assert demand.loc["d6", "A_i"] == 0.0
    assert demand.loc["d6", "n_supply_i"] == 0


def test_demand_is_conserved(prep):
    """Gaussian impedance is bounded by 1, so no more demand comes out than went in."""
    res = sfca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert res.supply["E_j"].sum() <= sum(P.values()) + 1e-9
