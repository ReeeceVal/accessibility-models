"""Golden E_j, A_i and Phi_i for the MAC-3SFCA-E family, plus its structural guarantees.

The guarantees are what the family exists for — monotonicity under a site opening, and
the bound against 3SFCA — so they are tested exhaustively over every subset of the toy
network's supply points rather than on a single configuration.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import D_MAX, TAU, P, Q, S, by_id
from test_ifca import MAC, SINGLE, SURVIVORS_MAC, SURVIVORS_SINGLE, mac, single, top_q

from interaction_models import prepare, sfca, sfca_e
from interaction_models._core import segment_max, segment_starts


def sfca_e_reference(survivors, impedance):
    """Phi_i, G_ij, E_j and the A_i dual, transcribed literally."""
    f = {i: {j: impedance(i, d) for j, d in row.items()} for i, row in survivors.items()}
    total = {i: sum(row.values()) for i, row in f.items()}
    phi = {i: (max(row.values()) if row else 0.0) for i, row in f.items()}
    G = {i: {j: fij / total[i] for j, fij in row.items()} for i, row in f.items() if total[i] > 0}
    E = {j: sum(P[i] * phi[i] * G[i].get(j, 0.0) for i in G) for j in S}
    R = {j: (S[j] / E[j] if E[j] > 0 else 0.0) for j in S}
    A = {
        i: (sum(R[j] * G[i][j] * f[i][j] for j in G[i]) if i in G else 0.0)
        for i in survivors
    }
    return E, A, phi


CASES = {
    "one-mode": (top_q(SURVIVORS_SINGLE, single), single, SINGLE),
    "two-mode": (top_q(SURVIVORS_MAC, mac), mac, MAC),
}


@pytest.mark.parametrize("case", list(CASES))
def test_golden_exposure_accessibility_and_participation(prep, case):
    survivors, impedance, modes = CASES[case]
    res = sfca_e(prep, modes=modes, D_max=D_MAX, tau=TAU, Q=Q)
    assert res.params["n_modes"] == len(modes)

    expected_E, expected_A, expected_phi = sfca_e_reference(survivors, impedance)
    assert by_id(res.supply, "supply_id", "E_j") == pytest.approx(expected_E)
    assert by_id(res.demand, "demand_id", "A_i") == pytest.approx(expected_A)
    assert by_id(res.demand, "demand_id", "Phi_i") == pytest.approx(expected_phi)


# --- the participation step -----------------------------------------------------


def test_participation_is_bounded_by_one(prep):
    """Gaussian impedance is bounded by 1, so Phi_i is a proportion."""
    res = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert ((res.demand["Phi_i"] >= 0.0) & (res.demand["Phi_i"] <= 1.0)).all()


def test_total_exposure_is_participation_weighted_demand(prep):
    """sum_j G_ij = 1, so sum_j E_j collapses to sum_i P_i Phi_i exactly."""
    res = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    expected = (res.demand["demand"] * res.demand["Phi_i"]).sum()
    assert res.supply["E_j"].sum() == pytest.approx(expected)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_participation_and_its_total_are_invariant_to_q(prep, q):
    """The maximising pair is the top-ranked one, so it survives every Q >= 1."""
    reference = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=1)
    res = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=q)
    np.testing.assert_allclose(res.demand["Phi_i"], reference.demand["Phi_i"])
    assert res.supply["E_j"].sum() == pytest.approx(reference.supply["E_j"].sum())


def test_q_still_redistributes_exposure_across_sites(prep):
    """Q is not inert: it moves exposure between sites, it just conserves the total."""
    narrow = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=1).supply["E_j"]
    wide = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=3).supply["E_j"]
    assert not np.allclose(narrow, wide)


def test_segment_max_handles_empty_segments():
    """d6-style empty segments must give 0.0, not the boundary element."""
    codes = np.array([0, 0, 2], dtype=np.int64)
    seg = segment_starts(codes, 4)
    values = np.array([0.25, 0.75, 0.5])
    np.testing.assert_allclose(segment_max(values, seg), [0.75, 0.0, 0.5, 0.0])


# --- the structural guarantees --------------------------------------------------


def subset_frames(frames, keep):
    """The toy network restricted to a subset of its supply points."""
    demand_df, supply_df, cost_df = frames
    return (
        demand_df,
        supply_df[supply_df["supply_id"].isin(keep)].reset_index(drop=True),
        cost_df[cost_df["supply_id"].isin(keep)].reset_index(drop=True),
    )


def total_exposure(model, frames, keep, **kwargs):
    if not keep:
        return 0.0
    prep = prepare(*subset_frames(frames, keep))
    return float(model(prep, output=("supply",), **kwargs).supply["E_j"].sum())


#: Every (site set -> site set + one) opening on the toy network.
SUBSETS = [
    (subset, added)
    for size in range(len(S))
    for subset in itertools.combinations(S, size)
    for added in S
    if added not in subset
]
NON_EMPTY = [s for size in range(1, len(S) + 1) for s in itertools.combinations(S, size)]

#: Both mode sets against every Q the toy network can bind at.
CONFIGS = [
    (f"{name}-Q{q}", {"modes": modes, "D_max": D_MAX, "tau": TAU, "Q": q})
    for name, modes in (("single", SINGLE), ("mac", MAC))
    for q in (1, 2, 3, 4)
]


@pytest.mark.parametrize(("label", "kwargs"), CONFIGS, ids=[c[0] for c in CONFIGS])
def test_opening_a_site_never_lowers_total_exposure(frames, label, kwargs):
    """Monotonicity — the property the family exists for — over every site subset."""
    worst, where = 0.0, None
    for subset, added in SUBSETS:
        delta = (total_exposure(sfca_e, frames, (*subset, added), **kwargs)
                 - total_exposure(sfca_e, frames, subset, **kwargs))
        if delta < worst:
            worst, where = delta, (subset, added)
    assert worst >= -1e-12, f"opening {where[1]} onto {where[0]} lost {worst:.6f}"


def test_3sfca_is_not_monotone_on_the_same_network(frames):
    """The contrast that motivates the family.

    Under the single-mode Gaussian at Q = 3 the choice sets are large enough for an
    opening to *enlarge* one rather than substitute into it, which is exactly when the
    contraharmonic mean can fall. The two-mode split filters harder at tau, so its
    choice sets never grow and the pathology stays hidden — hence the explicit config.
    """
    kwargs = {"modes": SINGLE, "D_max": D_MAX, "tau": TAU, "Q": 3}
    losses = [
        total_exposure(sfca, frames, (*subset, added), **kwargs)
        - total_exposure(sfca, frames, subset, **kwargs)
        for subset, added in SUBSETS
    ]
    assert min(losses) < -1e-9
    # and the same opening, under the same parameters, does not lose under sfca_e
    assert min(
        total_exposure(sfca_e, frames, (*subset, added), **kwargs)
        - total_exposure(sfca_e, frames, subset, **kwargs)
        for subset, added in SUBSETS
    ) >= -1e-12


@pytest.mark.parametrize(("label", "kwargs"), CONFIGS, ids=[c[0] for c in CONFIGS])
def test_total_is_bounded_below_by_3sfca(frames, label, kwargs):
    """Phi_i is a maximum where 3SFCA uses a contraharmonic mean of the same values."""
    for subset in NON_EMPTY:
        assert (total_exposure(sfca_e, frames, subset, **kwargs)
                >= total_exposure(sfca, frames, subset, **kwargs) - 1e-12), subset


def test_equal_impedances_collapse_onto_3sfca(frames):
    """d5 sits 22 minutes from both s1 and s2: equal f, so max == contraharmonic mean."""
    demand_df, supply_df, cost_df = subset_frames(frames, ("s1", "s2"))
    prep = prepare(demand_df[demand_df["demand_id"] == "d5"], supply_df,
                   cost_df[cost_df["demand_id"] == "d5"])
    kwargs = {"modes": SINGLE, "D_max": D_MAX, "tau": TAU, "Q": 2}
    np.testing.assert_allclose(
        sfca_e(prep, **kwargs).supply["E_j"], sfca(prep, **kwargs).supply["E_j"]
    )


# --- capacity, and the conventions ----------------------------------------------


def test_selection_is_capacity_blind_but_load_is_not(frames):
    """E_j must not move with capacity; L_j and A_i must scale with it."""
    demand_df, supply_df, cost_df = frames
    kwargs = {"modes": SINGLE, "D_max": D_MAX, "tau": TAU, "Q": Q}
    base = sfca_e(demand_df, supply_df, cost_df, **kwargs)
    doubled = sfca_e(
        demand_df, supply_df.assign(capacity=supply_df["capacity"] * 2), cost_df, **kwargs
    )
    np.testing.assert_allclose(base.supply["E_j"], doubled.supply["E_j"])
    np.testing.assert_allclose(base.supply["L_j"], doubled.supply["L_j"] * 2)
    np.testing.assert_allclose(base.demand["A_i"] * 2, doubled.demand["A_i"])


def test_operational_load_is_exposure_over_capacity(prep):
    res = sfca_e(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q)
    np.testing.assert_allclose(res.supply["L_j"], res.supply["E_j"] / res.supply["capacity"])
    reached = res.supply["E_j"] > 0
    np.testing.assert_allclose(
        res.supply.loc[reached, "L_j"], 1.0 / res.supply.loc[reached, "R_j"]
    )


def test_capacity_is_optional_for_exposure(frames):
    demand_df, supply_df, cost_df = frames
    res = sfca_e(
        demand_df, supply_df.drop(columns="capacity"), cost_df,
        modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, output=("supply",),
    )
    assert res.supply["E_j"].sum() > 0
    assert res.supply["R_j"].isna().all()
    assert res.supply["L_j"].isna().all()


def test_capacity_is_required_for_accessibility(frames):
    demand_df, supply_df, cost_df = frames
    with pytest.raises(ValueError, match="to compute A_i"):
        sfca_e(
            demand_df, supply_df.drop(columns="capacity"), cost_df,
            modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q,
        )


def test_unreached_site_and_node(prep):
    res = sfca_e(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q)
    supply = res.supply.set_index("supply_id")
    demand = res.demand.set_index("demand_id")
    assert supply.loc["s4", "E_j"] == 0.0
    assert np.isnan(supply.loc["s4", "R_j"])
    assert supply.loc["s4", "L_j"] == 0.0  # capacity is 1.0, so 0 / 1 is well defined
    assert demand.loc["d6", "A_i"] == 0.0
    assert demand.loc["d6", "Phi_i"] == 0.0
    assert demand.loc["d6", "n_supply_i"] == 0


def test_demand_is_conserved(prep):
    res = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert res.supply["E_j"].sum() <= sum(P.values()) + 1e-9


def test_modes_and_q_are_required(prep):
    with pytest.raises(ValueError, match="requires modes"):
        sfca_e(prep, modes=[], D_max=D_MAX, Q=Q)
    with pytest.raises(ValueError, match="requires a finite Q"):
        sfca_e(prep, modes=SINGLE, D_max=D_MAX, Q=None)
    with pytest.raises(ValueError, match="requires a finite Q"):
        sfca_e(prep, modes=SINGLE, D_max=D_MAX, Q=np.inf)
