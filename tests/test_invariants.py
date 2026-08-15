"""Cross-family invariants."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, P

from interaction_models import Mode, catchment, gaussian, prepare, voronoi

TOTAL_DEMAND = sum(P.values())

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]
EQUIVALENT_MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=1.0, decay=gaussian(BETA)),
]


def test_kappa_one_and_equal_beta_makes_mac_identical_to_single_mode(prep):
    single = catchment(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    multi = catchment(prep, modes=EQUIVALENT_MAC, D_max=D_MAX, tau=TAU)
    np.testing.assert_allclose(multi.supply["E_j"], single.supply["E_j"])
    np.testing.assert_allclose(multi.demand["A_i"], single.demand["A_i"])


def test_no_filters_drops_no_pairs(prep):
    res = catchment(prep, modes=SINGLE, D_max=np.inf, tau=0.0)
    assert res.demand["n_supply_i"].sum() == prep.n_pairs


def test_voronoi_conserves_demand(prep):
    for modes in (None, MAC):
        total = voronoi(prep, modes=modes, D_max=D_MAX).supply["E_j"].sum()
        assert total <= TOTAL_DEMAND + 1e-9


def test_voronoi_mac_never_exceeds_voronoi(prep):
    base = voronoi(prep, D_max=D_MAX).supply["E_j"]
    mac = voronoi(prep, modes=MAC, D_max=D_MAX).supply["E_j"]
    assert (mac <= base + 1e-9).all()


def test_prepared_and_raw_frames_agree_bit_for_bit(frames, prep):
    from_frames = catchment(*frames, modes=SINGLE, D_max=D_MAX, tau=TAU)
    from_prep = catchment(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    np.testing.assert_array_equal(from_frames.supply["E_j"], from_prep.supply["E_j"])
    np.testing.assert_array_equal(from_frames.demand["A_i"], from_prep.demand["A_i"])


def test_row_order_of_inputs_does_not_change_results(frames):
    demand_df, supply_df, cost_df = frames
    shuffled = prepare(
        demand_df, supply_df, cost_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    )
    a = catchment(prepare(*frames), modes=MAC, D_max=D_MAX, tau=TAU)
    b = catchment(shuffled, modes=MAC, D_max=D_MAX, tau=TAU)
    np.testing.assert_allclose(a.supply["E_j"], b.supply["E_j"])
    np.testing.assert_allclose(a.demand["A_i"], b.demand["A_i"])


def test_mixing_prepared_and_frames_is_rejected(frames, prep):
    demand_df, supply_df, cost_df = frames
    with pytest.raises(TypeError, match="not both"):
        catchment(prep, supply_df, cost_df)
    with pytest.raises(TypeError, match="all three"):
        catchment(demand_df)


def test_params_describe_every_configuration(prep):
    """params is the full description of a call: every family reports the same keys."""
    from interaction_models import ifca, sfca, sfca_e

    resolved = [
        catchment(prep, D_max=D_MAX).params,
        catchment(prep, modes=SINGLE, D_max=D_MAX, tau=TAU).params,
        catchment(prep, modes=MAC, D_max=D_MAX, tau=TAU).params,
        voronoi(prep, D_max=D_MAX).params,
        voronoi(prep, modes=MAC, D_max=D_MAX).params,
        ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU).params,
        ifca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=2).params,
        ifca(prep, modes=MAC, D_max=D_MAX, tau=TAU).params,
        ifca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2).params,
        sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=2).params,
        sfca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2).params,
        sfca_e(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=2).params,
        sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2).params,
    ]
    assert all(set(p) == {"D_max", "Q", "tau", "n_modes"} for p in resolved)
    assert [(p["n_modes"], p["Q"], p["tau"]) for p in resolved] == [
        (0, None, None), (1, None, TAU), (2, None, TAU),
        (0, None, None), (2, None, None),
        (1, None, TAU), (1, 2, TAU), (2, None, TAU), (2, 2, TAU),
        (1, 2, TAU), (2, 2, TAU),
        (1, 2, TAU), (2, 2, TAU),
    ]


def test_every_family_agrees_on_the_toy_network_totals(prep):
    """No family may emit more exposure than there is demand."""
    from interaction_models import sfca, sfca_e

    results = [
        catchment(prep, modes=SINGLE, D_max=D_MAX, tau=TAU),
        voronoi(prep, modes=MAC, D_max=D_MAX),
        sfca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2),
        sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2),
    ]
    # catchment double-counts by construction; the partitioning families cannot.
    for res in results[1:]:
        assert res.supply["E_j"].sum() <= TOTAL_DEMAND + 1e-9
    assert set(results[-1].supply.columns) >= {"E_j", "R_j", "L_j"}
    assert "Phi_i" in results[-1].demand.columns
    assert "Phi_i" not in results[2].demand.columns
