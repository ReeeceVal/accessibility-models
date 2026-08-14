"""Stage 2 — f_multi assembly, NaN renormalisation and the kappa reach gate."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, gauss

from interaction_models import Mode, gaussian, validate_inputs
from interaction_models.modes import impedance, is_kappa_only, reach_weight


def all_rows(prep):
    return np.arange(prep.n_pairs)


def pair_row(prep, demand_id, supply_id, frames):
    demand_df, supply_df, _ = frames
    d = list(demand_df["demand_id"]).index(demand_id)
    s = list(supply_df["supply_id"]).index(supply_id)
    (row,) = np.flatnonzero((prep.demand_code == d) & (prep.supply_code == s))
    return row


def test_gaussian_values_and_metadata():
    f = gaussian(BETA)
    assert f.__name__ == "gaussian(800)"
    assert f.beta == BETA
    np.testing.assert_allclose(f(np.array([0.0, 20.0])), [1.0, np.exp(-400 / 800)])


def test_single_mode_impedance_is_plain_decay(prep, frames):
    modes = [Mode(share=1.0, decay=gaussian(BETA))]
    f = impedance(prep, modes, all_rows(prep))
    assert f[pair_row(prep, "d1", "s1", frames)] == pytest.approx(gauss(5.0))
    assert f[pair_row(prep, "d5", "s3", frames)] == pytest.approx(gauss(44.0))


def test_two_mode_impedance_matches_the_share_weighted_sum(prep, frames):
    """f_multi = s*f(d) + (1-s)*f(kappa*d), hand-computed for d2 (pv_share = 0.5)."""
    modes = [
        Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
        Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
    ]
    f = impedance(prep, modes, all_rows(prep))
    expected = 0.5 * gauss(15.0) + 0.5 * gauss(2 * 15.0)
    assert f[pair_row(prep, "d2", "s1", frames)] == pytest.approx(expected)


def test_kappa_one_and_equal_beta_reduces_to_single_mode(prep):
    single = impedance(prep, [Mode(share=1.0, decay=gaussian(BETA))], all_rows(prep))
    multi = impedance(
        prep,
        [
            Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
            Mode(share="mbt_share", kappa=1.0, decay=gaussian(BETA)),
        ],
        all_rows(prep),
    )
    np.testing.assert_allclose(multi, single)


def test_nan_cost_renormalises_over_available_modes(prep, frames):
    """d1-s2 has no walk route, so the walk share redistributes across pv and mbt."""
    modes = [
        Mode(share=0.5, cost="cost_default", kappa=1.0, decay=gaussian(BETA)),
        Mode(share=0.3, cost="cost_default", kappa=KAPPA, decay=gaussian(BETA)),
        Mode(share=0.2, cost="w_cost", kappa=1.0, decay=gaussian(2000.0)),
    ]
    f = impedance(prep, modes, all_rows(prep))

    no_walk = (0.5 * gauss(20.0) + 0.3 * gauss(40.0)) / 0.8
    assert f[pair_row(prep, "d1", "s2", frames)] == pytest.approx(no_walk)

    with_walk = 0.5 * gauss(5.0) + 0.3 * gauss(10.0) + 0.2 * gauss(10.0, 2000.0)
    assert f[pair_row(prep, "d1", "s1", frames)] == pytest.approx(with_walk)


def test_all_modes_unavailable_gives_zero(prep, frames):
    modes = [Mode(share=1.0, cost="w_cost", decay=gaussian(2000.0))]
    f = impedance(prep, modes, all_rows(prep))
    assert f[pair_row(prep, "d1", "s2", frames)] == 0.0
    assert f[pair_row(prep, "d1", "s1", frames)] == pytest.approx(gauss(10.0, 2000.0))


def test_decay_none_is_unit_impedance(prep):
    f = impedance(prep, [Mode(share=1.0)], all_rows(prep))
    np.testing.assert_array_equal(f, np.ones(prep.n_pairs))


def test_reach_weight_gates_each_mode_without_renormalising(prep, frames):
    """Eq. vor_mac: pv always reaches (kappa=1 <= D_max), mbt only where 2d <= D_max."""
    modes = [Mode(share="pv_share", kappa=1.0), Mode(share="mbt_share", kappa=KAPPA)]
    w = reach_weight(prep, modes, all_rows(prep), D_MAX)

    # d2-s2: cost 10, 2*10 = 20 <= 45, so both modes reach -> weight 1.
    assert w[pair_row(prep, "d2", "s2", frames)] == pytest.approx(1.0)
    # d5-s1: cost 22, 2*22 = 44 <= 45 -> still both.
    assert w[pair_row(prep, "d5", "s1", frames)] == pytest.approx(1.0)
    # d4-s1: cost 30, 2*30 = 60 > 45, so only the pv share (0.25) survives.
    assert w[pair_row(prep, "d4", "s1", frames)] == pytest.approx(0.25)


def test_reach_weight_treats_missing_route_as_out_of_reach(prep, frames):
    modes = [Mode(share=0.5, cost="cost_default"), Mode(share=0.5, cost="w_cost")]
    w = reach_weight(prep, modes, all_rows(prep), D_MAX)
    assert w[pair_row(prep, "d1", "s1", frames)] == pytest.approx(1.0)
    assert w[pair_row(prep, "d1", "s2", frames)] == pytest.approx(0.5)


def test_is_kappa_only():
    assert is_kappa_only([Mode(share=1.0, kappa=2.0)])
    assert not is_kappa_only([Mode(share=0.5), Mode(share=0.5, cost="w_cost")])


# --- mode-aware validation (plan 10) -------------------------------------------


def test_validation_rejects_missing_share_column(frames):
    with pytest.raises(ValueError, match="share column 'nope'"):
        validate_inputs(*frames, modes=[Mode(share="nope")])


def test_validation_rejects_missing_cost_column(frames):
    with pytest.raises(ValueError, match="cost column 'nope'"):
        validate_inputs(*frames, modes=[Mode(share="pv_share", cost="nope")])


def test_validation_rejects_shares_out_of_range(frames):
    demand_df, supply_df, cost_df = frames
    demand_df = demand_df.assign(pv_share=1.5)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_inputs(demand_df, supply_df, cost_df, modes=[Mode(share="pv_share")])


def test_validation_rejects_shares_not_summing_to_one(frames):
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_inputs(*frames, modes=[Mode(share="pv_share"), Mode(share="pv_share")])


def test_validation_accepts_constant_shares(frames):
    validate_inputs(*frames, modes=[Mode(share=0.4), Mode(share=0.6)])
