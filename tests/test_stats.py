"""Stage 6 — statistic groups, Gini against known distributions."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import D_MAX, TAU, P, Q, S
from test_ifca import MAC, SINGLE

from interaction_models import catchment, ifca, sfca, voronoi
from interaction_models.stats import gini


@pytest.mark.parametrize("n", [2, 3, 10, 100])
def test_gini_of_a_uniform_ramp_matches_the_closed_form(n):
    """Gini of 1..n is exactly (n - 1) / (3n)."""
    assert gini(np.arange(1, n + 1, dtype=float)) == pytest.approx((n - 1) / (3 * n))


def test_gini_of_a_flat_distribution_is_zero():
    assert gini(np.full(50, 7.0)) == pytest.approx(0.0)


def test_gini_approaches_one_for_full_concentration():
    values = np.zeros(1000)
    values[0] = 1.0
    assert gini(values) == pytest.approx(0.999, abs=1e-3)


def test_gini_weights_are_frequency_weights():
    """Weighting [0, 1] by [3, 1] must equal the unweighted Gini of [0, 0, 0, 1]."""
    assert gini(np.array([0.0, 1.0]), np.array([3.0, 1.0])) == pytest.approx(
        gini(np.array([0.0, 0.0, 0.0, 1.0]))
    )


def test_gini_of_nothing_is_nan():
    assert np.isnan(gini(np.zeros(5)))
    assert np.isnan(gini(np.array([])))


# --- groups ---------------------------------------------------------------------


def test_coverage(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["coverage"])
    s = res.stats
    assert s["n_demand"] == 6
    assert s["n_supply"] == 4
    assert s["n_pairs_input"] == 18
    assert s["n_pairs_used"] == 10  # 2+2+2+2+2 for d1..d5, nothing for d6
    assert s["total_demand"] == pytest.approx(sum(P.values()))
    assert s["total_capacity"] == pytest.approx(sum(S.values()))
    assert s["sum_E_j"] == pytest.approx(res.supply["E_j"].sum())
    assert s["demand_capture_rate"] == pytest.approx(s["sum_E_j"] / s["total_demand"])
    assert s["n_supply_zero_exposure"] == 1  # s4
    assert s["n_demand_zero_access"] == 1  # d6


def test_distribution(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["distribution"])
    s = res.stats
    A_i, E_j = res.demand["A_i"].to_numpy(), res.supply["E_j"].to_numpy()
    assert s["mean_A_i"] == pytest.approx(A_i.mean())
    assert s["p90_E_j"] == pytest.approx(np.percentile(E_j, 90))
    assert s["min_A_i"] == 0.0
    assert s["weighted_mean_A_i"] == pytest.approx(
        (res.demand["demand"] * res.demand["A_i"]).sum() / res.demand["demand"].sum()
    )
    assert len([k for k in s if k != "weighted_mean_A_i"]) == 14


def test_inequality(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["inequality"])
    s = res.stats
    assert set(s) == {"gini_A_i", "gini_E_j", "p90_p10_ratio_A_i"}
    assert 0.0 <= s["gini_A_i"] <= 1.0
    assert s["gini_A_i"] == pytest.approx(
        gini(res.demand["A_i"].to_numpy(), res.demand["demand"].to_numpy())
    )


def test_p90_p10_ratio_is_infinite_when_the_tenth_percentile_is_zero(prep):
    """At D_max = 9 only d1 and d3 reach anything, so p10(A_i) is flat zero."""
    res = sfca(prep, modes=SINGLE, D_max=9.0, tau=TAU, Q=Q, stats=["inequality"])
    assert (res.demand["A_i"] == 0).sum() == 4
    assert np.isinf(res.stats["p90_p10_ratio_A_i"])


def test_p90_p10_ratio_is_finite_when_the_tenth_percentile_is_not(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["inequality"])
    p10, p90 = np.percentile(res.demand["A_i"], [10, 90])
    assert res.stats["p90_p10_ratio_A_i"] == pytest.approx(p90 / p10)


def test_choice_set(prep):
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["choice_set"])
    s = res.stats
    # d1..d5 all keep exactly Q=2 pairs; d6 keeps none.
    assert s["mean_choice_set_size"] == pytest.approx(10 / 6)
    assert s["median_choice_set_size"] == pytest.approx(2.0)
    assert s["frac_demand_q_binding"] == pytest.approx(5 / 6)
    assert 1.0 <= s["mean_N_eff_i"] <= Q


def test_choice_set_n_eff_is_two_for_an_even_split(prep):
    """d5 sits 22 minutes from both s1 and s2, so its two G_ij are equal."""
    res = sfca(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q, stats=["choice_set"])
    assert res.stats["mean_N_eff_i"] <= 2.0


def test_choice_set_available_from_ifca_with_q(prep):
    res = ifca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q, stats=["choice_set"])
    assert res.stats["frac_demand_q_binding"] == pytest.approx(3 / 6)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda p: catchment(p, modes=SINGLE, D_max=D_MAX, stats=["choice_set"]),
         "catchment has none"),
        (lambda p: voronoi(p, D_max=D_MAX, stats=["choice_set"]), "voronoi has none"),
        (lambda p: ifca(p, modes=SINGLE, D_max=D_MAX, stats=["choice_set"]),
         "ifca has none"),
    ],
)
def test_choice_set_raises_where_there_is_no_choice_set(prep, call, match):
    with pytest.raises(ValueError, match=match):
        call(prep)


def test_unknown_group_raises(prep):
    with pytest.raises(ValueError, match="unknown stat group"):
        sfca(prep, modes=SINGLE, D_max=D_MAX, Q=Q, stats=["nope"])


def test_stats_are_none_unless_requested(prep):
    assert sfca(prep, modes=SINGLE, D_max=D_MAX, Q=Q).stats is None


def test_stats_work_without_assembling_frames(prep):
    res = sfca(
        prep, modes=SINGLE, D_max=D_MAX, tau=TAU, Q=Q,
        stats=["coverage"], output=(),
    )
    assert res.demand is None and res.supply is None
    assert res.stats["n_pairs_used"] == 10
