"""Golden E_j and A_i for the Voronoi family."""

from __future__ import annotations

import pytest
from conftest import D_MAX, P, S, by_id

from interaction_models import Mode, voronoi

# kappa = 3 so the minibus share of d5 (nearest site at cost 22) fails the reach gate:
# 3 * 22 = 66 > 45. At kappa = 2 every node's nearest site is reachable by both modes
# and the mode split would make no difference at all.
KAPPA_VOR = 3.0

MAC_MODES = [Mode(share="pv_share", kappa=1.0), Mode(share="mbt_share", kappa=KAPPA_VOR)]

# j*(i): d1->s1 (5), d2->s2 (10), d3->s2 (8), d4->s3 (12), d5->s1 (22, tie with s2
# broken on the lower supply code), d6 unreachable.
E_VOR = {"s1": P["d1"] + P["d5"], "s2": P["d2"] + P["d3"], "s3": P["d4"], "s4": 0.0}
E_MAC = {"s1": P["d1"] + 0.8 * P["d5"], "s2": P["d2"] + P["d3"], "s3": P["d4"], "s4": 0.0}


def run(prep, **kwargs):
    return voronoi(prep, D_max=D_MAX, **kwargs)


def test_params_report_the_configuration(prep):
    """Voronoi has no impedance, so tau and Q are always None."""
    assert run(prep).params == {
        "D_max": D_MAX, "Q": None, "tau": None, "n_modes": 0, "n_open": None,
        "n_width_fallback": None,
    }
    assert run(prep, modes=[Mode(share=1.0)]).params["n_modes"] == 1
    assert run(prep, modes=MAC_MODES).params["n_modes"] == 2


def test_voronoi_exposure(prep):
    assert by_id(run(prep).supply, "supply_id", "E_j") == pytest.approx(E_VOR)


def test_voronoi_ties_break_on_lowest_supply_code(prep):
    """d5 is 22 minutes from both s1 and s2; s1 wins because it sorts first."""
    n_demand_j = by_id(run(prep).supply, "supply_id", "n_demand_j")
    assert n_demand_j == {"s1": 2, "s2": 2, "s3": 1, "s4": 0}


def test_voronoi_accessibility(prep):
    demand = by_id(run(prep).demand, "demand_id", "A_i")
    assert demand["d1"] == pytest.approx(S["s1"] / E_VOR["s1"])
    assert demand["d3"] == pytest.approx(S["s2"] / E_VOR["s2"])
    assert demand["d4"] == pytest.approx(S["s3"] / E_VOR["s3"])
    assert demand["d6"] == 0.0


def test_voronoi_assigns_each_node_at_most_once(prep):
    n_supply_i = by_id(run(prep).demand, "demand_id", "n_supply_i")
    assert n_supply_i == {"d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1, "d6": 0}


def test_voronoi_mac_exposure(prep):
    """Eq. vor_mac: d5's minibus share (0.2) cannot reach s1, so it is lost, not moved."""
    assert by_id(run(prep, modes=MAC_MODES).supply, "supply_id", "E_j") == pytest.approx(E_MAC)


def test_voronoi_mac_accessibility(prep):
    demand = by_id(run(prep, modes=MAC_MODES).demand, "demand_id", "A_i")
    assert demand["d5"] == pytest.approx(S["s1"] / E_MAC["s1"])
    assert demand["d2"] == pytest.approx(S["s2"] / E_MAC["s2"])


def test_voronoi_mac_does_not_move_the_assignment(prep):
    """kappa is a constant scale, so mode split can never change j*."""
    assert by_id(run(prep).supply, "supply_id", "n_demand_j") == by_id(
        run(prep, modes=MAC_MODES).supply, "supply_id", "n_demand_j"
    )


def test_decay_is_ignored(prep):
    from interaction_models import gaussian

    with_decay = [Mode(share="pv_share", kappa=1.0, decay=gaussian(800))] + MAC_MODES[1:]
    assert by_id(run(prep, modes=with_decay).supply, "supply_id", "E_j") == pytest.approx(E_MAC)
