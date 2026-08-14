"""Golden E_j and A_i for the Catchment family."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, P, S, by_id, gauss

from interaction_models import Mode, catchment, gaussian


def fmulti(pv: float, d: float) -> float:
    """The two-mode form s*f(d) + (1-s)*f(kappa*d), computed on scalars."""
    return pv * gauss(d) + (1.0 - pv) * gauss(KAPPA * d)


MAC_MODES = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]


def run(prep, **kwargs):
    return catchment(prep, D_max=D_MAX, **kwargs)


# --- No impedance ---------------------------------------------------------------


def test_no_decay_reports_no_impedance(prep):
    """tau is None in params when no mode carries a decay, since it cannot bite."""
    assert run(prep).params == {"D_max": D_MAX, "Q": None, "tau": None, "n_modes": 0}
    assert run(prep, modes=[Mode(share=1.0)]).params["tau"] is None


def test_binary_exposure_is_reachable_demand(prep):
    supply = by_id(run(prep).supply, "supply_id", "E_j")
    assert supply["s1"] == pytest.approx(P["d1"] + P["d2"] + P["d4"] + P["d5"])
    assert supply["s2"] == pytest.approx(P["d1"] + P["d2"] + P["d3"] + P["d5"])
    assert supply["s3"] == pytest.approx(P["d1"] + P["d2"] + P["d3"] + P["d4"] + P["d5"])
    assert supply["s4"] == 0.0


def test_binary_accessibility_is_reachable_capacity(prep):
    demand = by_id(run(prep).demand, "demand_id", "A_i")
    assert demand["d1"] == pytest.approx(S["s1"] + S["s2"] + S["s3"])
    assert demand["d3"] == pytest.approx(S["s2"] + S["s3"])
    assert demand["d4"] == pytest.approx(S["s3"] + S["s1"])
    assert demand["d6"] == 0.0


# --- Single-mode impedance ------------------------------------------------------


def test_single_mode_params(prep):
    res = run(prep, modes=[Mode(share=1.0, decay=gaussian(BETA))], tau=TAU)
    assert res.params == {"D_max": D_MAX, "Q": None, "tau": TAU, "n_modes": 1}


def test_catchment_f_exposure(prep):
    """tau = 0.1 drops d5-s3, whose f(44) = 0.0889."""
    res = run(prep, modes=[Mode(share=1.0, decay=gaussian(BETA))], tau=TAU)
    supply = by_id(res.supply, "supply_id", "E_j")
    assert supply["s1"] == pytest.approx(
        P["d1"] * gauss(5) + P["d2"] * gauss(15) + P["d4"] * gauss(30) + P["d5"] * gauss(22)
    )
    assert supply["s3"] == pytest.approx(
        P["d1"] * gauss(40) + P["d2"] * gauss(35) + P["d3"] * gauss(25) + P["d4"] * gauss(12)
    )
    assert supply["s4"] == 0.0


def test_catchment_f_accessibility(prep):
    res = run(prep, modes=[Mode(share=1.0, decay=gaussian(BETA))], tau=TAU)
    demand = by_id(res.demand, "demand_id", "A_i")
    assert demand["d1"] == pytest.approx(
        S["s1"] * gauss(5) + S["s2"] * gauss(20) + S["s3"] * gauss(40)
    )
    assert demand["d5"] == pytest.approx(S["s1"] * gauss(22) + S["s2"] * gauss(22))
    assert by_id(res.demand, "demand_id", "n_supply_i")["d5"] == 2


# --- Multi-modal impedance ------------------------------------------------------


def test_multi_mode_params(prep):
    res = run(prep, modes=MAC_MODES, tau=TAU)
    assert res.params["n_modes"] == 2


def test_catchment_mac_exposure(prep):
    """The multi-modal floor bites harder: d3-s3, d4-s1 and d5-s3 all fall below tau."""
    res = run(prep, modes=MAC_MODES, tau=TAU)
    supply = by_id(res.supply, "supply_id", "E_j")
    assert supply["s1"] == pytest.approx(
        P["d1"] * fmulti(1.0, 5) + P["d2"] * fmulti(0.5, 15) + P["d5"] * fmulti(0.8, 22)
    )
    assert supply["s2"] == pytest.approx(
        P["d1"] * fmulti(1.0, 20)
        + P["d2"] * fmulti(0.5, 10)
        + P["d3"] * fmulti(0.0, 8)
        + P["d5"] * fmulti(0.8, 22)
    )
    assert supply["s3"] == pytest.approx(
        P["d1"] * fmulti(1.0, 40) + P["d2"] * fmulti(0.5, 35) + P["d4"] * fmulti(0.25, 12)
    )


def test_catchment_mac_accessibility(prep):
    res = run(prep, modes=MAC_MODES, tau=TAU)
    demand = by_id(res.demand, "demand_id", "A_i")
    assert demand["d4"] == pytest.approx(S["s3"] * fmulti(0.25, 12))
    assert demand["d3"] == pytest.approx(S["s2"] * fmulti(0.0, 8))
    assert demand["d6"] == 0.0


# --- Result mechanics -----------------------------------------------------------


def test_models_emit_no_warnings(prep):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        catchment(prep, D_max=D_MAX, modes=MAC_MODES, tau=TAU)


def test_passthrough_columns_survive_unmodified(prep, frames):
    demand_df, supply_df, _ = frames
    res = run(prep)
    assert list(res.demand.columns) == [
        "demand_id",
        "demand",
        "A_i",
        "SPAR",
        "n_supply_i",
        "pv_share",
        "mbt_share",
        "province",
    ]
    assert list(res.supply.columns) == [
        "supply_id",
        "capacity",
        "E_j",
        "R_j",
        "n_demand_j",
        "site_type",
    ]
    assert list(res.demand["province"]) == list(demand_df["province"])
    assert list(res.supply["site_type"]) == list(supply_df["site_type"])


def test_output_selection(prep):
    assert run(prep, output=("supply",)).demand is None
    assert run(prep, output=("demand",)).supply is None


def test_spar_is_a_i_over_its_mean(prep):
    res = run(prep)
    np.testing.assert_allclose(res.demand["SPAR"], res.demand["A_i"] / res.demand["A_i"].mean())


def test_r_j_is_nan_where_exposure_is_zero(prep):
    supply = run(prep).supply.set_index("supply_id")
    assert np.isnan(supply.loc["s4", "R_j"])
    assert supply.loc["s1", "R_j"] == pytest.approx(S["s1"] / supply.loc["s1", "E_j"])


def test_accessibility_requires_capacity(frames):
    demand_df, supply_df, cost_df = frames
    with pytest.raises(ValueError, match="needs a 'capacity' column"):
        catchment(demand_df, supply_df.drop(columns="capacity"), cost_df, D_max=D_MAX)
