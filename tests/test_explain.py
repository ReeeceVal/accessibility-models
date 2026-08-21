"""``explain()``: the per-pair terms every family sums away, scoped to one demand node.

Golden values reuse the survivor sets already hand-derived in ``test_ifca.py`` and
``test_sfca.py`` for the same toy network. The contract under test is that ``weight``
reconstructs the family's own aggregate exactly -- ``P_i * weight`` for catchment and
voronoi, ``weight`` summed by site times ``S_j`` for ifca, ``P_i * weight * f_multi`` for
sfca/sfca_e -- and that the Prepared and Compiled entry points agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, P, Q, S

from interaction_models import Mode, catchment, compile_f, explain, gaussian, ifca, sfca, voronoi

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]


def test_sfca_weight_is_g_ij_and_reconstructs_e_j(prep):
    frame = explain(prep, family="sfca", demand_id=list(P), modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    res = sfca(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)

    # G_ij sums to 1.0 within every non-empty node.
    sums = frame.groupby("demand_id")["weight"].sum()
    assert sums[sums.index != "d6"].to_numpy() == pytest.approx(1.0)

    frame = frame.assign(P_i=frame["demand_id"].map(P))
    E_j = (frame["P_i"] * frame["weight"] * frame["f_multi"]).groupby(frame["supply_id"]).sum()
    want = res.supply.set_index("supply_id")["E_j"]
    for site in E_j.index:
        assert E_j[site] == pytest.approx(want[site])


def test_ifca_weight_is_r_i_f_ij_and_reconstructs_c_j(prep):
    frame = explain(prep, family="ifca", demand_id=list(P), modes=MAC, D_max=D_MAX, tau=TAU)
    res = ifca(prep, modes=MAC, D_max=D_MAX, tau=TAU)

    C_j = frame.groupby("supply_id")["weight"].sum()
    E_j = C_j * pd.Series(S).reindex(C_j.index)
    want = res.supply.set_index("supply_id")["E_j"]
    for site in E_j.index:
        assert E_j[site] == pytest.approx(want[site])


def test_catchment_weight_is_p_i_f_ij_and_reconstructs_e_j(prep):
    frame = explain(prep, family="catchment", demand_id=list(P), D_max=D_MAX)
    res = catchment(prep, D_max=D_MAX)

    E_j = frame.groupby("supply_id")["weight"].sum()
    want = res.supply.set_index("supply_id")["E_j"]
    for site in E_j.index:
        assert E_j[site] == pytest.approx(want[site])
    # No decay: f_multi is 1 everywhere, so weight == demand directly.
    np.testing.assert_array_equal(frame["weight"], frame["demand_id"].map(P))


def test_voronoi_weight_is_a_winner_flag_and_reconstructs_e_j(prep):
    frame = explain(prep, family="voronoi", demand_id=list(P), D_max=D_MAX)
    res = voronoi(prep, D_max=D_MAX)

    assert set(frame["weight"].unique()) <= {0.0, 1.0}
    # Exactly one winner per node that reaches anybody.
    winners = frame.groupby("demand_id")["weight"].sum()
    assert (winners == 1.0).all()
    assert "d6" not in winners.index  # reaches nobody, so it contributes no rows at all

    frame = frame.assign(P_i=frame["demand_id"].map(P))
    E_j = (frame["P_i"] * frame["weight"]).groupby(frame["supply_id"]).sum()
    want = res.supply.set_index("supply_id")["E_j"]
    for site in E_j.index:
        assert E_j[site] == pytest.approx(want[site])


def test_voronoi_tie_break_matches_the_model(prep):
    """d5 is equidistant from s1 and s2; both pick s1 on the lowest-supply-code tie-break."""
    frame = explain(prep, family="voronoi", demand_id="d5", D_max=D_MAX)
    assert frame.loc[frame["weight"] == 1.0, "supply_id"].iloc[0] == "s1"


def test_unreached_node_gives_an_empty_frame(prep):
    frame = explain(prep, family="sfca", demand_id="d6", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert frame.empty
    assert list(frame.columns) == ["demand_id", "supply_id", "cost_default", "f_multi", "weight"]


def test_rows_are_f_descending(prep):
    frame = explain(prep, family="sfca", demand_id="d1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert np.all(np.diff(frame["f_multi"].to_numpy()) <= 0.0)


def test_q_bounds_the_choice_set(prep):
    frame = explain(prep, family="sfca", demand_id="d1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert len(frame) == Q


def test_multiple_demand_ids_concatenate_in_the_given_order(prep):
    frame = explain(
        prep, family="sfca", demand_id=["d2", "d1"], modes=MAC, D_max=D_MAX, tau=TAU, Q=Q
    )
    assert frame["demand_id"].tolist() == ["d2", "d2", "d1", "d1"]


def test_open_mask_matches_the_model_functions_own_open_mask(prep):
    mask = np.isin(list(S), ["s1", "s3"])
    frame = explain(
        prep, family="sfca", demand_id="d1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q, open_mask=mask
    )
    assert set(frame["supply_id"]) <= {"s1", "s3"}


def test_compiled_matches_the_prepared_call(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    got = explain(comp, family="sfca", demand_id="d1", Q=Q)
    want = explain(prep, family="sfca", demand_id="d1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    pd.testing.assert_frame_equal(got, want)


def test_compiled_matches_under_an_open_mask(prep):
    mask = np.isin(list(S), ["s1", "s3"])
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    got = explain(comp, family="ifca", demand_id="d1", Q=Q, open_mask=mask)
    want = explain(
        prep, family="ifca", demand_id="d1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q, open_mask=mask
    )
    pd.testing.assert_frame_equal(got, want)


def test_unknown_family_is_rejected(prep):
    with pytest.raises(ValueError, match="family must be"):
        explain(prep, family="bogus", demand_id="d1")


def test_unknown_demand_id_is_rejected(prep):
    with pytest.raises(ValueError, match="not found"):
        explain(prep, family="sfca", demand_id="zzz", modes=MAC, Q=Q)


def test_sfca_requires_a_finite_q(prep):
    with pytest.raises(ValueError, match="requires a finite Q"):
        explain(prep, family="sfca", demand_id="d1", modes=MAC)


def test_catchment_rejects_q(prep):
    with pytest.raises(ValueError, match="does not accept Q"):
        explain(prep, family="catchment", demand_id="d1", Q=2)


def test_voronoi_rejects_tau_and_q(prep):
    with pytest.raises(ValueError, match="does not accept tau or Q"):
        explain(prep, family="voronoi", demand_id="d1", tau=0.1)
    with pytest.raises(ValueError, match="does not accept tau or Q"):
        explain(prep, family="voronoi", demand_id="d1", Q=2)


def test_ifca_sfca_sfca_e_require_modes(prep):
    for family in ("ifca", "sfca", "sfca_e"):
        with pytest.raises(ValueError, match="requires modes"):
            explain(prep, family=family, demand_id="d1", Q=Q)


def test_ifca_requires_capacity(frames):
    demand_df, supply_df, cost_df = frames
    from interaction_models import prepare

    no_capacity = prepare(demand_df, supply_df.drop(columns="capacity"), cost_df)
    with pytest.raises(ValueError, match="S_j is in the denominator"):
        explain(no_capacity, family="ifca", demand_id="d1", modes=MAC, D_max=D_MAX)


def test_voronoi_rejects_a_compiled(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(TypeError, match="voronoi has no impedance"):
        explain(comp, family="voronoi", demand_id="d1")


def test_compiled_rejects_baked_parameters(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(ValueError, match="modes is baked"):
        explain(comp, family="sfca", demand_id="d1", modes=MAC, Q=Q)


def _by_demand_subset(prep, family, supply_id, **kwargs):
    """The rows demand_id-scoped explain() would give for ``supply_id``, sorted to match
    supply_id-scoped explain()'s own ordering -- weight descending, cost_default
    ascending. Concatenating a mix of empty and non-empty per-node frames can leave the
    id columns as ``object`` on one side and pandas' string dtype on the other (a
    pre-existing quirk of the demand_id path, not specific to either axis), so callers
    compare with ``check_dtype=False``.
    """
    frame = explain(prep, family=family, demand_id=list(P), **kwargs)
    frame = frame[frame["supply_id"] == supply_id]
    return frame.sort_values(["weight", "cost_default"], ascending=[False, True]).reset_index(
        drop=True
    )


def test_supply_id_and_demand_id_are_mutually_exclusive(prep):
    with pytest.raises(ValueError, match="exactly one"):
        explain(prep, family="sfca", modes=MAC, Q=Q)
    with pytest.raises(ValueError, match="exactly one"):
        explain(prep, family="sfca", demand_id="d1", supply_id="s1", modes=MAC, Q=Q)


def test_unknown_supply_id_is_rejected(prep):
    with pytest.raises(ValueError, match="not found"):
        explain(prep, family="sfca", supply_id="zzz", modes=MAC, Q=Q)


def test_supply_id_sfca_matches_the_demand_id_view(prep):
    got = explain(prep, family="sfca", supply_id="s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    want = _by_demand_subset(prep, "sfca", "s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_supply_id_ifca_matches_the_demand_id_view(prep):
    got = explain(prep, family="ifca", supply_id="s3", modes=MAC, D_max=D_MAX, tau=TAU)
    want = _by_demand_subset(prep, "ifca", "s3", modes=MAC, D_max=D_MAX, tau=TAU)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_supply_id_catchment_matches_the_demand_id_view(prep):
    got = explain(prep, family="catchment", supply_id="s2", D_max=D_MAX)
    want = _by_demand_subset(prep, "catchment", "s2", D_max=D_MAX)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_supply_id_voronoi_matches_the_demand_id_view(prep):
    got = explain(prep, family="voronoi", supply_id="s1", D_max=D_MAX)
    want = _by_demand_subset(prep, "voronoi", "s1", D_max=D_MAX)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_supply_id_rows_are_weight_descending(prep):
    frame = explain(prep, family="sfca", supply_id="s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert np.all(np.diff(frame["weight"].to_numpy()) <= 0.0)


def test_supply_id_accepts_a_sequence(prep):
    frame = explain(
        prep, family="sfca", supply_id=["s1", "s3"], modes=MAC, D_max=D_MAX, tau=TAU, Q=Q
    )
    assert set(frame["supply_id"]) <= {"s1", "s3"}
    assert set(frame["supply_id"]) == {"s1", "s3"}


def test_unreached_supply_id_gives_an_empty_frame(prep):
    """s4 is beyond D_max for every demand node."""
    frame = explain(prep, family="sfca", supply_id="s4", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    assert frame.empty
    assert list(frame.columns) == ["demand_id", "supply_id", "cost_default", "f_multi", "weight"]


def test_open_mask_closing_the_queried_site_gives_an_empty_frame(prep):
    mask = np.isin(list(S), ["s2", "s3"])  # s1 excluded
    frame = explain(
        prep, family="sfca", supply_id="s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q, open_mask=mask
    )
    assert frame.empty


def test_compiled_supports_supply_id(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    got = explain(comp, family="sfca", supply_id="s1", Q=Q)
    want = explain(prep, family="sfca", supply_id="s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q)
    pd.testing.assert_frame_equal(got, want)


def test_compiled_supply_id_matches_under_an_open_mask(prep):
    mask = np.isin(list(S), ["s1", "s3"])
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    got = explain(comp, family="ifca", supply_id="s1", Q=Q, open_mask=mask)
    want = explain(
        prep, family="ifca", supply_id="s1", modes=MAC, D_max=D_MAX, tau=TAU, Q=Q, open_mask=mask
    )
    pd.testing.assert_frame_equal(got, want)


def test_voronoi_rejects_a_compiled_for_supply_id_too(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(TypeError, match="voronoi has no impedance"):
        explain(comp, family="voronoi", supply_id="s1")


def test_compiled_open_mask_must_be_a_subset_of_sites(prep, frames):
    supply_df = frames[1]
    comp = compile_f(
        prep,
        modes=MAC,
        D_max=D_MAX,
        tau=TAU,
        sites=supply_df["supply_id"].isin(["s1", "s2"]).to_numpy(),
    )
    with pytest.raises(ValueError, match="subset of the sites"):
        explain(
            comp,
            family="sfca",
            demand_id="d1",
            Q=Q,
            open_mask=supply_df["supply_id"].isin(["s1", "s3"]).to_numpy(),
        )
