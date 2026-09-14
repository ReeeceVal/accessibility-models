"""``compile_f`` against the uncompiled call it has to reproduce.

Compiling hoists pipeline stages 0-5 out of the model, so the contract is that it
changes *when* work happens and never *what comes out*. On the kappa-only fast path the
compiled rows keep the prepared order, so results are bit-for-bit identical. Where a mode
carries its own cost column the rows are physically reordered into f-descending order —
the point of compiling — so float accumulation runs in a different order and the
comparison is to within a few ulp rather than exact.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, S

from interaction_models import (
    Mode,
    catchment,
    compile_f,
    gaussian,
    ifca,
    sfca,
    sfca_e,
    sweep,
    voronoi,
)
from interaction_models._core import segment_lengths

SINGLE = [Mode(share=1.0, decay=gaussian(BETA))]
MAC = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", kappa=KAPPA, decay=gaussian(BETA)),
]
OWN_COL = [
    Mode(share="pv_share", kappa=1.0, decay=gaussian(BETA)),
    Mode(share="mbt_share", cost="w_cost", kappa=1.0, decay=gaussian(BETA)),
]

SITES = list(S)
SUBSETS = [
    subset for size in range(1, len(SITES) + 1) for subset in itertools.combinations(SITES, size)
]

#: (label, model, extra kwargs, modes) — every family that accepts a Compiled.
CASES = [
    ("catchment", catchment, {}, MAC),
    ("ifca", ifca, {"Q": 2}, MAC),
    ("ifca-noQ", ifca, {}, MAC),
    ("sfca", sfca, {"Q": 2}, MAC),
    ("sfca_e-single", sfca_e, {"Q": 2}, SINGLE),
    ("sfca_e-mac", sfca_e, {"Q": 2}, MAC),
    ("sfca_e-Q1", sfca_e, {"Q": 1}, MAC),
    ("sfca_e-Q4", sfca_e, {"Q": 4}, MAC),
    ("sfca_e-owncol", sfca_e, {"Q": 2}, OWN_COL),
    ("sfca-owncol", sfca, {"Q": 2}, OWN_COL),
]


def mask_for(supply_df, keep) -> np.ndarray:
    return supply_df["supply_id"].isin(keep).to_numpy()


def same(got, want, exact, label):
    """Exact on the fast path, a few ulp where compiling reorders rows."""
    for frame, column in (("supply", "E_j"), ("demand", "A_i")):
        a = getattr(got, frame)[column].to_numpy()
        b = getattr(want, frame)[column].to_numpy()
        if exact:
            np.testing.assert_array_equal(a, b, err_msg=f"{label} {frame}.{column}")
        else:
            np.testing.assert_allclose(
                a, b, rtol=1e-12, atol=0.0, err_msg=f"{label} {frame}.{column}"
            )


@pytest.mark.parametrize(("label", "model", "extra", "modes"), CASES, ids=[c[0] for c in CASES])
def test_compiled_reproduces_the_uncompiled_call(frames, prep, label, model, extra, modes):
    exact = all(m.cost == "cost_default" for m in modes)
    comp = compile_f(prep, modes=modes, D_max=D_MAX, tau=TAU)
    same(
        model(comp, **extra), model(prep, modes=modes, D_max=D_MAX, tau=TAU, **extra), exact, label
    )


@pytest.mark.parametrize(("label", "model", "extra", "modes"), CASES, ids=[c[0] for c in CASES])
def test_compiled_reproduces_the_uncompiled_call_under_every_mask(
    frames, prep, label, model, extra, modes
):
    """The optimisation path: one compile, many candidate networks."""
    exact = all(m.cost == "cost_default" for m in modes)
    supply_df = frames[1]
    comp = compile_f(prep, modes=modes, D_max=D_MAX, tau=TAU)
    for keep in SUBSETS:
        mask = mask_for(supply_df, keep)
        same(
            model(comp, open_mask=mask, **extra),
            model(prep, modes=modes, D_max=D_MAX, tau=TAU, open_mask=mask, **extra),
            exact,
            f"{label} {keep}",
        )


def test_compiled_rows_are_f_descending_within_each_segment(prep):
    """The invariant that makes rank-among-open a free prefix truncation."""
    for modes in (MAC, OWN_COL):
        comp = compile_f(prep, modes=modes, D_max=D_MAX, tau=TAU)
        for start, length in zip(comp.seg[:-1], segment_lengths(comp.seg), strict=True):
            block = comp.f[start : start + length]
            assert np.all(np.diff(block) <= 0.0), (modes, block)


def test_sites_bakes_a_permanent_restriction(frames, prep):
    """compile_f(sites=...) == the same restriction passed per call as open_mask."""
    supply_df = frames[1]
    keep = ["s1", "s3"]
    mask = mask_for(supply_df, keep)
    baked = sfca_e(compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, sites=mask), Q=2)
    per_call = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, open_mask=mask)
    same(baked, per_call, exact=True, label="sites")
    assert baked.params["n_open"] == 2


def test_open_mask_must_be_a_subset_of_sites(frames, prep):
    supply_df = frames[1]
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, sites=mask_for(supply_df, ["s1", "s2"]))
    # narrowing within the compiled sites is fine
    assert sfca_e(comp, Q=2, open_mask=mask_for(supply_df, ["s1"])).params["n_open"] == 1
    with pytest.raises(ValueError, match="subset of the sites"):
        sfca_e(comp, Q=2, open_mask=mask_for(supply_df, ["s1", "s3"]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"modes": MAC}, "modes is baked"),
        ({"D_max": D_MAX}, "D_max is baked"),
        ({"tau": TAU}, "tau is baked"),
        ({"modes": MAC, "tau": TAU}, "modes, tau are baked"),
    ],
    ids=["modes", "D_max", "tau", "two"],
)
def test_parameters_baked_into_the_compile_are_rejected(prep, kwargs, match):
    """The one way this design could silently return wrong numbers."""
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(ValueError, match=match):
        sfca_e(comp, Q=2, **kwargs)


def test_params_come_from_the_compile(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    params = sfca_e(comp, Q=2).params
    assert params == {
        "D_max": D_MAX,
        "Q": 2,
        "tau": TAU,
        "n_modes": 2,
        "n_open": None,
        "n_width_fallback": None,
    }


def test_voronoi_rejects_a_compiled(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(TypeError, match="voronoi does not accept a Compiled"):
        voronoi(comp)


def test_compiled_with_frames_is_rejected(frames, prep):
    _, supply_df, cost_df = frames
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    with pytest.raises(TypeError, match="not both"):
        sfca_e(comp, supply_df, cost_df, Q=2)


def test_compile_f_requires_modes(prep):
    with pytest.raises(ValueError, match="compile_f requires modes"):
        compile_f(prep, modes=[])


def test_a_malformed_sites_mask_is_rejected(prep):
    with pytest.raises(ValueError, match="sites must be a boolean array"):
        compile_f(prep, modes=MAC, sites=np.array([1, 0, 1, 0]))


def test_stats_survive_the_compiled_path(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    groups = ["coverage", "distribution", "inequality", "choice_set"]
    got = sfca_e(comp, Q=2, stats=groups, output=())
    want = sfca_e(prep, modes=MAC, D_max=D_MAX, tau=TAU, Q=2, stats=groups, output=())
    assert got.stats.keys() == want.stats.keys()
    for key in got.stats:
        np.testing.assert_allclose(got.stats[key], want.stats[key], rtol=1e-12, err_msg=key)


def test_sweep_accepts_a_compiled(prep):
    """A Q-only grid needs one compile, not one per combination."""
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    rows = sweep(comp, "sfca_e", grid={"Q": [1, 2, 3]}, stats=["coverage"])
    assert rows["Q"].tolist() == [1, 2, 3]
    assert (rows["n_modes"] == 2).all()

    want = sweep(
        prep, "sfca_e", modes=MAC, D_max=D_MAX, tau=TAU, grid={"Q": [1, 2, 3]}, stats=["coverage"]
    )
    np.testing.assert_allclose(rows["demand_capture_rate"], want["demand_capture_rate"])


def test_compiled_reports_its_shape(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    assert comp.n_sites == len(SITES)
    assert comp.n_pairs == comp.f.size == comp.demand_code.size == comp.supply_code.size
    assert "Compiled(" in repr(comp)
