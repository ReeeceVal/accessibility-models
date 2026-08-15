"""Grid expansion, labels and the single-import surface."""

from __future__ import annotations

import pytest
from conftest import BETA, D_MAX, TAU
from test_ifca import MAC, SINGLE

import interaction_models as im


def test_single_import_surface():
    for name in im.__all__:
        assert hasattr(im, name), name
    assert callable(im.sfca) and callable(im.sweep)
    assert callable(im.stats.gini)


def test_every_family_is_reachable_by_name():
    assert set(im.sweep.__globals__["MODELS"]) == {
        "catchment", "voronoi", "ifca", "sfca", "sfca_e",
    }


def test_sfca_e_sweeps_like_any_other_family(prep):
    rows = im.sweep(
        prep, "sfca_e", modes=MAC, D_max=D_MAX, tau=TAU,
        grid={"Q": [1, 2, 3]}, stats=["coverage"],
    )
    assert list(rows["model"]) == ["sfca_e"] * 3
    # Phi_i is invariant to Q, so the total is too — the family's headline property.
    assert rows["sum_E_j"].nunique() == 1


def test_grid_is_a_cartesian_product(prep):
    rows = im.sweep(
        prep,
        "sfca",
        modes=MAC,
        D_max=D_MAX,
        tau=TAU,
        grid={"Q": [1, 2, 3], "kappa": [1.5, 2.0]},
    )
    assert len(rows) == 6
    assert set(rows["Q"]) == {1, 2, 3}
    assert set(rows["kappa"]) == {1.5, 2.0}


def test_labels_land_in_rows(prep):
    rows = im.sweep(
        prep,
        "sfca",
        modes=MAC,
        D_max=D_MAX,
        tau=TAU,
        Q=2,
        grid={"decay": [("gauss800", im.gaussian(BETA)), ("gauss1200", im.gaussian(1200.0))]},
    )
    assert list(rows["decay"]) == ["gauss800", "gauss1200"]


def test_row_schema(prep):
    rows = im.sweep(
        prep,
        "sfca",
        modes=MAC,
        D_max=D_MAX,
        tau=TAU,
        grid={"Q": [2]},
        stats=["coverage"],
    )
    assert list(rows.columns)[:5] == ["model", "Q", "D_max", "tau", "n_modes"]
    assert rows.loc[0, "model"] == "sfca"
    assert rows.loc[0, "n_modes"] == 2
    assert rows.loc[0, "n_demand"] == 6
    assert rows.loc[0, "Q"] == 2


def test_decay_rewrites_every_mode(prep):
    """One grid value rewrites every mode: beta is shared across the split."""
    rows = im.sweep(
        prep,
        "catchment",
        modes=MAC,
        D_max=D_MAX,
        tau=TAU,
        grid={"decay": [im.gaussian(BETA)]},
        stats=["coverage"],
    )
    direct = im.catchment(
        prep,
        modes=[im.Mode(share=m.share, kappa=m.kappa, decay=im.gaussian(BETA)) for m in MAC],
        D_max=D_MAX,
        tau=TAU,
        stats=["coverage"],
        output=(),
    )
    assert rows.loc[0, "sum_E_j"] == pytest.approx(direct.stats["sum_E_j"])


def test_kappa_rewrites_only_the_non_reference_modes(prep):
    rows = im.sweep(
        prep, "catchment", modes=MAC, D_max=D_MAX, tau=TAU,
        grid={"kappa": [1.0]}, stats=["coverage"],
    )
    unimodal = im.sweep(
        prep, "catchment", modes=SINGLE, D_max=D_MAX, tau=TAU,
        grid={"decay": [im.gaussian(BETA)]}, stats=["coverage"],
    )
    # kappa = 1 on the second mode collapses the two-mode form onto the one-mode one.
    assert rows.loc[0, "sum_E_j"] == pytest.approx(unimodal.loc[0, "sum_E_j"])


def test_whole_mode_lists_can_be_swept(prep):
    rows = im.sweep(
        prep,
        "sfca",
        D_max=D_MAX,
        tau=TAU,
        Q=2,
        grid={"modes": [("single", MAC[:1]), ("mac", MAC)]},
    )
    assert list(rows["modes"]) == ["single", "mac"]
    assert list(rows["n_modes"]) == [1, 2]


def test_sweep_does_not_assemble_frames(prep):
    """output=() is forced, so a large sweep never builds per-node frames."""
    rows = im.sweep(prep, "voronoi", D_max=D_MAX, grid={"D_max": [10.0, 45.0]})
    assert list(rows["model"]) == ["voronoi", "voronoi"]
    assert list(rows["D_max"]) == [10.0, 45.0]


def test_unknown_grid_key_raises(prep):
    with pytest.raises(ValueError, match="unknown grid key"):
        im.sweep(prep, "sfca", modes=MAC, Q=2, grid={"beta": [800.0]})


def test_mode_rewriting_needs_a_template(prep):
    with pytest.raises(ValueError, match="pass modes"):
        im.sweep(prep, "catchment", D_max=D_MAX, grid={"decay": [im.gaussian(BETA)]})
