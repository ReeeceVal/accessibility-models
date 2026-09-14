"""``compile_f(width=...)`` against the unbounded compile it has to reproduce.

The width is a promise that only the top ``Q`` of a node's candidate list is ever read.
The fallback is what makes the promise safe: a node whose bounded prefix holds fewer than
``Q`` open options is re-read over its whole segment, so nothing the width hid can change
an answer.

Every assertion here is exact equality, not a tolerance. A bounded call retains a
*subset of the same physically ordered arrays*, gathered in the same order, so the
reductions accumulate identically — the width changes which rows are read, never their
order. The reference is always the ``width=None`` compile rather than the uncompiled
call, which keeps that exactness available even for ``OWN_COL``, where compiling reorders
rows and compiled-vs-uncompiled is only ``allclose``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import BETA, D_MAX, KAPPA, TAU, S, by_id

from interaction_models import Mode, catchment, compile_f, gaussian, ifca, sfca, sfca_e

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
#: Every non-empty subset of the four supply points.
SUBSETS = [
    subset for size in range(1, len(SITES) + 1) for subset in itertools.combinations(SITES, size)
]

#: (label, model, extra kwargs, modes) — every family that can supply a finite Q.
WIDTH_CASES = [
    ("ifca", ifca, {"Q": 2}, MAC),
    ("sfca", sfca, {"Q": 2}, MAC),
    ("sfca-owncol", sfca, {"Q": 2}, OWN_COL),
    ("sfca_e-Q1", sfca_e, {"Q": 1}, MAC),
    ("sfca_e-Q2", sfca_e, {"Q": 2}, MAC),
    ("sfca_e-Q4", sfca_e, {"Q": 4}, MAC),
    ("sfca_e-single", sfca_e, {"Q": 2}, SINGLE),
    ("sfca_e-owncol", sfca_e, {"Q": 2}, OWN_COL),
]


def mask_for(supply_df, keep) -> np.ndarray:
    return supply_df["supply_id"].isin(keep).to_numpy()


def same(got, want, label):
    for frame, column in (("supply", "E_j"), ("demand", "A_i")):
        np.testing.assert_array_equal(
            getattr(got, frame)[column].to_numpy(),
            getattr(want, frame)[column].to_numpy(),
            err_msg=f"{label} {frame}.{column}",
        )


@pytest.mark.parametrize(
    ("label", "model", "extra", "modes"), WIDTH_CASES, ids=[case[0] for case in WIDTH_CASES]
)
@pytest.mark.parametrize("width", [1, 2, 3])
def test_a_bounded_width_reproduces_the_unbounded_compile_under_every_mask(
    prep, supply_df, label, model, extra, modes, width
):
    """The headline: bit-for-bit, over the whole site lattice and both sides of Q."""
    baseline = compile_f(prep, modes=modes, D_max=D_MAX, tau=TAU)
    bounded = compile_f(prep, modes=modes, D_max=D_MAX, tau=TAU, width=width)
    for keep in [None, *SUBSETS]:
        mask = None if keep is None else mask_for(supply_df, keep)
        same(
            model(bounded, open_mask=mask, **extra),
            model(baseline, open_mask=mask, **extra),
            label=f"{label} width={width} keep={keep}",
        )


def test_the_compiled_prefix_lengths_describe_the_segments(prep):
    """d1 and d2 keep 3 pairs after D_max and tau, so only they are truncated at W=2."""
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    assert comp.width == 2
    np.testing.assert_array_equal(np.diff(comp.seg), [3, 3, 2, 2, 2, 0])
    np.testing.assert_array_equal(comp.k, [2, 2, 2, 2, 2, 0])
    np.testing.assert_array_equal(comp.truncated, [True, True, False, False, False, False])
    # Frozen, so a fallback cannot revise them in place.
    assert not comp.k.flags.writeable
    assert not comp.truncated.flags.writeable


def test_a_truncated_node_with_too_few_open_options_falls_back(prep, supply_df):
    """d1 and d2 rank s1 and s2 first, so opening only s3 hides their whole choice set."""
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    mask = mask_for(supply_df, ["s3", "s4"])
    res = sfca_e(comp, Q=2, open_mask=mask)
    assert res.params["n_width_fallback"] == 2
    # The fallback recovers exactly what the Q contract demands: the top Q *open*
    # options, not the open members of the top Q.
    assert by_id(res.demand, "demand_id", "n_supply_i")["d1"] == 1
    assert by_id(res.supply, "supply_id", "E_j")["s3"] > 0


def test_a_partially_open_prefix_still_falls_back(prep, supply_df):
    """One open option inside the prefix is still fewer than Q, so the node is re-read."""
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    baseline = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    mask = mask_for(supply_df, ["s2", "s3", "s4"])
    res = sfca_e(comp, Q=2, open_mask=mask)
    assert res.params["n_width_fallback"] == 2
    same(res, sfca_e(baseline, Q=2, open_mask=mask), label="partially open")


@pytest.mark.parametrize(
    ("Q", "keep", "expected"),
    [
        (2, None, 0),  # unmasked: every prefix holds k = W = Q open options
        (1, None, 0),  # Q below the width can never violate
        (2, ["s1", "s2"], 0),  # both of d1's and d2's top two are open
    ],
)
def test_no_fallback_when_the_prefix_already_holds_q_open_options(
    prep, supply_df, Q, keep, expected
):
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    mask = None if keep is None else mask_for(supply_df, keep)
    assert sfca_e(comp, Q=Q, open_mask=mask).params["n_width_fallback"] == expected


def test_a_q_above_the_width_falls_every_truncated_node_back(prep, supply_df):
    """Q > W makes every truncated node a violator by arithmetic; short-circuit to full."""
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    baseline = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU)
    for keep in [None, *SUBSETS]:
        mask = None if keep is None else mask_for(supply_df, keep)
        res = sfca_e(comp, Q=3, open_mask=mask)
        assert res.params["n_width_fallback"] == 2, keep
        same(res, sfca_e(baseline, Q=3, open_mask=mask), label=f"Q>W keep={keep}")


def test_a_fallback_does_not_mutate_the_compiled(prep, supply_df):
    """A widened k must be a fresh array: in place, it would leak into every later call."""
    comp = compile_f(prep, modes=SINGLE, D_max=D_MAX, tau=TAU, width=2)
    snapshot = comp.k.copy()
    mask = mask_for(supply_df, ["s3", "s4"])
    first = sfca_e(comp, Q=2, open_mask=mask)
    np.testing.assert_array_equal(comp.k, snapshot)
    # A different mask in between, then the original again.
    sfca_e(comp, Q=2, open_mask=mask_for(supply_df, ["s1", "s2"]))
    second = sfca_e(comp, Q=2, open_mask=mask)
    np.testing.assert_array_equal(comp.k, snapshot)
    same(second, first, label="repeat")


def test_width_leaves_every_row_in_the_compiled(prep):
    """The width bounds what a call reads, not what the Compiled stores."""
    baseline = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU)
    bounded = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, width=1)
    assert bounded.n_pairs == baseline.n_pairs


def test_a_width_needs_a_finite_q(prep):
    comp = compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, width=2)
    for call in (lambda: catchment(comp), lambda: ifca(comp), lambda: ifca(comp, Q=np.inf)):
        with pytest.raises(ValueError, match="needs a finite Q"):
            call()


@pytest.mark.parametrize("width", [0, -1, 2.5, np.inf, True, "2"])
def test_width_must_be_a_positive_integer(prep, width):
    with pytest.raises(ValueError, match="width must be a positive integer"):
        compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, width=width)


def test_width_composes_with_a_sites_restriction(prep, supply_df):
    """The width applies to the sites the Compiled was built over, and changes nothing."""
    sites = mask_for(supply_df, ["s1", "s2", "s3"])
    kwargs = {"modes": MAC, "D_max": D_MAX, "tau": TAU, "sites": sites}
    same(
        sfca_e(compile_f(prep, width=2, **kwargs), Q=2),
        sfca_e(compile_f(prep, **kwargs), Q=2),
        label="sites",
    )


def test_width_is_reported_in_the_repr(prep):
    assert "width=2" in repr(compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU, width=2))
    assert "width=None" in repr(compile_f(prep, modes=MAC, D_max=D_MAX, tau=TAU))
