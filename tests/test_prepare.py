"""Stage 1 — segment primitives against pandas, and every rejection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from interaction_models import _core, prepare, validate_inputs


@pytest.fixture
def big():
    """10k-row segmented fixture with a deliberately empty group."""
    rng = np.random.default_rng(0)
    n_groups, n_rows = 500, 10_000
    codes = np.sort(rng.integers(1, n_groups, size=n_rows)).astype(np.int32)  # group 0 empty
    values = rng.random(n_rows)
    return codes, values, n_groups


def test_segment_sum_matches_pandas(big):
    codes, values, n_groups = big
    seg = _core.segment_starts(codes, n_groups)
    expected = (
        pd.Series(values)
        .groupby(pd.Series(codes))
        .sum()
        .reindex(range(n_groups), fill_value=0.0)
        .to_numpy()
    )
    np.testing.assert_allclose(_core.segment_sum(values, seg), expected)


def test_segment_lengths_and_position_match_pandas(big):
    codes, values, n_groups = big
    seg = _core.segment_starts(codes, n_groups)
    counts = pd.Series(codes).value_counts().reindex(range(n_groups), fill_value=0).to_numpy()
    np.testing.assert_array_equal(_core.segment_lengths(seg), counts)
    np.testing.assert_array_equal(
        _core.segment_position(seg, codes.size),
        pd.Series(values).groupby(pd.Series(codes)).cumcount().to_numpy(),
    )


def test_segment_rank_desc_matches_pandas(big):
    codes, values, n_groups = big
    seg = _core.segment_starts(codes, n_groups)
    expected = (
        pd.Series(values)
        .groupby(pd.Series(codes))
        .rank(method="first", ascending=False)
        .to_numpy()
        - 1
    )
    np.testing.assert_array_equal(_core.segment_rank_desc(values, codes, seg), expected)


def test_group_sum_matches_pandas(big):
    codes, values, n_groups = big
    shuffled = np.random.default_rng(1).permutation(codes.size)
    expected = (
        pd.Series(values)
        .groupby(pd.Series(codes))
        .sum()
        .reindex(range(n_groups), fill_value=0.0)
        .to_numpy()
    )
    np.testing.assert_allclose(
        _core.group_sum(codes[shuffled], values[shuffled], n_groups), expected
    )


def test_segment_first_rows_is_argmin_given_the_sort(prep):
    groups, rows = _core.segment_first_rows(prep.seg_start)
    for g, r in zip(groups, rows, strict=True):
        seg = slice(prep.seg_start[g], prep.seg_start[g + 1])
        assert prep.cost[r] == prep.cost[seg].min()


# --- prepare() -----------------------------------------------------------------


def test_prepare_sorts_pairs_by_demand_then_cost_then_supply(prep):
    keys = list(zip(prep.demand_code, prep.cost, prep.supply_code, strict=True))
    assert keys == sorted(keys)


def test_prepare_exposes_expected_arrays(prep):
    assert (prep.n_demand, prep.n_supply, prep.n_pairs) == (6, 4, 18)
    assert prep.demand_code.dtype == np.int32
    assert prep.cost.dtype == np.float32
    assert prep.P.dtype == np.float64
    assert {"cost_default", "w_cost"} <= set(prep.cost_cols)
    assert {"pv_share", "mbt_share"} <= set(prep.share_cols)
    np.testing.assert_array_equal(prep.S, [10.0, 5.0, 20.0, 1.0])


def test_prepare_without_capacity_column(frames):
    demand_df, supply_df, cost_df = frames
    assert prepare(demand_df, supply_df.drop(columns="capacity"), cost_df).S is None


def test_prepare_accepts_string_ids_with_leading_zeros():
    demand_df = pd.DataFrame({"demand_id": ["001", "002"], "demand": [1.0, 2.0]})
    supply_df = pd.DataFrame({"supply_id": ["0010123456789"], "capacity": [1.0]})
    cost_df = pd.DataFrame(
        {
            "demand_id": ["001", "002"],
            "supply_id": ["0010123456789"] * 2,
            "cost_default": [1.0, 2.0],
        }
    )
    prepared = prepare(demand_df, supply_df, cost_df)
    np.testing.assert_array_equal(prepared.supply_code, [0, 0])


# --- validation ----------------------------------------------------------------


def _mutate(frames, which, fn):
    demand_df, supply_df, cost_df = (f.copy() for f in frames)
    frame = {"demand": demand_df, "supply": supply_df, "cost": cost_df}[which]
    fn(frame)
    return demand_df, supply_df, cost_df


@pytest.mark.parametrize(
    ("which", "mutate", "match"),
    [
        ("demand", lambda d: d.__setitem__("demand_id", ["d1"] * 6), "demand_id must be unique"),
        ("supply", lambda d: d.__setitem__("supply_id", ["s1"] * 4), "supply_id must be unique"),
        ("demand", lambda d: d.__setitem__("demand", [0.0] * 6), "demand must be non-null"),
        ("supply", lambda d: d.__setitem__("capacity", [-1.0] * 4), "capacity must be non-null"),
        ("cost", lambda d: d.__setitem__("cost_default", -1.0), "cost_default must be non-null"),
        ("cost", lambda d: d.__setitem__("cost_default", np.nan), "cost_default must be non-null"),
        ("cost", lambda d: d.__setitem__("demand_id", "nope"), "demand_id has 18 value"),
        ("cost", lambda d: d.__setitem__("supply_id", "nope"), "supply_id has 18 value"),
        ("demand", lambda d: d.drop(columns="demand", inplace=True), "missing required column"),
        ("cost", lambda d: d.drop(columns="cost_default", inplace=True), "missing required column"),
    ],
)
def test_validation_rejects(frames, which, mutate, match):
    with pytest.raises(ValueError, match=match):
        validate_inputs(*_mutate(frames, which, mutate))


def test_validation_rejects_duplicate_pairs(frames):
    demand_df, supply_df, cost_df = frames
    with pytest.raises(ValueError, match="duplicate"):
        validate_inputs(demand_df, supply_df, pd.concat([cost_df, cost_df.head(1)]))


def test_validate_false_skips_checks(frames):
    demand_df, supply_df, cost_df = _mutate(frames, "demand", lambda d: d.__setitem__("demand", -1.0))
    assert prepare(demand_df, supply_df, cost_df, validate=False).n_demand == 6
