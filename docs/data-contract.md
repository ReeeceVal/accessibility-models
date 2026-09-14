# Data contract

Every model takes three pandas DataFrames. Extra columns are **passed through untouched**
to the matching output frame, so you can carry province codes, site types or anything else
along for free.

`prepare()` validates and factorises them once; after that the package holds only numpy
arrays.

## `demand_df` — one row per demand node

| Column | Type | Rule |
|---|---|---|
| `demand_id` | str / int | unique |
| `demand` | float | `> 0`. This is `P_i`. |
| *share columns* | float | one per mode, within `[0, 1]`, **summing to 1.0 across modes** (±1e-6) |
| … | any | passed through to `res.demand` |

## `supply_df` — one row per supply point

| Column | Type | Rule |
|---|---|---|
| `supply_id` | str / int | unique |
| `capacity` | float | `>= 0`. This is `S_j`. Optional — see below. |
| … | any | passed through to `res.supply` |

`capacity` is **optional** for `E_j` in the Catchment, Voronoi and 3SFCA families, whose
exposure sums carry no `S_j`. It is **required** by `ifca()`, where `S_j` sits in the
denominator, and by `A_i` in every family. Omit it and ask for `A_i` and you get a
`ValueError`, not a silent zero.

## `cost_df` — one row per reachable (i, j) pair

| Column | Type | Rule |
|---|---|---|
| `demand_id` | str / int | must exist in `demand_df` |
| `supply_id` | str / int | must exist in `supply_df` |
| `cost_default` | float | `>= 0`, no NaN. The reachability cost `d_ij`. |
| *mode cost columns* | float | optional, one per explicit-cost mode. **NaN means no route.** |
| `rank` | int | optional; see [the pipeline](pipeline.md#the-rank-column) |

**Pairs absent from `cost_df` are unreachable.** There is no dense matrix anywhere in the
package — a missing row and an infinite cost mean the same thing.

`cost_default` is the *reachability* cost: `D_max` is applied to it and to nothing else,
and it is the cost the Voronoi assignment minimises. Per-mode costs are what the impedance
functions see.

## Identifiers

IDs are held as `str` where they are `str`. Zero-padded codes carry leading zeros that
numeric parsing silently drops, so never let such an ID become an `int64`.

`prepare()` factorises identifiers to `int32` codes immediately, so the string columns only
exist at the boundary. Codes follow input row order: `demand_code == i` refers to
`demand_df.iloc[i]`.

## What `prepare()` materialises

* every **numeric** column of `cost_df` (except `rank`) becomes a `float32` pair array —
  these are the candidate mode-cost columns;
* every **numeric** column of `demand_df` becomes a `float64` node array — these are the
  candidate share columns.

So keep `cost_df` to identifiers and costs. A 100M-row payload column costs 400 MB of RAM
whether or not any mode refers to it. Non-numeric passthrough columns are free: they stay
on the original frame and are only touched during output assembly.

## Validation

```python
im.validate_inputs(demand_df, supply_df, cost_df, modes=modes)   # standalone
im.prepare(demand_df, supply_df, cost_df, modes=modes)           # runs it for you
im.prepare(demand_df, supply_df, cost_df, validate=False)        # trusted repeat call
```

The rules, each raising `ValueError`:

* required columns present on each frame;
* `demand_id` and `supply_id` unique;
* `cost_df.demand_id` and `cost_df.supply_id` both resolve into their frames;
* no duplicate `(demand_id, supply_id)` pairs;
* `demand > 0`, non-null;
* `capacity >= 0`, non-null (when the column exists);
* `cost_default >= 0`, non-null;
* with `modes` given — every share column present and within `[0, 1]`, the shares summing
  to 1.0 across modes to within `1e-6`, and every mode cost column present on `cost_df`.

Validation runs **once**, inside `prepare()`. Model functions perform no checks at all;
that is what keeps the hot path free of pandas.

## Scale

`prepare()` is designed for ~1M demand nodes × ~1k supply points, so 10–100M pairs. It is
also where the cost is: validating, factorising and sorting happen once, and every model
call afterwards is pure numpy over the same arrays. Build **one** `Prepared` and reuse it.

Per-pair arrays held on it, at 100M pairs:

| Field | dtype | Size |
|---|---|---|
| `demand_code`, `supply_code` | int32 | 400 MB each |
| `cost` (`cost_default`) | float32 | 400 MB |
| each extra mode cost column | float32 | 400 MB |
| `seg_start` | int64 | 8 MB |
| `P`, share columns | float64 | 8 MB each |

So roughly 1.6 GB resident for two modes on `cost_default`, plus a transient `int64`
permutation (~800 MB) during the sort and one `float64` `f_multi` array while a model runs.
Budget ~4 GB of headroom at that size. Costs are `float32` and accumulators `float64`:
travel times in minutes have nowhere near seven significant digits, and halving the biggest
arrays is worth more than precision nobody has.

**The caller's `cost_df` is the real risk.** 100M rows with `object`-dtype string IDs is
6 GB before `prepare()` ever sees it. Pass identifiers as `category` dtype or pre-encoded
integers, and keep `cost_df` to identifiers and cost columns — every numeric column there
becomes a pair array whether or not a mode refers to it.

Two more levers: pass `validate=False` once the same frames have been validated, and cut
`D_max` before anything else, since it is applied first and every later stage works on what
survives it.
