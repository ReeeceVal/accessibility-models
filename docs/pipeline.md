# The pipeline

Every family runs the same first stages, in this order, always:

| # | Stage | What it does |
|---|---|---|
| 0 | `open_mask` | drop pairs whose supply point is closed — omitted, every site is open |
| 1 | `C_D(i)` | keep pairs with `cost_default <= D_max` |
| 2 | `f_m` | per mode, `f_m = decay_m(kappa_m * cost_m)`; `NaN` cost ⇒ mode unavailable |
| 3 | renormalise | redistribute unavailable modes' shares over the available ones — see [modes](modes.md) |
| 4 | `f_multi` | `f_multi = Σ_m π_m f_m` |
| 5 | `τ` | keep pairs with `f_multi >= tau` |
| 6 | `C_Q(i)` | keep the top `Q` pairs per `i` by `f_multi` descending |
| 7 | family arithmetic | see the family pages |

`E_j` and `A_i` are always computed from the **same** filtered pair set.

## Stage 0: the open set

`open_mask` is a boolean array, one flag per row of `supply_df`. It runs **first**, before
impedance and before the rank, which is the whole of its semantics: a closed site is
indistinguishable from one that was never in `cost_df` at all. Results are therefore
identical — bit-for-bit, not merely close — to re-running `prepare()` on a `cost_df`
restricted to the open sites, because `f_multi(i, j)` depends only on the pair and the
impedance parameters, never on which sites are open.

The consequence that matters is at stage 6: **`C_Q(i)` is the top `Q` open options, not
the open members of the top `Q` overall.** If a node's four best sites rank 1, 2, 4, 7
among the open set, `Q = 4` takes all four. Ranking first and masking afterwards would
leave that node with two options instead of four, and would let a node whose nearby sites
all closed drop out of the model entirely.

Output frames keep their full length, with `E_j = 0` and `n_demand_j = 0` at closed sites,
so results stay aligned across candidate networks. `params["n_open"]` records how many
sites were open.

This is the site-selection lever. For [MAC-3SFCA-E](families/sfca-e.md), `Σ_j E_j` is
monotone in it — opening a site can never lower the total — which is what makes the family
usable as an optimisation objective.

## `compile_f()`: hoisting stages 0–5

`f_multi(i, j)` depends only on the pair and the impedance parameters, **never on which
sites are open**. So when the network is what varies and the parameters are fixed, stages
0–5 can be computed once:

```python
comp = im.compile_f(prep, modes=modes, D_max=45, tau=0.01)
for mask in candidate_networks:               # only stages 0, 6 and 7 run here
    res = im.sfca_e(comp, Q=5, open_mask=mask)
```

The three stages form a pipeline you can enter at any point:

| Stage | Fixes | Reuse it when |
|---|---|---|
| `prepare()` | the input frames | always — nothing about it depends on parameters |
| `compile_f()` | `modes`, `D_max`, `tau`, and optionally `sites` | the **network** varies |
| the model | `Q`, `open_mask` | — |

When the *parameters* vary and the network is fixed, invert it: pass the fixed network to
`compile_f(sites=...)` instead, so the excluded pairs cost nothing to compile, and call
`compile_f()` once per parameter combination. Both masks take one flag per row of
`supply_df`; they differ only in lifetime, and an `open_mask` must be a subset of `sites`.

Accepted by `catchment()`, `ifca()`, `sfca()` and `sfca_e()`. **Not** by `voronoi()`,
which has no impedance and assigns on `cost_default` order rather than `f_multi` order —
it takes `open_mask` alone.

`modes`, `D_max` and `tau` are baked in, so passing them alongside a `Compiled` raises
rather than being silently ignored.

### Why this also fixes the slow path

Compiling stores rows `f_multi`-descending within each segment. Masking preserves relative
order, so `C_Q(i)` becomes a prefix truncation **whatever the mode set** — the
`lexsort` that a mode on its own cost column would otherwise cost on every call is paid
once, at compile time. The general path ends up as cheap per call as the fast path.

The reordering is also the one place results can move: on the kappa-only path the compiled
rows keep the prepared order, so results are bit-for-bit identical to the uncompiled call,
while a reordered segment sums its floats in a different order and can differ in the last
ulp.

## Which stages a family runs

A stage runs when the parameter that defines it is given. Two families never run some of
them at all:

| Family | 1 `D_max` | 2–4 `f_multi` | 5 `τ` | 6 `Q` |
|---|---|---|---|---|
| [Catchment](families/catchment.md) | ✓ | with `modes` carrying a decay | with impedance | — |
| [Voronoi](families/voronoi.md) | ✓ | reach gate only | — | — |
| [iFCA](families/ifca.md) | ✓ | ✓ | ✓ | optional |
| [3SFCA](families/sfca.md) | ✓ | ✓ | ✓ | required |
| [MAC-3SFCA-E](families/sfca-e.md) | ✓ | ✓ | ✓ | required |

MAC-3SFCA-E runs stage 6 like 3SFCA, but its `Φ_i` reads the largest `f_multi` in the
choice set — which is the top-ranked one — so the **total** exposure it produces is the
same for every `Q ≥ 1`. `Q` still decides how that total is split across sites.

## Filter, then rank

Stage 6 ranks **survivors**, not the raw pair table. Ranking first would let a node's
top-`Q` contain sites beyond `D_max`, leaving it with fewer than `Q` effective options.
Here `Q` always means `Q` real options, or as many as exist.

## Ranking and ties

Rank is by `f_multi` descending. Ties keep the underlying sort order — ascending
`cost_default`, then ascending supply code — so results are deterministic and independent
of input row order.

### The prefix-truncation fast path

`prepare()` sorts pairs by `(demand_code, cost_default, supply_code)`, so each node owns
one contiguous, cost-ascending segment. When **every mode travels on `cost_default`** and
each `decay` is non-increasing, `f_multi` is a monotone function of `cost_default`, which
means that segment order *already is* the `f_multi`-descending rank order.

Stages 1, 5 and 6 then reduce to **prefix truncations** of a segment. Top-`Q` costs one
`arange` subtraction rather than a sort, and it costs the same whatever `Q` you pass. This
is the dominant case.

When a mode carries **its own cost column** (walking, say), `f_multi` is no longer monotone
in `cost_default` and top-`Q` needs one `lexsort` over the surviving pairs per call.

### The `rank` column

`cost_df` may carry a precomputed `rank`. The package does not need it: on the fast path,
position-within-segment *is* the rank, at zero cost. Where some mode carries its own cost
column a supplied `rank` is **ignored with a warning**, because rank-by-cost is not
rank-by-`f_multi` there and using it would silently give the wrong choice set.

Passing `rank` is harmless. It is simply never faster than not passing it.

## Zero impedance

A pair whose modes are all unavailable gets `f_multi = 0`, which fails any `tau > 0`.
With `tau = 0` such pairs survive stage 5 but contribute nothing to any sum, and they do
count toward `|C_Q(i)|`. If that matters to you, pass a small positive `tau`.
