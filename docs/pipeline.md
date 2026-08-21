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
| `compile_f()` | `modes`, `D_max`, `tau`, and optionally `sites` and `width` | the **network** varies |
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

### Bounding the candidate list: `width=`

A model call reads every compiled row to find each node's top `Q` open options. `width`
bounds that read instead:

```python
comp = im.compile_f(prep, modes=modes, D_max=45, tau=0.01, width=20)
E_j = im.sfca_e(comp, Q=5, open_mask=mask, bare=True)
```

**No rows are dropped.** The `Compiled` still holds every pair surviving `sites`, `D_max`
and `tau`, and its memory is unchanged. What `width` adds is `k_i = min(len_i, width)`,
the prefix a call reads first, and `truncated_i`, whether that prefix hides anything.

Reading only a prefix is exact because the rows in a segment are already in a fixed total
order — `f_multi` descending, ties by the incoming `(cost_default, supply_code)` — and
`open_mask` is a per-row predicate. So "the first `Q` open rows of the segment" is a well
defined set however you go looking for it, and exactly three things can happen:

| the prefix holds | what follows |
|---|---|
| `≥ Q` open rows | the `Q`-th open row is inside it — the same rows a full read selects |
| `< Q` open, nothing hidden (`k_i == len_i`) | the prefix **is** the segment |
| `< Q` open, rows hidden | a **violator**: its hidden tail is re-read and spliced in |

One retry is always enough, because a node read in full is no longer truncated. A tail row
sorts after its own segment's prefix and before the next segment, so the spliced result is
the same rows in the same order a full read would have produced, and `add.reduceat`,
`maximum.reduceat` and `bincount` accumulate identically. The equivalence is bit-for-bit,
not within a tolerance, and `tests/test_width.py` asserts it with `assert_array_equal`
over every non-empty subset of the site set, both sides of `Q`, and all three mode paths.

`width` requires a finite `Q` at call time — an unbounded choice set cannot be truncated —
so `catchment()`, and `ifca()` without `Q`, raise rather than quietly ignoring it.

#### When it pays

The read shrinks from `O(n_pairs)` to `O(n_demand + Σ k_i)`, but a node needs `Q` of its
top `width` to be open, so the useful window is

```
Q / density  <<  width  <<  candidates per node
```

Below that window most truncated nodes are violators and the bounded pass becomes work
done on top of the full one; above it there is nothing left to truncate. Measured on a
synthetic 4M-pair matrix (20k demand nodes, ~74 candidates each after `D_max` and `tau`,
1000 sites, `Q = 5`, `width = 20`), per call against the same compile without a width:

| open density | fallbacks | per call |
|---|---|---|
| 0.20 | 12,605 | 1.68× — **slower** |
| 0.35 | 2,218 | 0.99× |
| 0.50 | 85 | 0.71× |
| 0.80 | 0 | 0.55× |
| 1.00 | 0 | 0.45× |

`params["n_width_fallback"]` is the instrument: it counts the nodes re-read on that call.
Persistently far from zero means `width` is too small for the density being explored —
raise it, or drop the width. This makes `width` a tool for **dense** open sets, such as
local search around an incumbent network, and the wrong tool for greedy construction
outward from an empty one, whose early iterations have almost nothing open.

`benchmarks/optimisation_loop.py` is the script those numbers come from.

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
