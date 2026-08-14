# The pipeline

Every family runs the same first stages, in this order, always:

| # | Stage | What it does |
|---|---|---|
| 1 | `C_D(i)` | keep pairs with `cost_default <= D_max` |
| 2 | `f_m` | per mode, `f_m = decay_m(kappa_m * cost_m)`; `NaN` cost ⇒ mode unavailable |
| 3 | renormalise | redistribute unavailable modes' shares over the available ones — see [modes](modes.md) |
| 4 | `f_multi` | `f_multi = Σ_m π_m f_m` |
| 5 | `τ` | keep pairs with `f_multi >= tau` |
| 6 | `C_Q(i)` | keep the top `Q` pairs per `i` by `f_multi` descending |
| 7 | family arithmetic | see the family pages |

`E_j` and `A_i` are always computed from the **same** filtered pair set.

## Which stages a family runs

A stage runs when the parameter that defines it is given. Two families never run some of
them at all:

| Family | 1 `D_max` | 2–4 `f_multi` | 5 `τ` | 6 `Q` |
|---|---|---|---|---|
| [Catchment](families/catchment.md) | ✓ | with `modes` carrying a decay | with impedance | — |
| [Voronoi](families/voronoi.md) | ✓ | reach gate only | — | — |
| [iFCA](families/ifca.md) | ✓ | ✓ | ✓ | optional |
| [3SFCA](families/sfca.md) | ✓ | ✓ | ✓ | required |

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
