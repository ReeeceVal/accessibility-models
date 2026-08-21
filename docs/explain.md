# Explain

Every model function sums per-pair terms into `A_i` and `E_j`, then discards the
individual values -- that is what keeps a call over 10-100M pairs cheap. `explain()`
recovers them, scoped to exactly one of a demand node or a supply point (or a short list
of either):

```python
im.explain(prep, family="sfca", demand_id="d1", modes=modes, D_max=45, tau=0.01, Q=5)
```

```
  demand_id supply_id  cost_default   f_multi    weight
0        d1        s1           5.0  0.969233  0.615088
1        d1        s2          20.0  0.606531  0.384912
```

```python
im.explain(prep, family="sfca", supply_id="s1", modes=modes, D_max=45, tau=0.01, Q=5)
```

```
  demand_id supply_id  cost_default   f_multi    weight
0        d1        s1           5.0  0.969233  0.615088
1        d5        s1          22.0  0.454644  0.500000
2        d2        s1          15.0  0.539746  0.420278
```

`demand_id` is the "click a demand node, see the ranked list of sites it reaches" view an interactive
drill-down needs; `supply_id` is "click a site, see which demand nodes it pulls, and how
strongly" -- the two views a network decision needs on either side of a pair.

## Signature

```python
im.explain(
    prep,                 # Prepared or Compiled
    *,
    family,                # "catchment" | "ifca" | "sfca" | "sfca_e" | "voronoi"
    demand_id=None,        # one id, or a short list of them
    supply_id=None,        # one id, or a short list of them -- exactly one of the two
    modes=None,
    D_max=np.inf,
    tau=0.0,
    Q=None,
    open_mask=None,
) -> pd.DataFrame
```

`family`, the two id parameters and `Q` aside, every parameter means exactly what it
means on the matching model function, and the same rules apply: a `Compiled` bakes in
`modes`, `D_max` and `tau`, `open_mask` must be a subset of `compile_f(sites=...)`,
`sfca`/`sfca_e` require a finite `Q >= 1`, `catchment` and `voronoi` do not accept `Q` at
all, and `voronoi` does not accept a `Compiled` or `tau`.

## Return value

One row per pair surviving `D_max` (and `tau`, `C_Q(i)` where the family has them):

| Column | Meaning |
|---|---|
| `demand_id` | the demand node |
| `supply_id` | the site |
| `cost_default` | `d_ij` |
| `f_multi` | combined impedance, `1.0` when there is no decay |
| `weight` | the per-pair term the family computes before summing it away |

A `demand_id` query ranks by `f_multi` descending within each node (`cost_default`
ascending for `voronoi`, which has no impedance) -- "which site is most attractive to
this node". A `supply_id` query ranks by `weight` descending, `cost_default` ascending,
across every node that reaches the named site(s) -- "which node is pulled here most
strongly", which for the competitive families is not the same ordering `f_multi` would
give, since each node's own denominator differs.

`weight` is family-specific -- it is a different formula each time, matching each
family's own notation:

| Family | `weight` | Sums to |
|---|---|---|
| `catchment` | `P_i * f_ij` | `E_j`, by site |
| `ifca` | `r_i * f_ij` | `C_j` (multiply by `S_j` for `E_j`), by site |
| `sfca`, `sfca_e` | `G_ij` | `1.0`, within one node's choice set |
| `voronoi` | `1.0` on the assigned pair, `0.0` elsewhere | `P_i * weight` gives `E_j`, by site |

A node (or site) reaching nobody returns an empty frame with the same five columns.

## Scope

`demand_id`/`supply_id` each accept a scalar or a sequence; results for several
nodes/sites concatenate. There is deliberately no bulk mode -- an unscoped per-pair dump
is exactly what this package's [scale target](index.md#three-ideas) rules out. For more
than a handful of nodes or sites at once, call the matching model function instead and
read `A_i` and `E_j`.

The two axes are not symmetric under the hood. `Prepared`/`Compiled` are sorted by
demand only, so a `demand_id` query only ever reads the named node(s)' own segment. A
`supply_id` query first pays a one-shot `O(n_pairs)` scan to find which demand nodes
reach the named site(s); for the competitive families (`ifca`, `sfca`, `sfca_e`) a
pair's `weight` also depends on that node's whole choice set, not just the site asked
about, so each candidate node then runs through the same bounded per-node computation a
`demand_id` query does. Still bounded by that site's own (thousands-scale) reachable
set, never by the network as a whole -- just not free the way the demand axis is.
