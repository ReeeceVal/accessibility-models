# Explain

Every model function sums per-pair terms into `A_i` and `E_j`, then discards the
individual values -- that is what keeps a call over 10-100M pairs cheap. `explain()`
recovers them, scoped to one demand node (or a short list of them):

```python
im.explain(prep, family="sfca", demand_id="d1", modes=modes, D_max=45, tau=0.01, Q=5)
```

```
  demand_id supply_id  cost_default   f_multi    weight
0        d1        s1           5.0  0.969233  0.615088
1        d1        s2          20.0  0.606531  0.384912
```

This is the "click a demand node, see the ranked list of sites it reaches" panel a UI
needs. Only the named node(s)' own pairs are ever read -- cost is independent of network
size, unlike every other entry point in this package.

## Signature

```python
im.explain(
    prep,                 # Prepared or Compiled
    *,
    family,                # "catchment" | "ifca" | "sfca" | "sfca_e" | "voronoi"
    demand_id,             # one id, or a short list of them
    modes=None,
    D_max=np.inf,
    tau=0.0,
    Q=None,
    open_mask=None,
) -> pd.DataFrame
```

`family`, `demand_id` and `Q` aside, every parameter means exactly what it means on the
matching model function, and the same rules apply: a `Compiled` bakes in `modes`, `D_max`
and `tau`, `open_mask` must be a subset of `compile_f(sites=...)`, `sfca`/`sfca_e` require
a finite `Q >= 1`, `catchment` and `voronoi` do not accept `Q` at all, and `voronoi` does
not accept a `Compiled` or `tau`.

## Return value

One row per pair surviving `D_max` (and `tau`, `C_Q(i)` where the family has them),
ranked by `f_multi` descending -- `cost_default` ascending for `voronoi`, which has no
impedance:

| Column | Meaning |
|---|---|
| `demand_id` | which of the requested node(s) this row belongs to |
| `supply_id` | the site |
| `cost_default` | `d_ij` |
| `f_multi` | combined impedance, `1.0` when there is no decay |
| `weight` | the per-pair term the family computes before summing it away |

`weight` is family-specific -- it is a different formula each time, matching each
family's own notation:

| Family | `weight` | Sums to |
|---|---|---|
| `catchment` | `P_i * f_ij` | `E_j`, by site |
| `ifca` | `r_i * f_ij` | `C_j` (multiply by `S_j` for `E_j`), by site |
| `sfca`, `sfca_e` | `G_ij` | `1.0`, within one node's choice set |
| `voronoi` | `1.0` on the assigned pair, `0.0` elsewhere | `P_i * weight` gives `E_j`, by site |

A node reaching nobody returns an empty frame with the same five columns.

## Scope

`demand_id` accepts a scalar or a sequence; results for several nodes concatenate in the
order given. There is deliberately no bulk mode -- an unscoped per-pair dump is exactly
what this package's [scale target](index.md#three-ideas) rules out. For more than a
handful of nodes at once, call the matching model function instead and read `A_i` and
`E_j`.
