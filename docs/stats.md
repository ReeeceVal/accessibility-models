# Stats

Statistics are requested by **group name** and returned as one flat `dict[str, float]`.
A group that is not asked for is not computed.

```python
res = im.sfca(prep, modes=modes, D_max=45, Q=5, tau=0.01,
              stats=["coverage", "inequality"])
res.stats["demand_capture_rate"]
```

`res.stats` is `None` unless `stats=[...]` was passed. Requesting stats forces `E_j` and
`A_i` to be computed even when `output` skips their frames, so `output=()` with
`stats=[...]` is a valid and cheap combination for a sweep.

This is the complete list. Anything richer — correlation against realised counts, dedup,
persistence — is left to the caller.

## `coverage`

| Key | Definition |
|---|---|
| `n_demand` | rows in `demand_df` |
| `n_supply` | rows in `supply_df` |
| `n_pairs_input` | rows in `cost_df` |
| `n_pairs_used` | pairs surviving every filter (pipeline stage 6 output) |
| `total_demand` | `Σ_i P_i` |
| `total_capacity` | `Σ_j S_j`; `NaN` when `supply_df` has no `capacity` |
| `sum_E_j` | `Σ_j E_j` |
| `demand_capture_rate` | `sum_E_j / total_demand` |
| `n_supply_zero_exposure` | count of `j` with `E_j == 0` |
| `n_demand_zero_access` | count of `i` with `A_i == 0` |

`demand_capture_rate` is only meaningful where `E_j` is a **demand count**: Catchment
without impedance, Voronoi, 3SFCA, and MAC-3SFCA-E. Add a decay to Catchment and it is
deflated by impedance; under `ifca()` `E_j` is a crowdedness × capacity product, not a
headcount. Nothing stops you reading it there; it just does not mean what the name
suggests.

Under `sfca_e()` it has a direct reading: since `Σ_j G_ij = 1`, the rate is exactly the
demand-weighted mean participation, `Σ_i P_i Φ_i / Σ_i P_i`.

For Voronoi, `n_pairs_used` counts **assigned nodes**, not candidate pairs — the family's
surviving pair set is one row per node.

## `distribution`

`mean`, `median`, `p10`, `p90`, `std`, `min`, `max` for each of `A_i` and `E_j`, giving 14
keys named `mean_A_i`, `p90_E_j`, and so on. Percentiles are numpy's linear interpolation,
so on a small network `p10` need not equal any observed value.

| Key | Definition |
|---|---|
| `weighted_mean_A_i` | `Σ_i P_i A_i / Σ_i P_i` — demand-weighted, the reportable one |

The unweighted `mean_A_i` treats a node of 12 people and a node of 12 000 identically.
Report `weighted_mean_A_i`.

## `inequality`

| Key | Definition |
|---|---|
| `gini_A_i` | demand-weighted Gini of `A_i` over demand nodes |
| `gini_E_j` | unweighted Gini of `E_j` over supply points |
| `p90_p10_ratio_A_i` | `p90(A_i) / p10(A_i)`; `inf` when `p10 == 0` |

The Gini is computed from the sorted Lorenz curve,

$$
G = 1 - \frac{\sum_k w_k (S_{k-1} + S_k)}{S_n \sum_k w_k},
\qquad S_k = \sum_{l \le k} w_l x_l ,
$$

with values sorted ascending. Weights are frequency weights: weighting `[0, 1]` by
`[3, 1]` gives exactly the unweighted Gini of `[0, 0, 0, 1]`. For `1..n` unweighted it
reproduces the closed form `(n − 1) / 3n`, which is the test.

It returns `NaN` rather than `0.0` when the total is zero — no distribution, no
inequality.

`im.stats.gini(values, weights=None)` is importable on its own.

## `choice_set`

**Only where a bounded choice set exists** — `sfca()`, `sfca_e()`, or `ifca()` with `Q`
set. Requesting it from `catchment()` or `voronoi()`, or from `ifca()` without `Q`, raises
`ValueError`: there is no choice set to describe, and returning `NaN` would hide the
mistake.

`sfca_e()` shares 3SFCA's `G_ij` exactly, so this group reports identical values for the
two families at the same parameters.

| Key | Definition |
|---|---|
| `mean_N_eff_i`, `median_N_eff_i` | `N_eff_i = 1 / Σ_j G_ij²` — the effective number of facilities, over nodes with a non-empty choice set |
| `mean_choice_set_size`, `median_choice_set_size` | realised `\|C_Q(i)\|` after all filters, over **all** nodes including empty ones |
| `frac_demand_q_binding` | share of nodes with `\|C_Q(i)\| == Q` |

`N_eff_i` is the inverse Herfindahl of the selection probabilities: a node splitting evenly
across 2 sites scores 2.0; a node sending 99% to one site scores just above 1.0. It answers
"how many facilities does this node *really* use", which `|C_Q(i)|` does not.

`frac_demand_q_binding` says whether `Q` is the active constraint. Near 1.0, `Q` is doing
the work; near 0.0, `D_max` and `τ` are already cutting harder than `Q` and raising `Q`
would change nothing.
