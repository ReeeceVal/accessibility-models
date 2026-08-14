# Outputs

Every model function returns a `Result`.

```python
res = im.sfca(prep, modes=modes, D_max=45, Q=5, tau=0.01)

res.params      # {"D_max": 45, "Q": 5, "tau": 0.01, "n_modes": 2}
res.supply      # DataFrame, one row per supply point
res.demand      # DataFrame, one row per demand node
res.stats       # None unless stats=[...] was requested
```

`output=("demand", "supply")` by default; drop either name to skip assembling that frame.
The skipped attribute is `None`, and the work behind it is not done at all — useful in a
sweep that only reports supply-side statistics.

Model functions emit no warnings of their own, with one exception: a `rank` column that
cannot be trusted is [reported and ignored](pipeline.md#the-rank-column).

## Column dictionary

Every column the package can emit.

| Column | Frame | dtype | Meaning | Present when |
|---|---|---|---|---|
| `demand_id` | demand | as input | node identifier | always |
| `demand` | demand | float64 | `P_i` | always |
| `A_i` | demand | float64 | accessibility of node `i` | `"demand"` in `output` |
| `SPAR` | demand | float64 | `A_i / mean(A_i)` — spatial access ratio | with `A_i` |
| `n_supply_i` | demand | int64 | surviving pairs for `i`, i.e. realised `\|C(i)\|` | with `A_i` |
| `r_i` | demand | float64 | demand-to-supply ratio, `P_i / Σ_j S_j f_ij`; `inf` where the denominator is 0 | `ifca()` only |
| `supply_id` | supply | as input | site identifier | always |
| `capacity` | supply | float64 | `S_j` | when `supply_df` has it |
| `E_j` | supply | float64 | expected exposure — **the primary output** | `"supply"` in `output` |
| `R_j` | supply | float64 | `S_j / E_j`, a diagnostic only; `NaN` where `E_j == 0` or capacity is absent | with `E_j` |
| `n_demand_j` | supply | int64 | surviving pairs naming `j` | with `E_j` |
| *anything else* | matching frame | unchanged | passthrough | always |

`A_i` is defined per family — cumulative capacity for Catchment, `S_{j*}/E_{j*}` for
Voronoi, `1/r_i` for iFCA, `Σ_j R_j G_ij f_ij` for 3SFCA. See the family pages.

### Passthrough

Any column on `demand_df` or `supply_df` that the package does not compute is copied to
the matching output frame **unmodified and in input row order**. Column order is:
identifier, then `demand` / `capacity`, then the computed columns, then everything else in
its original order.

Columns on `cost_df` are not passed through — there is no per-pair output frame.

## Zero and empty conventions

| Situation | Result |
|---|---|
| site reached by nobody | `E_j = 0`, `R_j = NaN`, `n_demand_j = 0` |
| node reaching nobody | `A_i = 0`, `n_supply_i = 0` |
| `mean(A_i) == 0` | `SPAR = NaN` everywhere |
| `ifca()` node with `Σ_j S_j f_ij = 0` | `r_i = inf`, `A_i = 0`, contributes nothing to any `C_j` |
| `capacity` column absent | `R_j = NaN`; requesting `A_i` raises `ValueError` |

## `params`

`params` is the full description of a call — it says which optional terms were active, so
nothing about the configuration has to be remembered separately. Always the same four
keys, so sweep rows line up regardless of family; a key the family does not use is `None`:

| Key | `catchment` | `voronoi` | `ifca` | `sfca` |
|---|---|---|---|---|
| `D_max` | ✓ | ✓ | ✓ | ✓ |
| `Q` | `None` | `None` | ✓ or `None` | ✓ (required) |
| `tau` | ✓, `None` without impedance | `None` | ✓ | ✓ |
| `n_modes` | ✓ | ✓ | ✓ | ✓ |
