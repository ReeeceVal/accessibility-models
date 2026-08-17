# interaction-models

A lean, environment-agnostic Python package computing spatial interaction and
accessibility models over a demand → supply cost matrix.

Five model families, each a single model whose optional terms switch on with the
parameters that define them:

| Family | What it does |
|---|---|
| **Catchment** | cumulative opportunity, no competition between sites |
| **Voronoi** | winner-take-all assignment to the nearest reachable site |
| **iFCA** | sites compete for demand, weighted by crowdedness |
| **3SFCA** | demand-side selection across a bounded choice set |
| **MAC-3SFCA-E** | 3SFCA with an explicit participation step; monotone under site openings |

Every family emits the same two outputs: `E_j`, the expected exposure of a supply point,
and `A_i`, the accessibility of a demand node.

## Why

Three things this package is built around:

* **`prepare()` once, sweep cheaply.** Identifiers are factorised to `int32` and the pair
  table is sorted once; every subsequent model call is pure numpy over the same arrays.
  Designed for ~1M demand nodes × ~1k supply points, so 10–100M pairs.
* **Parameters, not model names.** Impedance, mode split and a bounded choice set are
  terms you switch on by supplying `decay`, a second `Mode` or `Q`. The resolved
  configuration comes back as `res.params`.
* **No I/O, no orchestration.** Three DataFrames in, two DataFrames out.

## Install

Managed with [uv](https://docs.astral.sh/uv/), Python 3.13:

```powershell
uv sync
```

## Quickstart

```python
import pandas as pd
import interaction_models as im

demand_df = pd.DataFrame({
    "demand_id":  ["d1", "d2", "d3", "d4"],
    "demand":     [1200.0, 800.0, 450.0, 2100.0],
    "pv_share":   [0.35, 0.10, 0.60, 0.25],
    "mbt_share":  [0.65, 0.90, 0.40, 0.75],
    "province":   ["WC", "WC", "GP", "GP"],        # passthrough
})

supply_df = pd.DataFrame({
    "supply_id": ["0010001", "0010002", "0010003"],  # leading zeros stay str
    "capacity":  [2.0, 1.0, 3.0],
})

cost_df = pd.DataFrame({
    "demand_id":    ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d4"],
    "supply_id":    ["0010001", "0010002", "0010001", "0010003",
                     "0010002", "0010003", "0010001", "0010002", "0010003"],
    "cost_default": [8.0, 26.0, 14.0, 41.0, 19.0, 33.0, 5.0, 22.0, 30.0],
})

modes = [
    im.Mode(share="pv_share",  cost="cost_default", kappa=1.0, decay=im.gaussian(800)),
    im.Mode(share="mbt_share", cost="cost_default", kappa=2.0, decay=im.gaussian(800)),
]

prep = im.prepare(demand_df, supply_df, cost_df, modes=modes)
res = im.sfca(prep, modes=modes, D_max=45, Q=2, tau=0.01, stats=["coverage"])

print(res.params)
print(res.supply)
print(res.demand)
print(res.stats["demand_capture_rate"])
```

```
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2, 'n_open': None}
supply_id  capacity         E_j      R_j  n_demand_j
  0010001       2.0 2657.559490 0.000753           3
  0010002       1.0  264.803074 0.003776           3
  0010003       3.0   18.328194 0.163682           2
demand_id  demand      A_i     SPAR  n_supply_i  pv_share  mbt_share province
       d1  1200.0 0.000608 0.256337           2      0.35       0.65       WC
       d2   800.0 0.000363 0.153119           2      0.10       0.90       WC
       d3   450.0 0.007817 3.296836           2      0.60       0.40       GP
       d4  2100.0 0.000696 0.293708           2      0.25       0.75       GP
0.6463056611235294
```

## Varying the network, or varying the parameters

`f_ij` never depends on which sites are open, so hoist whichever stage holds the thing
that is not changing. Sites are addressed by a boolean mask, one flag per `supply_df` row:

```python
mask = supply_df["supply_id"].isin(["0010001", "0010003"]).to_numpy()   # leading zeros stay str
```

**a) Network changes, parameters constant** — optimisation.
`prepare()` → `compile_f(modes, D_max, tau)` → **mask** → `sfca_e(comp, Q, open_mask)`

```python
comp = im.compile_f(prep, modes=modes, D_max=45, tau=0.01)
for mask in candidate_networks:
    E_j = im.sfca_e(comp, Q=2, open_mask=mask).supply["E_j"]
```

**b) Network constant, parameters change** — calibration.
`prepare()` → **mask** → `compile_f(modes, D_max, tau, sites)` → `sfca_e(comp, Q)`

```python
for tuned in parameter_combinations:
    comp = im.compile_f(prep, modes=tuned, D_max=45, tau=0.01, sites=mask)
    E_j = im.sfca_e(comp, Q=2).supply["E_j"]
```

or in one call, `im.sweep(prep, "sfca_e", modes=modes, D_max=45, tau=0.01, Q=2,
open_mask=mask, grid={"kappa": [1.5, 2.0]})`.

`C_Q(i)` is the top `Q` **open** options, and closed sites stay in the output as zero rows.
Both masks are optional, `open_mask` must be a subset of `sites`, and `voronoi()` takes
`open_mask` but not a `Compiled` — see [`docs/pipeline.md`](docs/pipeline.md).

## Formula reference

`P_i` = demand, `S_j` = capacity, `f_ij` = combined impedance, `C_D(i)` = pairs within
`D_max`, `C_Q(i)` = the top-`Q` of those.

| Family | `E_j` | `A_i` |
|---|---|---|
| Catchment | `Σ_i P_i f_ij` | `Σ_j S_j f_ij` |
| Voronoi | `Σ_{i: j=j*(i)} P_i Σ_m π_m 1[κ_m d_ij ≤ D_max]` | `S_{j*} / E_{j*}` |
| iFCA | `S_j Σ_i r_i f_ij`, `r_i = P_i / Σ_j S_j f_ij` | `1 / r_i` |
| 3SFCA | `Σ_i P_i G_ij f_ij`, `G_ij = f_ij / Σ_k f_ik` | `Σ_j R_j G_ij f_ij` |
| MAC-3SFCA-E | `Σ_i P_i Φ_i G_ij`, `Φ_i = max_j f_ij` | `Σ_j R_j G_ij f_ij` |

With no `decay` on any mode, `f_ij = 1` and Catchment reduces to plain counts of demand
and capacity within reach.

MAC-3SFCA-E separates *whether* a node travels (`Φ_i`) from *which* site it picks
(`G_ij`), which 3SFCA conflates. Because `Σ_j G_ij = 1`, its total collapses to
`Σ_j E_j = Σ_i P_i max_j f_ij` — the classical facility-location function, monotone and
submodular in the site set, and therefore usable as an optimisation objective where the
other families' totals are not. It also emits `L_j = E_j / S_j`, the operational load per
unit capacity.

Combined impedance, for any number of modes:

```
f_ij = Σ_m π_m(i) · f_m(κ_m · cost_m(i,j))       (renormalised over available modes)
```

## The pipeline

Every family runs the same first stages, always in this order:

```
0.  open_mask   drop pairs whose supply point is closed
1.  C_D(i)      keep pairs where cost_default <= D_max
2.  f_m         per mode: f_m = decay_m(kappa_m * cost_m);  NaN cost -> unavailable
3.  renormalise shares over available modes only
4.  f_multi     f_multi = sum_m( pi_m * f_m )
5.  tau         keep pairs where f_multi >= tau
6.  C_Q(i)      keep top-Q pairs per i by f_multi descending
7.  family arithmetic
```

A stage runs when the parameter that defines it is given. `compile_f()` runs stages 0–5
and hands the model a `Compiled`, so a loop over site sets pays only for stages 0, 6 and 7
— see [`docs/pipeline.md`](docs/pipeline.md).

## Documentation

```powershell
uv run mkdocs serve
```

Every page also stands alone as plain markdown in the repo:

| Page | Contents |
|---|---|
| [data-contract.md](docs/data-contract.md) | the three input frames; dtypes, rules, scale and memory |
| [modes.md](docs/modes.md) | `Mode` spec, decay callables, availability vs reachability |
| [pipeline.md](docs/pipeline.md) | the stage order, ranking, `open_mask` and `compile_f()` |
| [families/catchment.md](docs/families/catchment.md) | cumulative opportunity |
| [families/voronoi.md](docs/families/voronoi.md) | nearest-site assignment |
| [families/ifca.md](docs/families/ifca.md) | inverted floating catchment area |
| [families/sfca.md](docs/families/sfca.md) | three-step floating catchment area |
| [families/sfca-e.md](docs/families/sfca-e.md) | 3SFCA with explicit participation, for optimisation |
| [outputs.md](docs/outputs.md) | the `Result` object and the full column dictionary |
| [stats.md](docs/stats.md) | the five optional stat groups |
| [sweep.md](docs/sweep.md) | grid syntax, labels, result schema |

## Tests

```powershell
uv run pytest
```

Self-contained: a 6 × 4 toy network with costs and shares chosen so every filter bites.
Golden `E_j` and `A_i` are hand-computed from the formulae above, independently of the
implementation.

## Layout

```
src/interaction_models/
    prepare.py     Prepared, prepare(), validate_inputs()
    compiled.py    Compiled, compile_f(); pipeline stages 0-5, hoisted
    modes.py       Mode, gaussian(), f_multi assembly + NaN renormalisation
    _core.py       segmented numpy primitives (no domain concepts)
    models.py      catchment(), voronoi(), ifca(), sfca(), sfca_e(); Result assembly
    stats.py       optional stat groups
    sweep.py       sweep()
```

## Development AI assistance

Development of this repository is assisted by Anthropic's `Claude Opus 5`
