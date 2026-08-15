# interaction-models

A lean, environment-agnostic Python package computing spatial interaction and
accessibility models over a demand → supply cost matrix.

Five model families, each a single model whose optional terms switch on with the
parameters that define them:

| Family | What it does | Page |
|---|---|---|
| **Catchment** | cumulative opportunity, no competition | [catchment](families/catchment.md) |
| **Voronoi** | winner-take-all assignment to the nearest site | [voronoi](families/voronoi.md) |
| **iFCA** | sites compete for demand, weighted by crowdedness | [ifca](families/ifca.md) |
| **3SFCA** | demand-side selection across a bounded choice set | [sfca](families/sfca.md) |
| **MAC-3SFCA-E** | 3SFCA with an explicit participation step; monotone under site openings | [sfca-e](families/sfca-e.md) |

Every family emits the same two outputs: `E_j`, the expected exposure of a supply point,
and `A_i`, the accessibility of a demand node.

## Three ideas

**`prepare()` once, sweep cheaply.** Identifiers are factorised to `int32` and the pair
table is sorted once; every subsequent model call is pure numpy over the same arrays.
Built for ~1M demand nodes × ~1k supply points — 10 to 100 million pairs.

**Parameters, not model names.** Impedance, mode split and a bounded choice set are terms
you switch on by supplying `decay`, a second `Mode` or `Q`. The resolved configuration
comes back as `res.params`.

**No I/O, no orchestration.** Three DataFrames in, two DataFrames out.

## Install

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
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2}
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

Three things to notice:

* `province` came through untouched — see [outputs](outputs.md#passthrough);
* IDs stayed `str`, so the leading zeros survived — see
  [the data contract](data-contract.md#identifiers);
* `params` records the whole configuration, so a result is self-describing.

## Where to next

* [The data contract](data-contract.md) — what the three frames must contain, and scale
* [Modes](modes.md) — impedance, mode shares, and the availability/reachability split
* [The pipeline](pipeline.md) — the seven stages, and which families run which
* [Outputs](outputs.md) — the `Result` object and every column it can emit
* [Sweep](sweep.md) — running a grid over one `Prepared`

Choosing a family for a **site-selection optimisation** rather than for estimation over a
fixed network? See [MAC-3SFCA-E](families/sfca-e.md#when-to-use-which) — most of the
families' `Σ_j E_j` behave badly as objective functions.

## Scope

| In scope | Out of scope |
|---|---|
| `E_j`, `A_i`, optional stats, for one configuration | time periods — the caller loops them |
| `prepare()` — factorise and sort once | I/O, orchestration, storage |
| `sweep()` — a minimal grid runner | dedup, persistence, resume |
| input validation | evaluation against realised counts |
