# accessibility-models

[![tests](https://github.com/ReeeceVal/accessibility-models/actions/workflows/tests.yml/badge.svg)](https://github.com/ReeeceVal/accessibility-models/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Spatial interaction and accessibility models over a demand → supply cost matrix**, in
pure numpy and pandas.

Given where demand lives, where supply points sit, and what it costs to travel between
them, this package answers two questions for any network of sites:

* **`E_j`**: how much demand is each supply point expected to attract?
* **`A_i`**: how well served is each demand node?

It implements five model families from the floating catchment area (FCA) and spatial
interaction literature behind one consistent interface. It is built to run at national
scale (about 1M demand nodes × 1k sites, or 10–100M origin-destination pairs) and fast
enough to sit inside a site-selection optimisation loop.

## Highlights

* **Five model families, one interface.** Every family takes the same inputs and returns
  the same two outputs, so switching model is a one-word change.
* **Parameters, not model variants.** Distance decay, multi-mode travel and a bounded
  choice set are switched on by passing `decay`, a second `Mode` or `Q`. The resolved
  configuration comes back on every result as `res.params`.
* **Prepare once, evaluate many times.** `prepare()` factorises identifiers to `int32` and
  sorts the pair table once. `compile_f()` also fixes the impedance, so a loop over
  candidate site sets pays only for selection and arithmetic.
* **Exact fast paths.** `open_mask`, `width=` and `bare=True` speed up optimisation loops
  and match the reference computation bit for bit, not just within a tolerance.
* **An objective fit for optimisation.** MAC-3SFCA-E's total exposure is monotone and
  submodular in the site set, so greedy selection carries the classical `(1 − 1/e)`
  guarantee.
* **Library, not framework.** No I/O and no orchestration: three DataFrames in, two
  DataFrames out. Typed, tested on Python 3.10–3.13, and depends only on numpy and pandas.

## Model families

| Family | What it models | `E_j` | `A_i` |
|---|---|---|---|
| **Catchment** | cumulative opportunity, no competition between sites | `Σ_i P_i f_ij` | `Σ_j S_j f_ij` |
| **Voronoi** | winner-take-all assignment to the nearest reachable site | `Σ_{i: j=j*(i)} P_i Σ_m π_m 1[κ_m d_ij ≤ D_max]` | `S_{j*} / E_{j*}` |
| **iFCA** | sites compete for demand, weighted by crowdedness | `S_j Σ_i r_i f_ij`, `r_i = P_i / Σ_j S_j f_ij` | `1 / r_i` |
| **3SFCA** | demand-side selection across a bounded choice set | `Σ_i P_i G_ij f_ij`, `G_ij = f_ij / Σ_k f_ik` | `Σ_j R_j G_ij f_ij` |
| **MAC-3SFCA-E** | mode-availability constrained 3SFCA with elastic participation | `Σ_i P_i Φ_i G_ij`, `Φ_i = max_j f_ij` | `Σ_j R_j G_ij f_ij` |

`P_i` is demand, `S_j` is capacity, and `f_ij` is the combined impedance across travel
modes:

```
f_ij = Σ_m π_m(i) · f_m(κ_m · cost_m(i,j))       (renormalised over available modes)
```

With no `decay` on any mode, `f_ij = 1` and Catchment reduces to plain counts of demand and
capacity within reach.

MAC-3SFCA-E, the elastic-participation variant of MAC-3SFCA, separates *whether* a node
travels (`Φ_i`) from *which* site it picks (`G_ij`), two things 3SFCA conflates. Because
`Σ_j G_ij = 1`, its total collapses to `Σ_i P_i max_j f_ij`, the classical
facility-location function. That makes it usable as an optimisation objective where the
other families' totals are not. It also emits `L_j = E_j / S_j`, the load per unit of
capacity. See [`docs/families/sfca-e.md`](docs/families/sfca-e.md).

The mode-availability constrained method, MAC-3SFCA, is introduced in Valentine & Grobler
(2026, in press). See [Citation](#citation).

## Installation

Requires Python 3.10 or newer. The distribution is named `interaction-models` and is
imported as `interaction_models`.

```bash
pip install "interaction-models @ git+https://github.com/ReeeceVal/accessibility-models.git"

# or, inside a uv project
uv add "interaction-models @ git+https://github.com/ReeeceVal/accessibility-models.git"
```

Append a tag, e.g. `...accessibility-models.git@v0.3.0`, to pin a release.

## Quickstart

```python
import pandas as pd
import interaction_models as im

demand_df = pd.DataFrame({
    "demand_id":  ["d1", "d2", "d3", "d4"],
    "demand":     [1200.0, 800.0, 450.0, 2100.0],
    "pv_share":   [0.35, 0.10, 0.60, 0.25],        # primary transport mode share
    "mbt_share":  [0.65, 0.90, 0.40, 0.75],        # secondary transport mode share
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
print(res.supply.to_string(index=False))
print(res.demand.to_string(index=False))
print(res.stats["demand_capture_rate"])
```

```
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2, 'n_open': None, 'n_width_fallback': None}
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

Extra columns such as `province` pass straight through to the output, and every result
carries its full configuration in `res.params`.

## Two workflows: vary the network, or vary the parameters

`f_ij` never depends on which sites are open, so you can hoist out of the loop whichever
stage holds the part that is not changing. Sites are addressed by a boolean mask with one
flag per `supply_df` row:

```python
mask = supply_df["supply_id"].isin(["0010001", "0010003"]).to_numpy()
```

**Optimisation: the network changes, the parameters don't.**
`prepare()` → `compile_f(modes, D_max, tau)` → **mask** → `sfca_e(comp, Q, open_mask)`

```python
comp = im.compile_f(prep, modes=modes, D_max=45, tau=0.01, width=20)
for mask in candidate_networks:
    E_j = im.sfca_e(comp, Q=2, open_mask=mask, bare=True)      # ndarray, no frames
```

`width` bounds the candidate list each call reads, and `bare` returns `E_j` alone. Both are
optional and both leave the numbers bit-for-bit unchanged. `width` only helps when the open
set is dense; see [`docs/pipeline.md`](docs/pipeline.md).

**Calibration: the network is fixed, the parameters change.**
`prepare()` → **mask** → `compile_f(modes, D_max, tau, sites)` → `sfca_e(comp, Q)`

```python
for tuned in parameter_combinations:
    comp = im.compile_f(prep, modes=tuned, D_max=45, tau=0.01, sites=mask)
    E_j = im.sfca_e(comp, Q=2).supply["E_j"]
```

Or grid-search in one call:

```python
im.sweep(prep, "sfca_e", modes=modes, D_max=45, tau=0.01, Q=2,
         open_mask=mask, grid={"kappa": [1.5, 2.0]})
```

`C_Q(i)` is a node's top `Q` **open** options, and closed sites stay in the output as zero
rows so results line up across candidate networks.

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

A stage runs only when you pass the parameter that defines it. `compile_f()` runs stages
0–5 once, so a loop over site sets pays only for stages 0, 6 and 7.

## Documentation

The full reference lives in [`docs/`](docs/index.md) as plain markdown:

| Page | Contents |
|---|---|
| [data-contract.md](docs/data-contract.md) | the three input frames: dtypes, rules, scale and memory |
| [modes.md](docs/modes.md) | `Mode` spec, decay callables, availability vs reachability |
| [pipeline.md](docs/pipeline.md) | the stage order, ranking, `open_mask` and `compile_f()` |
| [families/catchment.md](docs/families/catchment.md) | cumulative opportunity |
| [families/voronoi.md](docs/families/voronoi.md) | nearest-site assignment |
| [families/ifca.md](docs/families/ifca.md) | inverted floating catchment area |
| [families/sfca.md](docs/families/sfca.md) | three-step floating catchment area |
| [families/sfca-e.md](docs/families/sfca-e.md) | MAC-3SFCA with elastic participation, for optimisation |
| [outputs.md](docs/outputs.md) | the `Result` object and the full column dictionary |
| [explain.md](docs/explain.md) | per-pair terms for one demand node or supply point, without materialising the pair table |
| [stats.md](docs/stats.md) | the five optional statistics groups |
| [sweep.md](docs/sweep.md) | grid syntax, labels, result schema |

## Development

```bash
git clone https://github.com/ReeeceVal/accessibility-models.git
cd accessibility-models
uv sync                  # creates .venv with the dev dependencies

uv run pytest            # test suite
uv run ruff check        # lint
uv run ruff format       # format
```

The tests are self-contained. They run on a 6 × 4 toy network whose costs and shares are
chosen so every filter takes effect, and the golden `E_j` and `A_i` values are worked out
by hand from the formulae, independently of the implementation. CI runs the suite on
Python 3.10–3.13, plus lint, format and lockfile checks.

### Layout

```
src/interaction_models/
    prepare.py     Prepared, prepare(), validate_inputs()
    compiled.py    Compiled, compile_f(); pipeline stages 0-5, hoisted
    modes.py       Mode, gaussian(), f_multi assembly + NaN renormalisation
    _core.py       segmented numpy primitives (no domain concepts)
    models.py      catchment(), voronoi(), ifca(), sfca(), sfca_e(); Result assembly
    explain.py     explain(); per-pair terms for one demand node or one supply point
    stats.py       optional statistics groups
    sweep.py       sweep()
```

## Citation

MAC-3SFCA, the mode-availability constrained method this package builds on, is introduced
in the paper below, which has been **accepted for publication and is in press**. If you use
this package in academic work, please cite it. GitHub's *Cite this repository* button
generates the same reference from [`CITATION.cff`](CITATION.cff).

> Valentine, R.C., Grobler, J.: A mode-availability constrained variant and
> uncertainty-aware evaluation procedure for floating catchment area methods. In:
> *Decision Sciences: Fourth Decision Science Alliance International Summer Conference,
> DSA ISC 2026, Madrid, Spain, June 18–19, 2026, Proceedings*. Lecture Notes in Computer
> Science. Springer, Cham (2026, in press)

Reece C. Valentine ([ORCID](https://orcid.org/0009-0008-4826-639X)) and Jacomine Grobler
([ORCID](https://orcid.org/0000-0002-1868-0759)), Department of Industrial Engineering,
Stellenbosch University.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{valentine2026mac3sfca,
  author    = {Valentine, Reece C. and Grobler, Jacomine},
  title     = {A Mode-Availability Constrained Variant and Uncertainty-Aware
               Evaluation Procedure for Floating Catchment Area Methods},
  booktitle = {Decision Sciences: Fourth Decision Science Alliance International
               Summer Conference, DSA ISC 2026, Madrid, Spain, June 18--19, 2026,
               Proceedings},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  note      = {Accepted for publication, in press}
}
```

</details>

The volume, pages and DOI will be added once the proceedings are published.

## Background

The families build on the floating catchment area literature:

* Luo, W. & Wang, F. (2003). Measures of spatial accessibility to health care in a GIS
  environment. *Environment and Planning B*, 30(6), 865–884.
* Wan, N., Zou, B. & Sternberg, T. (2012). A three-step floating catchment area method for
  analyzing spatial access to health services. *International Journal of Geographical
  Information Science*, 26(6), 1073–1089.
* Wang, F. (2018). Inverted two-step floating catchment area method for measuring facility
  crowdedness. *The Professional Geographer*, 70(2), 251–260.

## AI assistance

Development of this repository was assisted by Anthropic's Claude.

## License

[MIT](LICENSE) © Reece Valentine
