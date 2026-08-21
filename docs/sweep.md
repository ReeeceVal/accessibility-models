# Sweep

`sweep()` runs one model across the Cartesian product of a parameter grid, reusing a
single `Prepared`. That reuse is the whole point: factorising and sorting happens once,
and every combination after that is pure numpy over the same arrays.

```python
rows = im.sweep(prep, "sfca", modes=modes, D_max=45, tau=0.01, grid={
    "decay": [("gauss800", im.gaussian(800)), ("gauss1200", im.gaussian(1200))],
    "kappa": [1.5, 2.0],
    "Q":     [1, 2],
}, stats=["coverage", "choice_set"])
```

Run against the frames of the [quickstart](index.md#quickstart):

```
model     decay  kappa  Q  D_max  tau  n_modes     sum_E_j  mean_N_eff_i  frac_demand_q_binding
 sfca  gauss800    1.5  1     45 0.01        2 3730.786360      1.000000                    1.0
 sfca  gauss800    1.5  2     45 0.01        2 3178.392011      1.452439                    1.0
 sfca  gauss800    2.0  1     45 0.01        2 3387.275412      1.000000                    1.0
 sfca  gauss800    2.0  2     45 0.01        2 2940.690758      1.380428                    1.0
 sfca gauss1200    1.5  1     45 0.01        2 3964.885044      1.000000                    1.0
 sfca gauss1200    1.5  2     45 0.01        2 3347.506268      1.608554                    1.0
 sfca gauss1200    2.0  1     45 0.01        2 3688.039718      1.000000                    1.0
 sfca gauss1200    2.0  2     45 0.01        2 3127.758735      1.505823                    1.0
```

(columns trimmed for width — `stats=["coverage", "choice_set"]` emits all fifteen keys)

## Signature

```python
im.sweep(prep, model, grid, *, modes=None, stats=None, **fixed)
```

| Argument | Meaning |
|---|---|
| `prep` | built once, reused for every combination |
| `model` | `"catchment"`, `"voronoi"`, `"ifca"`, `"sfca"`, `"sfca_e"`, or a model function |
| `grid` | `{key: [values]}` — the Cartesian product |
| `modes` | the mode template that `"decay"` and `"kappa"` rewrite |
| `stats` | stat groups; their keys become result columns |
| `**fixed` | constant model arguments, e.g. `D_max=45` |

## Grid keys

Exactly six are allowed. Anything else raises rather than being silently forwarded.

| Key | Effect |
|---|---|
| `D_max`, `tau`, `Q` | passed straight to the model |
| `modes` | replaces the whole mode list |
| `decay` | applied to **every** mode of the template |
| `kappa` | applied to `modes[1:]` — every non-reference mode |

`decay` and `kappa` treat `β` and `κ` as single scalars rather than per-mode ones, which is
how the two-mode form `f_multi = s·f(d) + (1−s)·f(κd)` reads: mode 0 holds the reference
`κ = 1` and the rest are penalised relative to it. So one grid value rewrites all the modes
it should.

When that is not what you want, sweep whole mode lists instead:

```python
grid={"modes": [("single", single_modes), ("mac", mac_modes)]}
```

`"decay"` and `"kappa"` require `modes=` to be given; without a template there is nothing
to rewrite, and that raises.

## Labels

Every grid value is either a plain value or a `(label, value)` tuple. The label is what
lands in the result column:

```python
"decay": [("gauss800", im.gaussian(800)), ("gauss1200", im.gaussian(1200))]
```

Without labels the column would hold `<function decay at 0x...>`, which is useless in a
results table. There is deliberately no decay registry — you name your own functions and
the name is whatever you chose.

Plain values need no label: `"Q": [5, 10, 20]` gives a `Q` column holding 5, 10, 20.

## Result schema

One row per combination:

1. `model` — which model function produced the row;
2. one column per grid key, holding its label;
3. the resolved `D_max`, `Q`, `tau`, `n_modes`, `n_open`, `n_width_fallback`, skipping
   any already present as a grid
   column;
4. every key of every requested stat group.

## What it does not do

`output=()` is forced. A sweep row is a scalar summary, so per-node and per-site frames
would be assembled and thrown away. Call the model directly when you want them.

That also makes `stats` the only channel out of a sweep, so pick the groups with the cost
in mind: when the rows need only supply-side numbers, ask for
[`exposure`](stats.md#exposure) rather than `coverage` and `A_i` is never computed — the
bulk of the per-combination work.

No dedup, no persistence, no resume, no evaluation against realised counts. If you need
something `sweep()` does not do, write the loop yourself — the expensive part is already
cached in `prep`, which is exactly why no harness is needed here.

Time periods are also out of scope: the caller loops them, passing a fresh `cost_df` or
`supply_df` per period.
