# Modes and impedance

## The `Mode` spec

```python
from interaction_models import Mode, gaussian

Mode(
    share = "pv_share",        # column on demand_df, or a float constant
    cost  = "cost_default",    # column on cost_df
    kappa = 1.0,               # multiplier on that cost
    decay = gaussian(800),     # callable ndarray -> ndarray, or None
)
```

| Field | Default | Meaning |
|---|---|---|
| `share` | `1.0` | `π_m(i)` — the fraction of node `i`'s demand travelling this way |
| `cost` | `"cost_default"` | which cost column this mode travels on; `NaN` means no route |
| `kappa` | `1.0` | `κ_m`, the multiplier applied before decay |
| `decay` | `None` | `f_m`; `None` means no impedance, i.e. `f_m = 1` |

The combined impedance is a share-weighted sum over any number of modes:

$$
f^{multi}_{ij} \;=\; \sum_m \pi_m(i)\, f_m\!\left(\kappa_m\, c^{(m)}_{ij}\right)
$$

A two-mode split, `f_multi = s·f(d) + (1−s)·f(κd)`:

```python
modes = [
    Mode(share="pv_share",  cost="cost_default", kappa=1.0, decay=gaussian(800)),
    Mode(share="mbt_share", cost="cost_default", kappa=2.0, decay=gaussian(800)),
]
```

**A single-element `modes` list is the unimodal case** — one share of 1.0, one decay, no
split. It is the same arithmetic with one term.

### Shares live on `demand_df`

`π_m` is indexed by `i`, never by `(i, j)`. At 100M pairs a per-pair share column would
cost 400 MB per mode and the data cannot support that resolution anyway. Shares must sum
to 1.0 across the modes of a call, to within `1e-6`; `validate_inputs` enforces it.

A `float` share is a constant across all nodes, which is convenient for tests and for
national-average splits.

## Decay callables

`gaussian(beta)` returns `f(d) = exp(-d² / β)` and is the only decay shipped. Any
vectorised callable works:

```python
def power(alpha):
    return lambda d: (1.0 + d) ** -alpha
```

Two expectations:

* **vectorised** — it receives a `float32` ndarray and must return an ndarray;
* **non-increasing** — assumed by the
  [prefix-truncation fast path](pipeline.md#the-prefix-truncation-fast-path). A
  non-monotone callable gives correct results only when some mode carries its own cost
  column, which forces the general ranking path. Nothing checks this, because checking it
  costs more than the fast path saves.

There is deliberately **no decay registry**. Sweep grids carry `(label, fn)` tuples, so
the label in the result frame is whatever you chose to call it.

## Availability vs reachability

Two mechanisms that look similar and are deliberately **not** unified.

### Unavailable — `cost_m` is `NaN`

No route exists for that mode on that pair. Its share is redistributed **proportionally
over the modes that remain**:

$$
f^{multi}_{ij} \;=\;
\frac{\sum_{m \in \mathcal{A}(i,j)} \pi_m(i)\, f_m(\kappa_m c^{(m)}_{ij})}
     {\sum_{m \in \mathcal{A}(i,j)} \pi_m(i)}
$$

where `A(i,j)` is the set of modes with a route. If every mode is unavailable,
`f_multi = 0` and the `τ` filter drops the pair.

*Example.* With `pv=0.5`, `mbt=0.3`, `walk=0.2` and no walking route on this pair:

```
f_multi = (0.5·f_pv + 0.3·f_mbt) / 0.8
```

### Out of reach — `κ_m · cost_m > D_max`

The route exists but is too costly. The term is dropped with **no** renormalisation. This
is the modelled penalty, and it appears only in the [Voronoi](families/voronoi.md) family,
which is the one place a per-mode reach gate is applied:

$$
E_j = \sum_{i:\,j=j^*(i)} P_i \sum_m \pi_m(i)\,
\mathbf{1}\!\left[\kappa_m d_{ij} \le D_{max}\right]
$$

Renormalising here would erase the loss, leaving the gate with no effect at all. A `NaN`
mode cost fails this gate too — no route is at least as bad as a slow one.

Everywhere else, `D_max` is a **single global filter on `cost_default`**, applied before
any impedance is computed.

## Combining modes

The combine rule is a **share-weighted sum only**. There is no `max()` over modes: a
walking mode belongs in the sum like any other, with its own cost column and its own share
column on `demand_df`.
