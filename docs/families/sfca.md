# 3SFCA

## What it models

The three-step floating catchment area method adds a **demand-side selection step**: each
node spreads its demand across the sites in its choice set in proportion to their relative
impedance, rather than counting itself fully at every one.

Because the selection probability already turns demand into an expected count, there is no
multiplication by `S_j` anywhere. That is the structural difference from iFCA, not a
parameter choice.

## Formulation

Selection probability over the choice set:

$$
G_{ij} = \frac{f^{multi}_{ij}}{\sum_{k \in \mathbf{C_Q}(i)} f^{multi}_{ik}}
$$

Expected exposure, and its accessibility dual:

$$
E_j = \sum_{i \,:\, j \in \mathbf{C_Q}(i)} P_i\, G_{ij}\, f^{multi}_{ij}
= \sum_{i \,:\, j \in \mathbf{C_Q}(i)}
\frac{P_i\, (f^{multi}_{ij})^{2}}{\sum_{k \in \mathbf{C_Q}(i)} f^{multi}_{ik}}
\qquad
A_i = \sum_{j \in \mathbf{C_Q}(i)} R_j\, G_{ij}\, f^{multi}_{ij}
$$

`A_i` is the classic two-step accessibility read-back: the node's share of the
supply-to-demand ratio at every site it selects. The ratio itself is retained **only as a
diagnostic**:

$$
R_{j} = \frac{S_{j}}{E_{j}}
$$

`C_Q(i)` is the top-`Q` set *after* `D_max` and `τ`, so both sums run over the same
filtered pairs.

## What the parameters add

| Give it | And you get |
|---|---|
| `modes`, `Q`, `D_max`, `τ` | The base formulation. Both are required: `G_ij` is undefined without impedance and without a choice set. |
| two or more `Mode`s | `f^{multi}` becomes the share-weighted combination of [modes](../modes.md), propagating mode availability into both the selection probability and the exposure weight. |

## Signature

```python
im.sfca(prep, *, modes, D_max=inf, tau=0.0, Q,
        stats=None, output=("demand", "supply"))
```

```python
res = im.sfca(prep, modes=modes, D_max=45, tau=0.01, Q=2)
print(res.params)
print(res.supply)
print(res.demand)
```

```
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2}
supply_id  capacity        E_j      R_j  n_demand_j
       s1      10.0 104.985013 0.095252           2
       s2       5.0 114.143066 0.043805           3
demand_id  demand       A_i     SPAR  n_supply_i  pv_share  mbt_share
       d1   100.0 0.067012 1.803889           2      1.00       0.00
       d2   200.0 0.040514 1.090582           2      0.50       0.50
       d3    50.0 0.003920 0.105530           1      0.25       0.75
```

## Conventions

* **`Σ_j G_ij = 1`** for every node with a non-empty choice set. That is a test.
* **`capacity` is optional for `E_j`** and required for `A_i` — `R_j` is the only place
  `S_j` appears. Doubling capacity leaves `E_j` untouched and doubles `A_i`; also a test.
* **`R_j = NaN` where `E_j = 0`.** Such a site draws no pair, so the `NaN` never
  propagates into anyone's `A_i`.
* **Empty choice set** (a node reaching nobody, or everything falling below `τ`) gives
  `A_i = 0`, `n_supply_i = 0`, and `G_ij` undefined but unused.
* **`τ` is applied before `G_ij` is normalised** — `C_Q(i)` is restricted to pairs within
  `D_max` and above `τ` first, so the selection probabilities sum to 1 over what survives.
