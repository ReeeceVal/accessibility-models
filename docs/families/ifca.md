# iFCA

## What it models

The inverted floating catchment area method (Wang 2018) makes sites **compete for
demand**. A node first works out how thinly its demand is spread across the capacity it
can reach (`r_i`), then each site accumulates the resulting per-unit demand — its
*crowdedness*. Two nodes with the same population contribute very differently depending
on how much other supply they have available.

## Formulation

$$
r_{i} = \frac{P_i}{\sum_{j} S_{j} f^{multi}_{ij}}
\qquad
C_{j} = \sum_{i} r_{i} f^{multi}_{ij}
\qquad
E_{j} = S_{j}\, C_{j}
$$

Substituted out:

$$
E_j = S_j \sum_{i} \frac{P_i f^{multi}_{ij}}{\sum_{k} S_{k} f^{multi}_{ik}}
\qquad
A_i = \frac{1}{r_i} = \frac{\sum_j S_j f^{multi}_{ij}}{P_i}
$$

`A_i` is the natural accessibility reading of the inverted formulation: impedance-weighted
capacity per head. `r_i` is returned as its own column, since it is the quantity the
formulation is written in.

Every sum runs over the same filtered pair set: `C_D(i)` after `τ`, further restricted to
`C_Q(i)` when `Q` is set. `D_max` is a common bound on the whole package, this denominator
included.

## What the parameters add

| Give it | And you get |
|---|---|
| `modes`, `D_max`, `τ` | The base formulation. Denominators run over everything within reach. |
| `Q` | The denominators and both sums are restricted to each node's top-`Q` sites by `f_multi`, so a node competes only over the sites it would realistically use. |
| two or more `Mode`s | `f^{multi}` becomes the share-weighted combination of [modes](../modes.md). |

`modes` is **required**: with no impedance the ratio `r_i` is undefined.

## Signature

```python
im.ifca(prep, *, modes, D_max=inf, tau=0.0, Q=None,
        stats=None, output=("demand", "supply"))
```

```python
res = im.ifca(prep, modes=modes, D_max=45, tau=0.01, Q=2)
print(res.params)
print(res.supply)
print(res.demand)
```

```
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2}
supply_id  capacity      E_j      R_j  n_demand_j
       s1      10.0 194.5327 0.051405           2
       s2       5.0 155.4673 0.032161           3
demand_id  demand      A_i     SPAR        r_i  n_supply_i  pv_share  mbt_share
       d1   100.0 0.127250 2.099839   7.858556           2      1.00       0.00
       d2   200.0 0.045600 0.752480  21.929755           2      0.50       0.50
       d3    50.0 0.008949 0.147682 111.738266           1      0.25       0.75
```

## Conventions

* **`capacity` is required.** `S_j` sits in the denominator, so omitting it raises
  `ValueError` rather than defaulting to 1.
* **Zero denominator.** A node with `Σ_j S_j f_ij = 0` (unreachable, or reaching only
  zero-capacity sites) gets `r_i = inf`, `A_i = 0`, and contributes **nothing** to any
  `C_j`. It is not silently dropped from the demand frame.
* **`Q = None` and `Q = inf` are the same thing**, and both report `params["Q"] = None`.
* **Ties in the ranking** break on ascending `cost_default`, then supply code.
* `E_j = 0` for a site nobody reaches; `R_j = NaN` there.
