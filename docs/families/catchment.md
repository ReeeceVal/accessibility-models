# Catchment

## What it models

Every site accumulates the demand of every node that can reach it within `D_max`, with
**no competition** between sites — one node's demand can be counted by several sites at
once. This is the cumulative-opportunity reading of access: how much demand is nearby,
and how much capacity a node can see.

## Formulation

Over the pairs surviving `D_max` (and `τ`, when there is impedance):

$$
E_j = \sum_{i \,:\, (i,j) \in \mathcal{S}} P_i\, f^{multi}_{ij}
\qquad
A_i = \sum_{j \,:\, (i,j) \in \mathcal{S}} S_j\, f^{multi}_{ij}
$$

`A_i` is the accessibility dual of the same sum — cumulative opportunity seen from the
demand side. Both run over the **same** pair set.

## What the parameters add

| Give it | And you get |
|---|---|
| `D_max` only | Binary accumulation. `f_multi = 1`, so `E_j` is reachable demand and `A_i` is reachable capacity. |
| one `Mode` with a `decay` | Distance-weighted accumulation. `τ` becomes live as a floor on `f_multi`. |
| two or more `Mode`s | The impedance becomes the share-weighted combination of [modes](../modes.md); `κ` penalises the slower modes. |

Nothing else changes between those: the arithmetic above is the whole family, and the
parameters only decide what `f_multi` is.

## Signature

```python
im.catchment(prep, *, modes=None, D_max=inf, tau=0.0,
             stats=None, output=("demand", "supply"))

im.catchment(demand_df, supply_df, cost_df, modes=..., ...)   # prepares internally
```

```python
res = im.catchment(prep, modes=modes, D_max=45, tau=0.01)
print(res.params)
print(res.supply)
print(res.demand)
```

```
{'D_max': 45, 'Q': None, 'tau': 0.01, 'n_modes': 2}
supply_id  capacity        E_j      R_j  n_demand_j
       s1      10.0 204.872513 0.048811           2
       s2       5.0 214.030565 0.023361           3
demand_id  demand       A_i     SPAR  n_supply_i  pv_share  mbt_share
       d1   100.0 12.724985 1.712458           2      1.00       0.00
       d2   200.0  9.120029 1.227323           2      0.50       0.50
       d3    50.0  0.447474 0.060219           1      0.25       0.75
```

## Conventions

* **`E_j = 0`** for a site nobody reaches; **`R_j = NaN`** there, since `S_j / 0` is not a
  number worth inventing.
* **`A_i = 0`** for a node reaching nobody. `SPAR` is still finite (`0 / mean`).
* **`capacity` is optional** for `E_j`, which sums demand only. It is **required** for
  `A_i`; asking for the demand frame without it raises `ValueError`.
* **`tau` is only live where there is impedance.** With no `decay` on any mode there is
  nothing for it to floor, and `params["tau"]` comes back `None` to say so — including
  when two modes are given but neither carries a decay.
