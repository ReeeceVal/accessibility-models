# Voronoi

## What it models

Each node sends **all** of its demand to its single nearest reachable site — winner takes
all. Where Catchment counts a node at every site in range, Voronoi partitions demand
exactly once, so total exposure can never exceed total demand.

## Formulation

The nearest reachable site:

$$
j^{*}(i) := \operatorname*{arg\,min}_{\,j \in \mathbf{C_D}(i)} d_{ij}
$$

Winner-take-all assignment, with a per-mode reach gate:

$$
E_j = \sum_{i \,:\, j = j^{*}(i)} P_i \sum_m \pi_m(i)\,
\mathbf{1}\!\left[\kappa_m c^{(m)}_{ij^{*}} \le D_{max}\right]
\qquad
A_i = \frac{S_{j^{*}(i)}}{E_{j^{*}(i)}}
$$

Every node in the same cell has the same accessibility — capacity per unit of competing
demand. Unassigned nodes get `A_i = 0`.

## What the parameters add

| Give it | And you get |
|---|---|
| `D_max` only | Plain closest-facility assignment. The inner sum is 1, so `E_j` is the demand of the cell. |
| two or more `Mode`s | The reach gate bites: a node's demand is discounted by the share of modes that cannot reach `j*` under `κ_m · c ≤ D_max`. |

`τ` and `Q` are not parameters of this family — there is no impedance to threshold or
rank — and a `decay` on a mode is accepted but ignored.

## Signature

```python
im.voronoi(prep, *, modes=None, D_max=inf,
           stats=None, output=("demand", "supply"))
```

```python
res = im.voronoi(prep, modes=modes, D_max=45)
print(res.params)
print(res.supply)
print(res.demand)
```

```
{'D_max': 45, 'Q': None, 'tau': None, 'n_modes': 2}
supply_id  capacity   E_j      R_j  n_demand_j
       s1      10.0 100.0 0.100000           1
       s2       5.0 212.5 0.023529           2
demand_id  demand      A_i  SPAR  n_supply_i  pv_share  mbt_share
       d1   100.0 0.100000  2.04           1      1.00       0.00
       d2   200.0 0.023529  0.48           1      0.50       0.50
       d3    50.0 0.023529  0.48           1      0.25       0.75
```

## Conventions

* **Ties** break on the lowest supply code, i.e. the site appearing first in `supply_df`.
  Deterministic, and independent of `cost_df` row order.
* **Decay is ignored.** There is no impedance here; passing a `decay` on a mode changes
  nothing. `κ` still matters, because it drives the reach gate.
* **`κ` can never move `j*`.** It is a constant scale on a cost that is already minimised,
  so mode split only ever removes demand from the winner — it does not reassign it. Hence
  adding modes can only lower `E_j`, which is a test.
* **The reach gate does not renormalise.** A share whose mode cannot reach `j*` is
  *lost*, not redistributed to the modes that can. See
  [availability vs reachability](../modes.md#availability-vs-reachability).
* `n_supply_i` is 1 for an assigned node, 0 for an unreachable one.
* `capacity` is optional for `E_j`, required for `A_i`.
