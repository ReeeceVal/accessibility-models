# MAC-3SFCA-E

## What it models

The same three-step structure as [3SFCA](sfca.md), with the **participation decision made
explicit**. In 3SFCA the impedance `f_ij` does two jobs at once — deciding *which* site a
node selects, and implicitly deciding *whether* it travels at all. This family separates
them: `Φ_i` gates participation on the node's single best option, and `G_ij` — unchanged,
and capacity-blind — allocates the participating demand across the choice set.

The consequence is that `Σ_j E_j` is **monotone**: opening a site can never lower it. That
is the reason the family exists, and it is what makes `E_j` usable as an objective function
in a site-selection problem. 3SFCA's total can fall when a below-average site opens, which
makes an optimiser refuse to open marginal sites even at zero cost.

## Formulation

Participation, over the choice set `C_Q(i)`:

$$
\Phi_i = \max_{k \in \mathbf{C_Q}(i)} f^{multi}_{ik} \; , \qquad 0 \le \Phi_i \le 1
$$

Selection — identical to 3SFCA, and independent of capacity:

$$
G_{ij} = \frac{f^{multi}_{ij}}{\sum_{k \in \mathbf{C_Q}(i)} f^{multi}_{ik}} \; ,
\qquad \sum_{j} G_{ij} = 1
$$

Expected exposure, the operational load it implies, and the accessibility dual:

$$
E_j = \sum_{i \,:\, j \in \mathbf{C_Q}(i)} P_i\, \Phi_i\, G_{ij}
\qquad
L_j = \frac{E_j}{S_j}
\qquad
A_i = \sum_{j \in \mathbf{C_Q}(i)} R_j\, G_{ij}\, f^{multi}_{ij}
$$

with `R_j = S_j / E_j` retained as a diagnostic, exactly as in 3SFCA.

Capacity never influences **selection**. It enters only through `L_j` and `R_j`, which is
the modelling claim: people choose a site, not a unit of capacity, and capacity then determines how
much load each unit at that site carries.

## Why the total is monotone

Because `Σ_j G_ij = 1`, the network total collapses:

$$
\sum_j E_j = \sum_i P_i \max_{j \in \mathbf{C_Q}(i)} f^{multi}_{ij}
$$

This is the classical **facility-location function** — monotone non-decreasing and
submodular in the site set — so a greedy site-selection heuristic carries the standard
`(1 − 1/e)` approximation guarantee. Opening a site can only raise a node's maximum, never
lower it.

Contrast 3SFCA, whose per-node total is `Σ_j G_ij f_ij`, the *contraharmonic mean* of the
same impedances. Adding an option below that mean drags it down.

## What the parameters add

| Give it | And you get |
|---|---|
| `modes`, `Q`, `D_max`, `τ` | The base formulation. Both `modes` and `Q` are required, as in 3SFCA. |
| two or more `Mode`s | `f^multi` becomes the share-weighted combination of [modes](../modes.md), propagating mode availability into both `Φ_i` and `G_ij`. |

`Q` is a special case here — see [Q moves exposure, not its total](#q-moves-exposure-not-its-total).

## Signature

```python
im.sfca_e(prep, *, modes, D_max=inf, tau=0.0, Q,
          stats=None, output=("demand", "supply"))
```

```python
res = im.sfca_e(prep, modes=modes, D_max=45, tau=0.01, Q=2)
print(res.params)
print(res.supply)
print(res.demand)
```

```
{'D_max': 45, 'Q': 2, 'tau': 0.01, 'n_modes': 2}
supply_id  capacity        E_j      R_j       L_j  n_demand_j
       s1      10.0 122.196888 0.081835 12.219689           2
       s2       5.0 128.103926 0.039031 25.620785           3
demand_id  demand      A_i     SPAR    Phi_i  n_supply_i  pv_share  mbt_share
       d1   100.0 0.057899 1.794358 0.969233           2      1.00       0.00
       d2   200.0 0.035410 1.097388 0.744514           2      0.50       0.50
       d3    50.0 0.003493 0.108253 0.089495           1      0.25       0.75
```

Read `Φ_i` directly: `d1` sits five minutes from `s1` on the primary transport mode, so 97% of it
participates. `d3` is 30 minutes from its only option and mostly reliant on the secondary transport mode,
so under 9% does.

`s2` draws marginally more total exposure than `s1` (128.1 against 122.2) but carries
**more than twice the per-unit load** (25.6 against 12.2), because it has half the
capacity. That gap is what `L_j` exists to surface, and it is invisible in `E_j` alone.

## Compared with 3SFCA

The same network, same parameters, through [`sfca()`](sfca.md):

| | `E_j` at `s1` | `E_j` at `s2` | `Σ_j E_j` |
|---|---|---|---|
| 3SFCA | 104.985013 | 114.143066 | 219.128079 |
| MAC-3SFCA-E | 122.196888 | 128.103926 | 250.300814 |

MAC-3SFCA-E is always the larger of the two in total. `Φ_i` is the **maximum** of the
impedances in the choice set where 3SFCA uses their contraharmonic mean, and a weighted
average never exceeds a maximum. The two coincide exactly when every impedance in a node's
choice set is equal — a node equidistant from all its options.

The allocation across sites also differs: 3SFCA spreads a node's demand in proportion to
`f²`, this family in proportion to `f`. So it is not simply a rescaling of 3SFCA, and
per-site the difference can go either way even though the total cannot.

## `Q` moves exposure, not its total

`Φ_i` is **invariant to `Q`**. The maximising pair is necessarily the highest-ranked member
of `C_Q(i)`, so it survives any `Q ≥ 1`, and therefore so does `Σ_j E_j`:

```
Q=1: sum E_j = 250.300814   E_j = [96.9233, 153.3775]
Q=2: sum E_j = 250.300814   E_j = [122.1969, 128.1039]
Q=3: sum E_j = 250.300814   E_j = [122.1969, 128.1039]
```

`Q` is not inert — it redistributes exposure between sites, visibly so above — it simply
conserves the total. For an optimisation this decouples the two mechanisms: `Q` can be
tuned for allocation realism without perturbing the objective.

## Conventions

* **`capacity` is optional for `E_j`** and required for `A_i`. Without it `R_j` and `L_j`
  are both `NaN`, and requesting the demand frame raises `ValueError`.
* **Doubling capacity leaves `E_j` untouched**, halves `L_j` and doubles `A_i`. Selection
  is capacity-blind; that is a test.
* **`E_j = 0`** for a site nobody reaches, with `R_j = NaN`. `L_j` is `0.0` there rather
  than `NaN`, since `0 / S_j` is well defined for positive capacity.
* **Empty choice set** (a node reaching nobody, or everything falling below `τ`) gives
  `Φ_i = 0`, `A_i = 0` and `n_supply_i = 0`.
* **`Φ_i ≤ 1`** whenever the decay is bounded by 1, which every decay shipped here is. It
  is a participation *proportion*, so a decay exceeding 1 would make it meaningless.
* **`τ` is applied before `Φ_i` and `G_ij`** — `C_Q(i)` is restricted to pairs within
  `D_max` and above `τ` first.

## When to use which

| | Use |
|---|---|
| Estimating exposure over a **fixed** network | [`sfca()`](sfca.md) — non-monotonicity is irrelevant when the site set never changes |
| Choosing **which sites to open** | `sfca_e()` — the objective must not fall when a site is added |
| Allocating **capacity** across chosen sites | `sfca_e()`, minimising the dispersion of `L_j` |

A natural pairing is to maximise `Σ_j E_j` to fix the geography, then allocate `S_j` to
level `L_j` across the chosen sites. The first is capacity-blind and monotone; the second
is where capacity does its work.
