# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- MIT license, package metadata (keywords, classifiers, project URLs) and a `py.typed`
  marker.
- `ruff format` and a broader lint rule set (import sorting, bugbear, pyupgrade), both
  enforced in CI.
- This changelog.
- `CITATION.cff` and a README citation section for the MAC-3SFCA paper (DSA ISC 2026,
  in press).

### Changed

- README rewritten; installation now targets `pip` / `uv` from git.
- Documentation examples use neutral identifiers and wording.

## [0.3.0] - 2026-08-21

### Added

- `explain()` accepts `supply_id` as a mutually exclusive alternative to `demand_id`,
  returning the per-pair terms for every demand node that reaches the named site(s).

## [0.2.0] - 2026-08-21

### Added

- `explain()`: the per-pair terms a model call sums away (`cost_default`, `f_multi` and a
  family-specific `weight`) for one demand node or a short list, without materialising
  the pair table.

## [0.1.0] - 2026-08-21

First tagged release.

### Added

- `prepare()` and `validate_inputs()`: validate, factorise and sort the three input frames
  once for reuse across many model calls.
- Five model families: `catchment()`, `voronoi()`, `ifca()`, `sfca()` (3SFCA) and
  `sfca_e()` (MAC-3SFCA-E, the elastic-participation variant of MAC-3SFCA, with `L_j`).
- Multi-mode impedance through `Mode` and `gaussian()`, renormalised over the modes
  available on each pair.
- `open_mask` on every family, for evaluating candidate site sets without re-preparing.
- `compile_f()`, which hoists pipeline stages 0–5 out of the model call. `width=` bounds the
  candidate read exactly, and `bare=True` returns `E_j` as a plain array.
- Optional statistics groups, including `exposure` and a capacity-aware `gini_L_j`
  scoped to the open sites.
- `sweep()`: a minimal parameter-grid runner.
- CI across Python 3.10–3.13.

[Unreleased]: https://github.com/ReeeceVal/accessibility-models/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ReeeceVal/accessibility-models/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ReeeceVal/accessibility-models/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ReeeceVal/accessibility-models/releases/tag/v0.1.0
