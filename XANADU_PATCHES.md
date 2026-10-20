# Xanadu patches

This branch is an upstream fdtdx release ([ymahlau/fdtdx](https://github.com/ymahlau/fdtdx)) plus the short series
of Xanadu commits below, which SilverHammer needs inside fdtdx itself. Everything else Xanadu-specific (subpixel
averaging, mode caching, dtype choices, buffer donation, plotting, S3) lives in SilverHammer.

- **Branch naming:** `xanadu/<upstream tag>`, for example `xanadu/v0.6.3` is `v0.6.3` plus the patches below.
- **Syncing:** a sync replays the series onto the next upstream tag as a new branch. Existing branches are never
  rewritten, so commit pins in SilverHammer keep working.
- **Markers:** every edit inside an upstream file carries a `XANADU:` comment (or, for the dtype pins, uses the
  `material_dtype` name), so conflicts are quick to find.
- **Tests:** Xanadu tests live in `tests/**/xanadu/`. The upstream suite (`-m "unit or integration"`) must stay green.

| # | Patch | Why | Files | Tests |
|---|---|---|---|---|
| 1 | Precomputed materials in `place_objects` | SilverHammer builds the whole volume's materials itself, subpixel averaging included. fdtdx uses `precomputed_inv_permittivities` / `precomputed_electric_conductivity` as given (no second full-grid array, no rescaling) and solves sources against them. It refuses fdtdx subpixel smoothing (which would average twice), other material objects (which would be ignored) and dispersive materials. | `src/fdtdx/fdtd/initialization.py` | `tests/integration/xanadu/test_precomputed_materials.py` |
| 2 | Mixed precision: `SimulationConfig.material_dtype` | SilverHammer runs under jax x64 with float64 fields, PML, sources, mode solves and detectors, and float32 full-grid material arrays (permittivity, permeability, conductivity) for half the memory. `material_dtype` defaults to `dtype`, so upstream behavior is unchanged unless it is set. Precomputed arrays must match it. Also fixes upstream's PML `sigma_end`, which became a strongly typed float64 under x64 and broke float32 runs. | `src/fdtdx/config.py`, `src/fdtdx/fdtd/initialization.py`, `src/fdtdx/objects/boundaries/perfectly_matched_layer.py` | `tests/integration/xanadu/test_mixed_precision.py` |
