# Xanadu patches

This branch is an upstream fdtdx release ([ymahlau/fdtdx](https://github.com/ymahlau/fdtdx)) plus the short series
of Xanadu commits below, which SilverHammer needs inside fdtdx itself. Everything else Xanadu-specific (subpixel
averaging, mode caching, dtype choices, buffer donation, plotting, S3) lives in SilverHammer.

- **Branch naming:** `xanadu/<upstream tag>`, for example `xanadu/v0.6.3` is `v0.6.3` plus the patches below.
- **Syncing:** a sync replays the series onto the next upstream tag as a new branch. Existing branches are never
  rewritten, so commit pins in SilverHammer keep working.
- **Markers:** every edit inside an upstream file carries a `XANADU:` comment, so conflicts are quick to find.
- **Tests:** Xanadu tests live in `tests/**/xanadu/`. The upstream suite (`-m "unit or integration"`) must stay green.

| # | Patch | Why | Files | Tests |
|---|---|---|---|---|
| 1 | Precomputed materials in `place_objects` | SilverHammer builds the whole volume's materials itself, subpixel averaging included. fdtdx uses `precomputed_inv_permittivities` / `precomputed_electric_conductivity` as given (no second full-grid array, no rescaling) and solves sources against them. It refuses fdtdx subpixel smoothing (which would average twice), other material objects (which would be ignored) and dispersive materials. | `src/fdtdx/fdtd/initialization.py` | `tests/integration/xanadu/test_precomputed_materials.py` |
