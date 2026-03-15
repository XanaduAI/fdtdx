import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0
from fdtdx.core.misc import pad_fields

# Each psi field has 6 components (xy, xz, yz, yx, zx, zy).
# Each component is non-zero only along one PML axis.
# This map gives the PML axis for each component index 0-5.
PSI_COMPONENT_AXIS = (1, 2, 2, 0, 0, 1)

# PML coefficient indices used for each psi component in curl_E (H-field PML):
PSI_H_COEFF_IDX = (4, 5, 5, 3, 3, 4)
# PML coefficient indices used for each psi component in curl_H (E-field PML):
PSI_E_COEFF_IDX = (1, 2, 2, 0, 0, 1)

SparsePsi = tuple[tuple[jax.Array, jax.Array], ...]
PMLCoeffs1D = tuple[jax.Array, ...]


def interpolate_fields(
    E_field: jax.Array,
    H_field: jax.Array,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Interpolates E and H fields onto E_z in a FDTD grid with PEC/periodic boundary conditions."""
    E_field = pad_fields(E_field, periodic_axes)
    H_field = pad_fields(H_field, periodic_axes)

    E_x, E_y, E_z = E_field[0], E_field[1], E_field[2]
    H_x, H_y, H_z = H_field[0], H_field[1], H_field[2]

    E_x = (E_x[1:-1,1:-1,1:-1] + E_x[:-2,1:-1,1:-1] + E_x[1:-1,1:-1,2:] + E_x[:-2,1:-1,2:]) / 4.0
    E_y = (E_y[1:-1,1:-1,1:-1] + E_y[1:-1,:-2,1:-1] + E_y[1:-1,1:-1,2:] + E_y[1:-1,:-2,2:]) / 4.0
    E_z = E_z[1:-1, 1:-1, 1:-1]
    H_x = (H_x[1:-1,1:-1,1:-1] + H_x[1:-1,:-2,1:-1]) / 2.0
    H_y = (H_y[1:-1,1:-1,1:-1] + H_y[:-2,1:-1,1:-1]) / 2.0
    H_z = (H_z[1:-1,1:-1,1:-1] + H_z[:-2,1:-1,1:-1] + H_z[1:-1,:-2,1:-1] + H_z[:-2,:-2,1:-1]
        + H_z[1:-1,1:-1,2:] + H_z[:-2,1:-1,2:] + H_z[1:-1,:-2,2:] + H_z[:-2,:-2,2:]) / 8.0

    E_interp = jnp.stack([E_x, E_y, E_z], axis=0)
    H_interp = jnp.stack([H_x, H_y, H_z], axis=0)
    return E_interp, H_interp


def _extract_pml_slab(field: jax.Array, axis: int, L: int, side: str) -> jax.Array:
    """Extract a PML slab from a full-volume field along an axis."""
    s = [slice(None)] * 3
    if side == "min":
        s[axis] = slice(0, L)
    else:
        s[axis] = slice(-L, None)
    return field[tuple(s)]


def _scatter_psi_component(
    curl_comp: jax.Array,
    psi_min: jax.Array,
    psi_max: jax.Array,
    axis: int,
    sign: float,
) -> jax.Array:
    """Scatter one sparse psi (min, max) pair into a full-volume curl component."""
    L_min = psi_min.shape[axis]
    L_max = psi_max.shape[axis]
    if L_min > 0:
        s = [slice(None)] * 3
        s[axis] = slice(0, L_min)
        curl_comp = curl_comp.at[tuple(s)].add(sign * psi_min)
    if L_max > 0:
        s = [slice(None)] * 3
        s[axis] = slice(-L_max, None)
        curl_comp = curl_comp.at[tuple(s)].add(sign * psi_max)
    return curl_comp


def _compute_pml_ab(
    alpha: PMLCoeffs1D,
    kappa: PMLCoeffs1D,
    sigma: PMLCoeffs1D,
    config: SimulationConfig,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Compute PML update coefficients a and b from 1D profiles (all 6 components)."""
    factor = -config.courant_number * config.resolution / c0 / eps0
    b_list = []
    a_list = []
    for i in range(6):
        b_i = jnp.expm1(factor * (sigma[i] / kappa[i] + alpha[i])) + 1
        a_i = jnp.nan_to_num(
            (b_i - 1.0) * sigma[i] / (sigma[i] + alpha[i] * kappa[i]) / kappa[i],
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        b_list.append(b_i)
        a_list.append(a_i)
    return b_list, a_list


def _update_sparse_psi(
    psi_min: jax.Array,
    psi_max: jax.Array,
    b_profile: jax.Array,
    a_profile: jax.Array,
    d_field: jax.Array,
    axis: int,
) -> tuple[jax.Array, jax.Array]:
    """Update one sparse psi component's min/max slabs."""
    L_min = psi_min.shape[axis]
    L_max = psi_max.shape[axis]

    if L_min > 0:
        d_min = _extract_pml_slab(d_field, axis, L_min, "min")
        b_min = _extract_pml_slab(b_profile, axis, L_min, "min")
        a_min = _extract_pml_slab(a_profile, axis, L_min, "min")
        psi_min = b_min * psi_min + a_min * d_min

    if L_max > 0:
        d_max = _extract_pml_slab(d_field, axis, L_max, "max")
        b_max = _extract_pml_slab(b_profile, axis, L_max, "max")
        a_max = _extract_pml_slab(a_profile, axis, L_max, "max")
        psi_max = b_max * psi_max + a_max * d_max

    return psi_min, psi_max


def curl_E(
    config: SimulationConfig,
    E: jax.Array,
    psi_H: SparsePsi,
    alpha: PMLCoeffs1D,
    kappa: PMLCoeffs1D,
    sigma: PMLCoeffs1D,
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, SparsePsi]:
    """Computes curl of E for the H-field update with memory-efficient PML.

    Args:
        config: Simulation configuration.
        E: Electric field (3, nx, ny, nz).
        psi_H: Sparse PML auxiliary fields. Tuple of 6 (min_slab, max_slab) pairs.
        alpha: 1D PML alpha profiles (tuple of 6).
        kappa: 1D PML kappa profiles (tuple of 6).
        sigma: 1D PML sigma profiles (tuple of 6).
        simulate_boundaries: Whether to update PML auxiliary fields.
        periodic_axes: Which axes are periodic.

    Returns:
        (curl (3, nx, ny, nz), updated psi_H)
    """
    E_pad = pad_fields(E, periodic_axes)

    #dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    #dzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    #dzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    #dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    #dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    #dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]
    dyEz = E_pad[2, 1:-1, 2:, 1:-1] - E_pad[2, 1:-1, 1:-1, 1:-1]
    dzEy = E_pad[1, 1:-1, 1:-1, 2:] - E_pad[1, 1:-1, 1:-1, 1:-1]
    dzEx = E_pad[0, 1:-1, 1:-1, 2:] - E_pad[0, 1:-1, 1:-1, 1:-1]
    dxEz = E_pad[2, 2:, 1:-1, 1:-1] - E_pad[2, 1:-1, 1:-1, 1:-1]
    dxEy = E_pad[1, 2:, 1:-1, 1:-1] - E_pad[1, 1:-1, 1:-1, 1:-1]
    dyEx = E_pad[0, 1:-1, 2:, 1:-1] - E_pad[0, 1:-1, 1:-1, 1:-1]

    # Derivative fields in the order matching psi components [xy, xz, yz, yx, zx, zy]
    d_fields = (dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)

    psi_H_list = list(psi_H)
    if simulate_boundaries:
        b, a = _compute_pml_ab(alpha, kappa, sigma, config)
        for i in range(6):
            axis = PSI_COMPONENT_AXIS[i]
            ci = PSI_H_COEFF_IDX[i]
            psi_H_list[i] = _update_sparse_psi(
                *psi_H[i], b[ci], a[ci], d_fields[i], axis,
            )

    psi_H_updated = tuple(psi_H_list)

    # Build curl: curl_x = (1/kappa[1]*dyEz + psi_Hxy) - (1/kappa[2]*dzEy + psi_Hxz)
    #             curl_y = (1/kappa[2]*dzEx + psi_Hyz) - (1/kappa[0]*dxEz + psi_Hyx)
    #             curl_z = (1/kappa[0]*dxEy + psi_Hzx) - (1/kappa[1]*dyEx + psi_Hzy)
    # kappa indices 0,1,2 correspond to x,y,z E-field PML
    curl_x = 1.0 / kappa[1] * dyEz - 1.0 / kappa[2] * dzEy
    curl_y = 1.0 / kappa[2] * dzEx - 1.0 / kappa[0] * dxEz
    curl_z = 1.0 / kappa[0] * dxEy - 1.0 / kappa[1] * dyEx

    # Scatter sparse psi contributions: +psi[0], -psi[1], +psi[2], -psi[3], +psi[4], -psi[5]
    curl_x = _scatter_psi_component(curl_x, *psi_H_updated[0], axis=1, sign=+1.0)
    curl_x = _scatter_psi_component(curl_x, *psi_H_updated[1], axis=2, sign=-1.0)
    curl_y = _scatter_psi_component(curl_y, *psi_H_updated[2], axis=2, sign=+1.0)
    curl_y = _scatter_psi_component(curl_y, *psi_H_updated[3], axis=0, sign=-1.0)
    curl_z = _scatter_psi_component(curl_z, *psi_H_updated[4], axis=0, sign=+1.0)
    curl_z = _scatter_psi_component(curl_z, *psi_H_updated[5], axis=1, sign=-1.0)

    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    return curl, psi_H_updated


def curl_H(
    config: SimulationConfig,
    H: jax.Array,
    psi_E: SparsePsi,
    alpha: PMLCoeffs1D,
    kappa: PMLCoeffs1D,
    sigma: PMLCoeffs1D,
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, SparsePsi]:
    """Computes curl of H for the E-field update with memory-efficient PML.

    Args:
        config: Simulation configuration.
        H: Magnetic field (3, nx, ny, nz).
        psi_E: Sparse PML auxiliary fields. Tuple of 6 (min_slab, max_slab) pairs.
        alpha: 1D PML alpha profiles (tuple of 6).
        kappa: 1D PML kappa profiles (tuple of 6).
        sigma: 1D PML sigma profiles (tuple of 6).
        simulate_boundaries: Whether to update PML auxiliary fields.
        periodic_axes: Which axes are periodic.

    Returns:
        (curl (3, nx, ny, nz), updated psi_E)
    """
    H_pad = pad_fields(H, periodic_axes)

    #dyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    #dzHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    #dzHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    #dxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    #dxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    #dyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1]
    dyHz = H_pad[2, 1:-1, :-2, 1:-1] - H_pad[2, 1:-1, 1:-1, 1:-1]
    dzHy = H_pad[1, 1:-1, 1:-1, :-2] - H_pad[1, 1:-1, 1:-1, 1:-1]
    dzHx = H_pad[0, 1:-1, 1:-1, :-2] - H_pad[0, 1:-1, 1:-1, 1:-1]
    dxHz = H_pad[2, :-2, 1:-1, 1:-1] - H_pad[2, 1:-1, 1:-1, 1:-1]
    dxHy = H_pad[1, :-2, 1:-1, 1:-1] - H_pad[1, 1:-1, 1:-1, 1:-1]
    dyHx = H_pad[0, 1:-1, :-2, 1:-1] - H_pad[0, 1:-1, 1:-1, 1:-1]

    d_fields = (dyHz, dzHy, dzHx, dxHz, dxHy, dyHx)

    psi_E_list = list(psi_E)
    if simulate_boundaries:
        b, a = _compute_pml_ab(alpha, kappa, sigma, config)
        for i in range(6):
            axis = PSI_COMPONENT_AXIS[i]
            ci = PSI_E_COEFF_IDX[i]
            psi_E_list[i] = _update_sparse_psi(
                *psi_E[i], b[ci], a[ci], d_fields[i], axis,
            )

    psi_E_updated = tuple(psi_E_list)

    curl_x = 1.0 / kappa[1] * dyHz - 1.0 / kappa[2] * dzHy
    curl_y = 1.0 / kappa[2] * dzHx - 1.0 / kappa[0] * dxHz
    curl_z = 1.0 / kappa[0] * dxHy - 1.0 / kappa[1] * dyHx

    curl_x = _scatter_psi_component(curl_x, *psi_E_updated[0], axis=1, sign=+1.0)
    curl_x = _scatter_psi_component(curl_x, *psi_E_updated[1], axis=2, sign=-1.0)
    curl_y = _scatter_psi_component(curl_y, *psi_E_updated[2], axis=2, sign=+1.0)
    curl_y = _scatter_psi_component(curl_y, *psi_E_updated[3], axis=0, sign=-1.0)
    curl_z = _scatter_psi_component(curl_z, *psi_E_updated[4], axis=0, sign=+1.0)
    curl_z = _scatter_psi_component(curl_z, *psi_E_updated[5], axis=1, sign=-1.0)

    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    return curl, psi_E_updated
