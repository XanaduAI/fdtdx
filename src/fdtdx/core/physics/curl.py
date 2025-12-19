import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0


def interpolate_fields(
    E_field: jax.Array,
    H_field: jax.Array,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Interpolates E and H fields onto E_z in a FDTD grid with PEC/periodic boundary conditions.

    Performs spatial interpolation of the electric and magnetic fields to align them
    onto the same grid points as E_z. This is necessary because E and H fields are
    naturally staggered in the Yee grid.

    Args:
        E_field (jax.Array): 4D tensor representing the electric field.
                Dimensions are (width, depth, height, direction).
        H_field (jax.Array): 4D tensor representing the magnetic field.
                Dimensions are (width, depth, height, direction).
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        tuple[jax.Array, jax.Array]: A tuple (E_interp, H_interp) containing:
            - E_interp: Interpolated electric field as 4D tensor
            - H_interp: Interpolated magnetic field as 4D tensor

    Note:
        Uses PEC (Perfect Electric Conductor) boundary conditions where fields
        at boundaries are zero, unless periodic boundaries are specified.
    """
    # Apply boundary conditions: PEC (zero) or periodic for each axis separately
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        E_field = jnp.pad(E_field, pad_width, mode=pad_mode)
        H_field = jnp.pad(H_field, pad_width, mode=pad_mode)

    E_x, E_y, E_z = E_field[0], E_field[1], E_field[2]
    H_x, H_y, H_z = H_field[0], H_field[1], H_field[2]

    E_x = (E_x[1:-1, 1:-1, 1:-1] + E_x[1:-1, 1:-1, :-2] + E_x[2:, 1:-1, 1:-1] + E_x[2:, 1:-1, :-2]) / 4.0
    E_y = (E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1] + E_y[2:, 1:-1, 1:-1] + E_y[2:, :-2, 1:-1]) / 4.0
    E_z = E_z[1:-1, 1:-1, 1:-1]  # leave as is since we project onto the E_z

    H_x = (H_x[1:-1, 2:, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0
    H_y = (H_y[1:-1, 1:-1, 2:] + H_y[1:-1, 1:-1, :-2]) / 2.0
    H_z = (
        H_z[:-2, 2:, 2:]
        + H_z[:-2, 2:, :-2]
        + H_z[:-2, :-2, 2:]
        + H_z[:-2, :-2, :-2]
        + H_z[2:, 2:, 2:]
        + H_z[2:, 2:, :-2]
        + H_z[2:, :-2, 2:]
        + H_z[2:, :-2, :-2]
    ) / 8.0

    # Constructing the interpolated fields
    E_interp = jnp.stack([E_x, E_y, E_z], axis=0)
    H_interp = jnp.stack([H_x, H_y, H_z], axis=0)

    return E_interp, H_interp


def curl_E(
    config: SimulationConfig,
    E: jax.Array,
    psi_H: jax.Array,
    alpha: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    kappa: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    sigma: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        E (jax.Array): Electric field to take the curl of. A 4D tensor representing the E-type field
            located on the edges of the grid cell (integer gridpoints).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_H (jax.Array): Auxiliary field for the magnetic field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (tuple): Tuple of 6 1D alpha arrays for PML:
            (alpha_E_x, alpha_E_y, alpha_E_z, alpha_H_x, alpha_H_y, alpha_H_z)
        kappa (tuple): Tuple of 6 1D kappa arrays for PML:
            (kappa_E_x, kappa_E_y, kappa_E_z, kappa_H_x, kappa_H_y, kappa_H_z)
        sigma (tuple): Tuple of 6 1D sigma arrays for PML:
            (sigma_E_x, sigma_E_y, sigma_E_z, sigma_H_x, sigma_H_y, sigma_H_z)
        simulate_boundaries (bool): Whether to simulate boundaries.
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        jax.Array: The curl of E - an H-type field located on the faces of the grid
                  (half-integer grid points). Has same shape as input (3, nx, ny, nz).
    """
    # Pad each axis separately based on boundary conditions
    E_pad = E
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        E_pad = jnp.pad(E_pad, pad_width, mode=pad_mode)

    dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Hxy = psi_H[0, :, :, :]
    psi_Hxz = psi_H[1, :, :, :]
    psi_Hyz = psi_H[2, :, :, :]
    psi_Hyx = psi_H[3, :, :, :]
    psi_Hzx = psi_H[4, :, :, :]
    psi_Hzy = psi_H[5, :, :, :]

    # Reshape 1D arrays for broadcasting:
    # - x-component arrays: (nx,) -> (nx, 1, 1)
    # - y-component arrays: (ny,) -> (1, ny, 1)
    # - z-component arrays: (nz,) -> (1, 1, nz)
    alpha_H_x = alpha[3][:, None, None]
    alpha_H_y = alpha[4][None, :, None]
    alpha_H_z = alpha[5][None, None, :]
    kappa_H_x = kappa[3][:, None, None]
    kappa_H_y = kappa[4][None, :, None]
    kappa_H_z = kappa[5][None, None, :]
    sigma_H_x = sigma[3][:, None, None]
    sigma_H_y = sigma[4][None, :, None]
    sigma_H_z = sigma[5][None, None, :]

    # Also reshape kappa for E-field (used in curl calculation)
    kappa_E_x = kappa[0][:, None, None]
    kappa_E_y = kappa[1][None, :, None]
    kappa_E_z = kappa[2][None, None, :]

    if simulate_boundaries:
        # Get H-field PML coefficients
        b_x = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_H_x / kappa_H_x + alpha_H_x)
            )
            + 1
        )
        b_y = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_H_y / kappa_H_y + alpha_H_y)
            )
            + 1
        )
        b_z = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_H_z / kappa_H_z + alpha_H_z)
            )
            + 1
        )

        a_x = (
            (b_x - 1.0)
            * sigma_H_x
            / (sigma_H_x + alpha_H_x * kappa_H_x)
            / kappa_H_x
        )
        a_y = (
            (b_y - 1.0)
            * sigma_H_y
            / (sigma_H_y + alpha_H_y * kappa_H_y)
            / kappa_H_y
        )
        a_z = (
            (b_z - 1.0)
            * sigma_H_z
            / (sigma_H_z + alpha_H_z * kappa_H_z)
            / kappa_H_z
        )

        a_x = jnp.nan_to_num(a_x, nan=0.0, posinf=0.0, neginf=0.0)
        a_y = jnp.nan_to_num(a_y, nan=0.0, posinf=0.0, neginf=0.0)
        a_z = jnp.nan_to_num(a_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Hxy = b_y * psi_Hxy + a_y * dyEz
        psi_Hxz = b_z * psi_Hxz + a_z * dzEy
        psi_Hyz = b_z * psi_Hyz + a_z * dzEx
        psi_Hyx = b_x * psi_Hyx + a_x * dxEz
        psi_Hzx = b_x * psi_Hzx + a_x * dxEy
        psi_Hzy = b_y * psi_Hzy + a_y * dyEx

    psi_H_updated = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)

    curl_x = (1.0 / kappa_E_y * dyEz + psi_Hxy) - (1.0 / kappa_E_z * dzEy + psi_Hxz)
    curl_y = (1.0 / kappa_E_z * dzEx + psi_Hyz) - (1.0 / kappa_E_x * dxEz + psi_Hyx)
    curl_z = (1.0 / kappa_E_x * dxEy + psi_Hzx) - (1.0 / kappa_E_y * dyEx + psi_Hzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_H_updated


def curl_H(
    config: SimulationConfig,
    H: jax.Array,
    psi_E: jax.Array,
    alpha: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    kappa: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    sigma: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        H (jax.Array): Magnetic field to take the curl of. A 4D tensor representing the H-type field
            located on the faces of the grid (half-integer grid points).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_E (jax.Array): Auxiliary field for the electric field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (tuple): Tuple of 6 1D alpha arrays for PML:
            (alpha_E_x, alpha_E_y, alpha_E_z, alpha_H_x, alpha_H_y, alpha_H_z)
        kappa (tuple): Tuple of 6 1D kappa arrays for PML:
            (kappa_E_x, kappa_E_y, kappa_E_z, kappa_H_x, kappa_H_y, kappa_H_z)
        sigma (tuple): Tuple of 6 1D sigma arrays for PML:
            (sigma_E_x, sigma_E_y, sigma_E_z, sigma_H_x, sigma_H_y, sigma_H_z)
        simulate_boundaries (bool): Whether to simulate boundaries.
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        jax.Array: The curl of H - an E-type field located on the edges of the grid
                  (integer grid points). Has same shape as input (3, nx, ny, nz).
    """
    # Pad each axis separately based on boundary conditions
    H_pad = H
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        H_pad = jnp.pad(H_pad, pad_width, mode=pad_mode)

    dyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Exy = psi_E[0, :, :, :]
    psi_Exz = psi_E[1, :, :, :]
    psi_Eyz = psi_E[2, :, :, :]
    psi_Eyx = psi_E[3, :, :, :]
    psi_Ezx = psi_E[4, :, :, :]
    psi_Ezy = psi_E[5, :, :, :]

    # Reshape 1D arrays for broadcasting:
    # - x-component arrays: (nx,) -> (nx, 1, 1)
    # - y-component arrays: (ny,) -> (1, ny, 1)
    # - z-component arrays: (nz,) -> (1, 1, nz)
    alpha_E_x = alpha[0][:, None, None]
    alpha_E_y = alpha[1][None, :, None]
    alpha_E_z = alpha[2][None, None, :]
    kappa_E_x = kappa[0][:, None, None]
    kappa_E_y = kappa[1][None, :, None]
    kappa_E_z = kappa[2][None, None, :]
    sigma_E_x = sigma[0][:, None, None]
    sigma_E_y = sigma[1][None, :, None]
    sigma_E_z = sigma[2][None, None, :]

    # Also reshape kappa for H-field (used in curl calculation)
    kappa_H_x = kappa[3][:, None, None]
    kappa_H_y = kappa[4][None, :, None]
    kappa_H_z = kappa[5][None, None, :]

    if simulate_boundaries:
        # Get E-field PML coefficients
        b_x = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_E_x / kappa_E_x + alpha_E_x)
            )
            + 1
        )
        b_y = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_E_y / kappa_E_y + alpha_E_y)
            )
            + 1
        )
        b_z = (
            jnp.expm1(
                -config.courant_number
                * config.resolution
                / c0
                / eps0
                * (sigma_E_z / kappa_E_z + alpha_E_z)
            )
            + 1
        )

        a_x = (
            (b_x - 1.0)
            * sigma_E_x
            / (sigma_E_x + alpha_E_x * kappa_E_x)
            / kappa_E_x
        )
        a_y = (
            (b_y - 1.0)
            * sigma_E_y
            / (sigma_E_y + alpha_E_y * kappa_E_y)
            / kappa_E_y
        )
        a_z = (
            (b_z - 1.0)
            * sigma_E_z
            / (sigma_E_z + alpha_E_z * kappa_E_z)
            / kappa_E_z
        )

        a_x = jnp.nan_to_num(a_x, nan=0.0, posinf=0.0, neginf=0.0)
        a_y = jnp.nan_to_num(a_y, nan=0.0, posinf=0.0, neginf=0.0)
        a_z = jnp.nan_to_num(a_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Exy = b_y * psi_Exy + a_y * dyHz
        psi_Exz = b_z * psi_Exz + a_z * dzHy
        psi_Eyz = b_z * psi_Eyz + a_z * dzHx
        psi_Eyx = b_x * psi_Eyx + a_x * dxHz
        psi_Ezx = b_x * psi_Ezx + a_x * dxHy
        psi_Ezy = b_y * psi_Ezy + a_y * dyHx

    psi_E_updated = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)

    curl_x = (1.0 / kappa_H_y * dyHz + psi_Exy) - (1.0 / kappa_H_z * dzHy + psi_Exz)
    curl_y = (1.0 / kappa_H_z * dzHx + psi_Eyz) - (1.0 / kappa_H_x * dxHz + psi_Eyx)
    curl_z = (1.0 / kappa_H_x * dxHy + psi_Ezx) - (1.0 / kappa_H_y * dyHx + psi_Ezy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_E_updated
