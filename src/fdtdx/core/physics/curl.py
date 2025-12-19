import jax
import jax.numpy as jnp


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
    E: jax.Array,
    psi_H: jax.Array,
    pml_a_H: tuple[jax.Array, jax.Array, jax.Array],
    pml_b_H: tuple[jax.Array, jax.Array, jax.Array],
    inv_kappa_E: tuple[jax.Array, jax.Array, jax.Array],
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    Args:
        E (jax.Array): Electric field to take the curl of. A 4D tensor representing the E-type field
            located on the edges of the grid cell (integer gridpoints).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_H (jax.Array): Auxiliary field for the magnetic field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        pml_a_H (tuple): Precomputed PML 'a' coefficients for H-field update.
            Tuple of 3 1D arrays: (a_H_x, a_H_y, a_H_z)
        pml_b_H (tuple): Precomputed PML 'b' coefficients for H-field update.
            Tuple of 3 1D arrays: (b_H_x, b_H_y, b_H_z)
        inv_kappa_E (tuple): Precomputed inverse kappa for E-field curl calculation.
            Tuple of 3 1D arrays: (inv_kappa_E_x, inv_kappa_E_y, inv_kappa_E_z)
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
    a_H_x = pml_a_H[0][:, None, None]
    a_H_y = pml_a_H[1][None, :, None]
    a_H_z = pml_a_H[2][None, None, :]
    b_H_x = pml_b_H[0][:, None, None]
    b_H_y = pml_b_H[1][None, :, None]
    b_H_z = pml_b_H[2][None, None, :]

    inv_kappa_E_x = inv_kappa_E[0][:, None, None]
    inv_kappa_E_y = inv_kappa_E[1][None, :, None]
    inv_kappa_E_z = inv_kappa_E[2][None, None, :]

    if simulate_boundaries:
        # Update auxiliary fields using precomputed PML coefficients
        psi_Hxy = b_H_y * psi_Hxy + a_H_y * dyEz
        psi_Hxz = b_H_z * psi_Hxz + a_H_z * dzEy
        psi_Hyz = b_H_z * psi_Hyz + a_H_z * dzEx
        psi_Hyx = b_H_x * psi_Hyx + a_H_x * dxEz
        psi_Hzx = b_H_x * psi_Hzx + a_H_x * dxEy
        psi_Hzy = b_H_y * psi_Hzy + a_H_y * dyEx

    psi_H_updated = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)

    curl_x = (inv_kappa_E_y * dyEz + psi_Hxy) - (inv_kappa_E_z * dzEy + psi_Hxz)
    curl_y = (inv_kappa_E_z * dzEx + psi_Hyz) - (inv_kappa_E_x * dxEz + psi_Hyx)
    curl_z = (inv_kappa_E_x * dxEy + psi_Hzx) - (inv_kappa_E_y * dyEx + psi_Hzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_H_updated


def curl_H(
    H: jax.Array,
    psi_E: jax.Array,
    pml_a_E: tuple[jax.Array, jax.Array, jax.Array],
    pml_b_E: tuple[jax.Array, jax.Array, jax.Array],
    inv_kappa_H: tuple[jax.Array, jax.Array, jax.Array],
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    Args:
        H (jax.Array): Magnetic field to take the curl of. A 4D tensor representing the H-type field
            located on the faces of the grid (half-integer grid points).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_E (jax.Array): Auxiliary field for the electric field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        pml_a_E (tuple): Precomputed PML 'a' coefficients for E-field update.
            Tuple of 3 1D arrays: (a_E_x, a_E_y, a_E_z)
        pml_b_E (tuple): Precomputed PML 'b' coefficients for E-field update.
            Tuple of 3 1D arrays: (b_E_x, b_E_y, b_E_z)
        inv_kappa_H (tuple): Precomputed inverse kappa for H-field curl calculation.
            Tuple of 3 1D arrays: (inv_kappa_H_x, inv_kappa_H_y, inv_kappa_H_z)
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
    a_E_x = pml_a_E[0][:, None, None]
    a_E_y = pml_a_E[1][None, :, None]
    a_E_z = pml_a_E[2][None, None, :]
    b_E_x = pml_b_E[0][:, None, None]
    b_E_y = pml_b_E[1][None, :, None]
    b_E_z = pml_b_E[2][None, None, :]

    inv_kappa_H_x = inv_kappa_H[0][:, None, None]
    inv_kappa_H_y = inv_kappa_H[1][None, :, None]
    inv_kappa_H_z = inv_kappa_H[2][None, None, :]

    if simulate_boundaries:
        # Update auxiliary fields using precomputed PML coefficients
        psi_Exy = b_E_y * psi_Exy + a_E_y * dyHz
        psi_Exz = b_E_z * psi_Exz + a_E_z * dzHy
        psi_Eyz = b_E_z * psi_Eyz + a_E_z * dzHx
        psi_Eyx = b_E_x * psi_Eyx + a_E_x * dxHz
        psi_Ezx = b_E_x * psi_Ezx + a_E_x * dxHy
        psi_Ezy = b_E_y * psi_Ezy + a_E_y * dyHx

    psi_E_updated = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)

    curl_x = (inv_kappa_H_y * dyHz + psi_Exy) - (inv_kappa_H_z * dzHy + psi_Exz)
    curl_y = (inv_kappa_H_z * dzHx + psi_Eyz) - (inv_kappa_H_x * dxHz + psi_Eyx)
    curl_z = (inv_kappa_H_x * dxHy + psi_Ezx) - (inv_kappa_H_y * dyHx + psi_Ezy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_E_updated
