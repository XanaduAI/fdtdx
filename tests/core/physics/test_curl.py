import jax.numpy as jnp

from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields


def test_interpolate_fields_basic():
    """Test basic interpolation functionality with simple field configuration."""
    # Create simple test fields
    E_field = jnp.ones((3, 5, 5, 5))
    H_field = jnp.ones((3, 5, 5, 5)) * 0.5

    E_interp, H_interp = interpolate_fields(E_field, H_field)

    # Check output shapes
    assert E_interp.shape == (3, 5, 5, 5)
    assert H_interp.shape == (3, 5, 5, 5)

    # Check that interpolated values are reasonable
    assert jnp.all(E_interp[2] == 1.0)  # E_z should remain unchanged
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_periodic_boundaries():
    """Test interpolation with periodic boundary conditions."""
    # Create test fields with gradients
    x = jnp.linspace(0, 2 * jnp.pi, 6)
    y = jnp.linspace(0, 2 * jnp.pi, 6)
    z = jnp.linspace(0, 2 * jnp.pi, 6)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    E_field = jnp.stack([jnp.sin(X), jnp.cos(Y), jnp.sin(Z)], axis=0)
    H_field = jnp.stack([jnp.cos(X), jnp.sin(Y), jnp.cos(Z)], axis=0)

    # Test with all periodic boundaries
    E_interp, H_interp = interpolate_fields(E_field, H_field, periodic_axes=(True, True, True))

    assert E_interp.shape == (3, 6, 6, 6)
    assert H_interp.shape == (3, 6, 6, 6)
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_mixed_boundaries():
    """Test interpolation with mixed periodic and PEC boundary conditions."""
    E_field = jnp.ones((3, 6, 6, 6))
    H_field = jnp.ones((3, 6, 6, 6)) * 2.0

    # Test with periodic x, PEC y,z
    E_interp, H_interp = interpolate_fields(E_field, H_field, periodic_axes=(True, False, False))

    assert E_interp.shape == (3, 6, 6, 6)
    assert H_interp.shape == (3, 6, 6, 6)
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_zero_fields():
    """Test interpolation with zero input fields."""
    E_field = jnp.zeros((3, 4, 4, 4))
    H_field = jnp.zeros((3, 4, 4, 4))

    E_interp, H_interp = interpolate_fields(E_field, H_field)

    assert E_interp.shape == (3, 4, 4, 4)
    assert H_interp.shape == (3, 4, 4, 4)
    assert jnp.allclose(E_interp, 0.0)
    assert jnp.allclose(H_interp, 0.0)


def test_curl_E_uniform_field():
    """Test curl_E with uniform electric field (should give zero curl)."""
    n = 5
    E = jnp.ones((3, n, n, n))
    psi_H = jnp.zeros((6, n, n, n))
    # Use precomputed PML coefficients: a=0, b=1 means no PML effect
    pml_a_H = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_E(E, psi_H, pml_a_H, pml_b_H, inv_kappa_E, True, periodic_axes=(True, True, True))

    assert curl_result.shape == (3, n, n, n)
    assert jnp.allclose(curl_result, 0.0, atol=1e-10)


def test_curl_E_linear_field():
    """Test curl_E with a linear field that has known curl."""
    # Create a field with E_x = y, E_y = -x, E_z = 0
    # This should give curl = (0, 0, -2)
    nx, ny, nz = 6, 6, 6
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    E = jnp.stack(
        [
            Y.astype(float),  # E_x = y
            -X.astype(float),  # E_y = -x
            jnp.zeros_like(X, dtype=float),  # E_z = 0
        ],
        axis=0,
    )
    psi_H = jnp.zeros((6, nx, ny, nz))
    pml_a_H = (jnp.zeros(nx), jnp.zeros(ny), jnp.zeros(nz))
    pml_b_H = (jnp.ones(nx), jnp.ones(ny), jnp.ones(nz))
    inv_kappa_E = (jnp.ones(nx), jnp.ones(ny), jnp.ones(nz))

    curl_result, _ = curl_E(
        E,
        psi_H,
        pml_a_H,
        pml_b_H,
        inv_kappa_E,
        True,
        periodic_axes=(True, True, True),
    )

    assert curl_result.shape == (3, nx, ny, nz)
    # The z-component should be approximately -2 (discrete approximation)
    assert jnp.allclose(curl_result[2][:-1, :-1], -2.0, atol=0.1)


def test_curl_E_periodic_boundaries():
    """Test curl_E with periodic boundary conditions."""
    # Create sinusoidal field
    n = 8
    x = jnp.linspace(0, 2 * jnp.pi, n)
    y = jnp.linspace(0, 2 * jnp.pi, n)
    z = jnp.linspace(0, 2 * jnp.pi, n)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    E = jnp.stack([jnp.sin(Y), jnp.cos(X), jnp.sin(Z)], axis=0)
    psi_H = jnp.zeros((6, n, n, n))
    pml_a_H = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_E(E, psi_H, pml_a_H, pml_b_H, inv_kappa_E, True, periodic_axes=(True, True, True))

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))


def test_curl_E_zero_field():
    """Test curl_E with zero electric field."""
    n = 4
    E = jnp.zeros((3, n, n, n))
    psi_H = jnp.zeros((6, n, n, n))
    pml_a_H = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_E(
        E,
        psi_H,
        pml_a_H,
        pml_b_H,
        inv_kappa_E,
        True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.allclose(curl_result, 0.0)


def test_curl_H_uniform_field():
    """Test curl_H with uniform magnetic field (should give zero curl)."""
    n = 5
    H = jnp.ones((3, n, n, n)) * 2.0
    psi_E = jnp.zeros((6, n, n, n))
    pml_a_E = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_H(H, psi_E, pml_a_E, pml_b_E, inv_kappa_H, True, periodic_axes=(True, True, True))

    assert curl_result.shape == (3, n, n, n)
    assert jnp.allclose(curl_result, 0.0, atol=1e-10)


def test_curl_H_linear_field():
    """Test curl_H with a linear field that has known curl."""
    # Create a field with H_x = z, H_y = 0, H_z = -x
    # This should give curl = (0, 2, 0)
    nx, ny, nz = 6, 6, 6
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    H = jnp.stack(
        [
            Z.astype(float),  # H_x = z
            jnp.zeros_like(Y, dtype=float),  # H_y = 0
            -X.astype(float),  # H_z = -x
        ],
        axis=0,
    )
    psi_E = jnp.zeros((6, nx, ny, nz))
    pml_a_E = (jnp.zeros(nx), jnp.zeros(ny), jnp.zeros(nz))
    pml_b_E = (jnp.ones(nx), jnp.ones(ny), jnp.ones(nz))
    inv_kappa_H = (jnp.ones(nx), jnp.ones(ny), jnp.ones(nz))

    curl_result, _ = curl_H(H, psi_E, pml_a_E, pml_b_E, inv_kappa_H, True, periodic_axes=(True, True, True))

    assert curl_result.shape == (3, nx, ny, nz)
    # The y-component should be approximately 2 (discrete approximation)
    assert jnp.allclose(curl_result[1][1:-1, 1:-1, 1:-1], 2.0, atol=0.1)


def test_curl_H_periodic_boundaries():
    """Test curl_H with periodic boundary conditions."""
    # Create sinusoidal field
    n = 8
    x = jnp.linspace(0, 2 * jnp.pi, n)
    y = jnp.linspace(0, 2 * jnp.pi, n)
    z = jnp.linspace(0, 2 * jnp.pi, n)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    H = jnp.stack([jnp.cos(Y), jnp.sin(X), jnp.cos(Z)], axis=0)
    psi_E = jnp.zeros((6, n, n, n))
    pml_a_E = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_H(H, psi_E, pml_a_E, pml_b_E, inv_kappa_H, True, periodic_axes=(True, True, True))

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))


def test_curl_H_zero_field():
    """Test curl_H with zero magnetic field."""
    n = 4
    H = jnp.zeros((3, n, n, n))
    psi_E = jnp.zeros((6, n, n, n))
    pml_a_E = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    curl_result, _ = curl_H(
        H,
        psi_E,
        pml_a_E,
        pml_b_E,
        inv_kappa_H,
        True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.allclose(curl_result, 0.0)


def test_curl_reciprocity():
    """Test that curl operations are consistent with Maxwell's equations structure."""
    # Create a simple test field
    n = 6
    E = jnp.ones((3, n, n, n))
    E = E.at[0].set(jnp.sin(jnp.linspace(0, jnp.pi, n)).reshape(-1, 1, 1))
    E = E.at[1].set(jnp.cos(jnp.linspace(0, jnp.pi, n)).reshape(1, -1, 1))

    psi_E = jnp.zeros((6, n, n, n))
    psi_H = jnp.zeros((6, n, n, n))
    pml_a_E = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    pml_a_H = (jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pml_b_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_E = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    inv_kappa_H = (jnp.ones(n), jnp.ones(n), jnp.ones(n))

    # Apply curl_E then curl_H
    curl_E_result, _ = curl_E(E, psi_H, pml_a_H, pml_b_H, inv_kappa_E, True, periodic_axes=(True, True, True))
    double_curl, _ = curl_H(curl_E_result, psi_E, pml_a_E, pml_b_E, inv_kappa_H, True, periodic_axes=(True, True, True))

    assert curl_E_result.shape == (3, n, n, n)
    assert double_curl.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_E_result))
    assert jnp.all(jnp.isfinite(double_curl))
