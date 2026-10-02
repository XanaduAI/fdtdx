"""Mode fields from compute_mode must sit on the FDTD Yee grid.

The checks apply fdtdx's own curl operators to the returned mode and test the
longitudinal components of Faraday's and Ampere's laws,

    curl_E[p] / (dl k0) = i H[p]          curl_H[p] / (dl k0) = -i eps_p E[p],

with H = eta0 H_SI as compute_mode returns it.  A component placed half a cell
off its Yee position breaks these at O(1).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from fdtdx.core.physics.curl import curl_E, curl_H
from fdtdx.core.physics.modes import MODE_SOLVER, compute_mode

WAVELENGTH = 1.55e-6
FREQUENCY = 299792458.0 / WAVELENGTH
K0 = 2 * np.pi / WAVELENGTH
WIDTH = 4.0e-6
# Fields come back as complex64 unless jax x64 is enabled.
EXACT = 1e-4


def graded_profile(u, v):
    """Smooth index bump, asymmetric in both transverse directions."""
    bump = np.exp(-(((u - 0.1e-6) / 0.7e-6) ** 2) - ((v + 0.1e-6) / 0.5e-6) ** 2)
    tilt = (1 + 0.4 * np.tanh(u / 0.5e-6)) * (1 + 0.6 * np.tanh(v / 0.4e-6))
    return 2.1 + 4.0 * bump * tilt


def yee_inv_permittivity(axis, step):
    """(3, ...) inverse permittivity with each component sampled at its own E Yee position."""
    n = int(round(WIDTH / step))
    a, b = [d for d in range(3) if d != axis]
    nodes = -WIDTH / 2 + step * np.arange(n)
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    components = [
        np.expand_dims(1.0 / graded_profile(u + 0.5 * step * (c == a), v + 0.5 * step * (c == b)), axis)
        for c in range(3)
    ]
    return jnp.asarray(np.stack(components), dtype=jnp.float32)


def longitudinal_residuals(axis, direction, step):
    """Relative max residuals of the longitudinal Faraday and Ampere relations."""
    inv_eps = yee_inv_permittivity(axis, step)
    E, H, _ = compute_mode(FREQUENCY, inv_eps, 1.0, step, direction)

    no_psi = tuple((jnp.zeros((0, 0, 0)), jnp.zeros((0, 0, 0))) for _ in range(6))
    ones = (1.0,) * 6
    curl_e, _ = curl_E(None, E, no_psi, ones, ones, ones, simulate_boundaries=False)
    curl_h, _ = curl_H(None, H, no_psi, ones, ones, ones, simulate_boundaries=False)

    e_p, h_p = np.asarray(E[axis]), np.asarray(H[axis])
    d_e = e_p / np.asarray(inv_eps[axis])
    faraday = np.asarray(curl_e[axis]) / (step * K0) - 1j * h_p
    ampere = np.asarray(curl_h[axis]) / (step * K0) + 1j * d_e

    # Tangential E is pinned to zero on the min-side PEC walls.
    interior = tuple(slice(None) if d == axis else slice(1, None) for d in range(3))
    return np.abs(faraday).max() / np.abs(h_p).max(), np.abs(ampere[interior]).max() / np.abs(d_e).max()


@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_yee_mode_satisfies_fdtd_curl_equations(axis, direction, monkeypatch):
    monkeypatch.setenv("ELEANORMODES_SOLVER", "yee")
    faraday, ampere = longitudinal_residuals(axis, direction, 0.16e-6)
    assert faraday < EXACT
    assert ampere < EXACT


@pytest.mark.skipif(MODE_SOLVER != "eleanormodes", reason="collocated engine is eleanormodes-only")
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_collocated_mode_is_aligned_with_yee_grid(axis, direction, monkeypatch):
    """Collocated E is the Yee Ampere image of its H; Faraday holds up to a converging error."""
    monkeypatch.setenv("ELEANORMODES_SOLVER", "collocated")
    monkeypatch.setenv("ELEANORMODES_SAMPLING", "yee")
    coarse_faraday, coarse_ampere = longitudinal_residuals(axis, direction, 0.16e-6)
    fine_faraday, fine_ampere = longitudinal_residuals(axis, direction, 0.08e-6)

    assert max(coarse_ampere, fine_ampere) < EXACT
    assert fine_faraday < 0.05
    assert coarse_faraday / fine_faraday > 2.5
