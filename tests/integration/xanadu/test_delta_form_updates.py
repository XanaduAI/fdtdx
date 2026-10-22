"""XANADU: delta-form lossy E/H updates.

Upstream writes the lossy update as ``((1 - half) * E + K) / (1 + half)``. With float32 materials
(SilverHammer's ``material_dtype``) ``1 - half`` is formed in float32 and rounds to exactly 1 once ``half`` drops
below ~6e-8, so weak losses vanish even though the fields are float64. The delta form
``E + (K - 2 * half * E) / (1 + half)`` never subtracts ``half`` from a bare 1. These tests pin that weak losses
decay as they should and that the reverse updates stay exact inverses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.fdtd.update import update_E, update_E_reverse, update_H, update_H_reverse

_SPACING = 25e-9
_SHAPE = (8, 8, 8)
_PERMITTIVITY = (2.0, 2.2, 2.4)
_CENTER = (slice(None), 4, 4, 4)


def _place(electric_conductivity: float, magnetic_conductivity: float):
    """A lossy, diagonally anisotropic volume with float64 fields and float32 materials, nothing else."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING), time=1e-15, dtype=jnp.float64, material_dtype=jnp.float32
    )
    material = fdtdx.Material(
        permittivity=_PERMITTIVITY,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=_SHAPE, material=material)
    container, arrays, _, config, _ = fdtdx.place_objects(
        object_list=[volume], config=config, constraints=[], key=jax.random.PRNGKey(0)
    )
    assert arrays.electric_conductivity.dtype == arrays.inv_permittivities.dtype == jnp.float32
    return container, arrays, config


def _with_fields(arrays, E, H):
    return arrays.aset("fields->E", jnp.asarray(E, jnp.float64)).aset("fields->H", jnp.asarray(H, jnp.float64))


def _half_E(arrays, config):
    # The same product update_E forms, in float32 like the materials.
    return config.courant_number * arrays.electric_conductivity * fdtdx.constants.eta0 * arrays.inv_permittivities / 2


def _half_H(arrays, config):
    return config.courant_number * arrays.magnetic_conductivity / fdtdx.constants.eta0 / 2


def test_weak_losses_decay():
    with jax.enable_x64(True):
        # Conductivities chosen so that half ~ 1e-9 in both updates: far below float32's resolution at 1.
        container, arrays, config = _place(electric_conductivity=7e-4, magnetic_conductivity=50.0)
        half_E, half_H = _half_E(arrays, config), _half_H(arrays, config)
        for half in (half_E, half_H):
            assert half.dtype == jnp.float32
            assert jnp.all(1 - half == 1), "half must be small enough that the 1 - half form loses it"
            assert jnp.all(half > 1e-10)

        ones, zeros = np.ones((3, *_SHAPE)), np.zeros((3, *_SHAPE))
        step = jnp.asarray(0)

        # H = 0, so curl(H) = 0 and E only decays: E1 = E0 * (1 - half) / (1 + half).
        E1 = update_E(step, _with_fields(arrays, ones, zeros), container, config, True).fields.E
        h = np.broadcast_to(np.asarray(half_E, np.float64)[_CENTER], (3,))
        np.testing.assert_allclose(1 - np.asarray(E1[_CENTER]), 2 * h / (1 + h), rtol=1e-6)

        H1 = update_H(step, _with_fields(arrays, zeros, ones), container, config, True).fields.H
        h = np.broadcast_to(np.asarray(half_H, np.float64)[_CENTER], (3,))
        np.testing.assert_allclose(1 - np.asarray(H1[_CENTER]), 2 * h / (1 + h), rtol=1e-6)


@pytest.mark.parametrize("field", ["E", "H"])
def test_reverse_updates_invert_forward_updates(field):
    with jax.enable_x64(True):
        container, arrays, config = _place(electric_conductivity=2e4, magnetic_conductivity=1e8)
        assert float(_half_E(arrays, config).max()) > 1e-3 and float(_half_H(arrays, config).max()) > 1e-3
        rng = np.random.default_rng(0)
        arrays = _with_fields(arrays, rng.standard_normal((3, *_SHAPE)), rng.standard_normal((3, *_SHAPE)))
        step = jnp.asarray(0)

        if field == "E":
            forward = update_E(step, arrays, container, config, True)
            back = update_E_reverse(step, forward, container, config)
        else:
            forward = update_H(step, arrays, container, config, True)
            back = update_H_reverse(step, forward, container, config)

        before, after = getattr(arrays.fields, field), getattr(forward.fields, field)
        assert float(jnp.abs(after - before).max()) > 1e-3
        np.testing.assert_allclose(np.asarray(getattr(back.fields, field)), np.asarray(before), rtol=0, atol=1e-12)
