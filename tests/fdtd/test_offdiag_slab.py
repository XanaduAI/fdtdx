from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.fdtd.update import OFFDIAG_ENTRIES, update_E, update_E_reverse, update_H, update_H_reverse
from fdtdx.objects.boundaries.periodic import PeriodicBoundary

SHAPE = (7, 6, 12)
Z_START, NZ = 3, 6


class Arrays:
    """Bare container with the attributes the E and H updates read, and the at[...].set / aset they write with."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def at(self):
        container = self

        class _Setter:
            def __getitem__(self, name):
                class _Set:
                    def set(self, value):
                        return container.aset(name, value)

                return _Set()

        return _Setter()

    def aset(self, name, value):
        return Arrays(**{**self.__dict__, name: value})


def _materials(seed):
    """Diagonal entries everywhere and off-diagonal entries inside the slab, zero on its first and last layer."""
    rng = np.random.default_rng(seed)
    diag = rng.uniform(0.3, 1.0, (3, *SHAPE)).astype(np.float32)
    offdiag = np.zeros((6, *SHAPE[:2], NZ), np.float32)
    offdiag[..., 1:-1] = rng.uniform(-0.2, 0.2, (6, *SHAPE[:2], NZ - 2))
    full = np.zeros((3, 3, *SHAPE), np.float32)
    for i in range(3):
        full[i, i] = diag[i]
    for entry, (i, j) in zip(offdiag, OFFDIAG_ENTRIES):
        full[i, j, ..., Z_START : Z_START + NZ] = entry
    return jnp.asarray(diag), jnp.asarray(offdiag), jnp.asarray(full.reshape(9, *SHAPE))


def _objects(periodic_axes):
    boundaries = []
    for axis in periodic_axes:
        boundary = Mock(spec=PeriodicBoundary)
        boundary.axis = axis
        boundaries.append(boundary)
    objects = Mock()
    objects.boundary_objects = boundaries
    objects.sources = []
    return objects


@pytest.mark.parametrize("periodic_axes", [(), (0,), (0, 1)])
@pytest.mark.parametrize("electric", [True, False])
def test_slab_matches_full_tensor(periodic_axes, electric):
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.normal(size=(3, *SHAPE)).astype(np.float32))
    curl = jnp.asarray(rng.normal(size=(3, *SHAPE)).astype(np.float32))
    diag, offdiag, full = _materials(1)
    config = Mock(courant_number=0.5)
    objects = _objects(periodic_axes)
    common = dict(E=field, H=field, psi_E=None, psi_H=None, alpha=None, kappa=None, sigma=None,
                  electric_conductivity=None, magnetic_conductivity=None, offdiag_z_start=Z_START)
    if electric:
        update, curl_name, name = update_E, "curl_H", "E"
        full_arrays = Arrays(**common, inv_permittivities=full, inv_permeabilities=1.0,
                             offdiag_inv_permittivities=None, offdiag_inv_permeabilities=None)
        slab_arrays = Arrays(**common, inv_permittivities=diag, inv_permeabilities=1.0,
                             offdiag_inv_permittivities=offdiag, offdiag_inv_permeabilities=None)
    else:
        update, curl_name, name = update_H, "curl_E", "H"
        full_arrays = Arrays(**common, inv_permittivities=diag, inv_permeabilities=full,
                             offdiag_inv_permittivities=None, offdiag_inv_permeabilities=None)
        slab_arrays = Arrays(**common, inv_permittivities=diag, inv_permeabilities=diag,
                             offdiag_inv_permittivities=None, offdiag_inv_permeabilities=offdiag)

    with patch(f"fdtdx.fdtd.update.{curl_name}", return_value=(curl, None)):
        expected = getattr(update(jnp.array(0), full_arrays, objects, config, simulate_boundaries=False), name)
        result = getattr(update(jnp.array(0), slab_arrays, objects, config, simulate_boundaries=False), name)

    assert not np.allclose(np.asarray(expected), np.asarray(field + 0.5 * curl * diag if electric else field))
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_slab_term_jits_with_traced_start():
    rng = np.random.default_rng(2)
    curl = jnp.asarray(rng.normal(size=(3, *SHAPE)).astype(np.float32))
    _, offdiag, _ = _materials(3)
    from fdtdx.fdtd.update import offdiag_slab_term

    eager = offdiag_slab_term(offdiag, curl, Z_START, (False, False, False), electric=True)
    jitted = jax.jit(lambda start: offdiag_slab_term(offdiag, curl, start, (False, False, False), electric=True))
    np.testing.assert_allclose(np.asarray(jitted(jnp.int32(Z_START))), np.asarray(eager), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("update, name", [(update_E_reverse, "offdiag_inv_permittivities"),
                                          (update_H_reverse, "offdiag_inv_permeabilities")])
def test_reverse_update_rejects_slab(update, name):
    arrays = Arrays(**{name: jnp.zeros((6, *SHAPE[:2], NZ))})
    with pytest.raises(NotImplementedError):
        update(jnp.array(0), arrays, _objects(()), Mock(courant_number=0.5))
