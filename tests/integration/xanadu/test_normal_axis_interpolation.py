"""XANADU: detector co-location along the normal axis only (``Detector.normal_axis_interpolation``).

Upstream's co-location averages every component onto the E_z point, transverse directions included, which smears
fields across waveguide walls. The Xanadu fork averaged only along the monitor normal (fork commit 02c6492); these
tests pin the patch to that implementation bit for bit and check how it is wired into the time loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.physics.curl import interpolate_fields_along_normal
from fdtdx.fdtd.update import _colocate_along_normal


def _fork_reference(E, H, axis):
    """The fork's interpolate_fields (02c6492), verbatim apart from zero padding by jnp.pad."""
    E = jnp.pad(E, ((0, 0), (1, 1), (1, 1), (1, 1)))
    H = jnp.pad(H, ((0, 0), (1, 1), (1, 1), (1, 1)))
    E_x, E_y, E_z = E[0], E[1], E[2]
    H_x, H_y, H_z = H[0], H[1], H[2]
    if axis == 0:
        E_y = E_y[1:-1, 1:-1, 1:-1]
        E_z = E_z[1:-1, 1:-1, 1:-1]
        H_x = H_x[1:-1, 1:-1, 1:-1]
        E_x = (E_x[1:-1, 1:-1, 1:-1] + E_x[:-2, 1:-1, 1:-1]) / 2.0
        H_y = (H_y[1:-1, 1:-1, 1:-1] + H_y[:-2, 1:-1, 1:-1]) / 2.0
        H_z = (H_z[1:-1, 1:-1, 1:-1] + H_z[:-2, 1:-1, 1:-1]) / 2.0
    elif axis == 1:
        E_x = E_x[1:-1, 1:-1, 1:-1]
        E_z = E_z[1:-1, 1:-1, 1:-1]
        H_y = H_y[1:-1, 1:-1, 1:-1]
        E_y = (E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1]) / 2.0
        H_x = (H_x[1:-1, 1:-1, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0
        H_z = (H_z[1:-1, 1:-1, 1:-1] + H_z[1:-1, :-2, 1:-1]) / 2.0
    else:
        E_x = E_x[1:-1, 1:-1, 1:-1]
        E_y = E_y[1:-1, 1:-1, 1:-1]
        H_z = H_z[1:-1, 1:-1, 1:-1]
        E_z = (E_z[1:-1, 1:-1, 1:-1] + E_z[1:-1, 1:-1, :-2]) / 2.0
        H_x = (H_x[1:-1, 1:-1, 1:-1] + H_x[1:-1, 1:-1, :-2]) / 2.0
        H_y = (H_y[1:-1, 1:-1, 1:-1] + H_y[1:-1, 1:-1, :-2]) / 2.0
    return jnp.stack([E_x, E_y, E_z]), jnp.stack([H_x, H_y, H_z])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_matches_the_fork_bit_for_bit(axis):
    rng = np.random.default_rng(axis)
    E = jnp.asarray(rng.standard_normal((3, 6, 7, 8)), dtype=jnp.float32)
    H = jnp.asarray(rng.standard_normal((3, 6, 7, 8)), dtype=jnp.float32)
    # A one-cell plane at index 3 along the axis, spanning cells 1..4 of the others.
    region = tuple((3, 4) if a == axis else (1, 5) for a in range(3))
    block = (slice(None), *(slice(s - 1 if a == axis else s, e) for a, (s, e) in enumerate(region)))

    got_E, got_H = interpolate_fields_along_normal(E[block], H[block], axis, region_slice=region)

    ref_E, ref_H = _fork_reference(E, H, axis)
    target = (slice(None), *(slice(s, e) for s, e in region))
    np.testing.assert_array_equal(np.asarray(got_E), np.asarray(ref_E[target]))
    np.testing.assert_array_equal(np.asarray(got_H), np.asarray(ref_H[target]))


def test_fields_are_not_averaged_across_a_wall():
    # E_y and H_x jump at y = 4, like a field confined by a waveguide wall; an x-normal plane must keep the jump.
    E = jnp.zeros((3, 4, 8, 4)).at[1, :, 4:, :].set(1.0)
    H = jnp.zeros((3, 4, 8, 4)).at[0, :, 4:, :].set(1.0)
    region = ((2, 3), (0, 8), (0, 4))
    got_E, got_H = interpolate_fields_along_normal(E[:, 1:3], H[:, 1:3], 0, region_slice=region)
    np.testing.assert_array_equal(np.asarray(got_E[1, 0, :, 0]), [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(np.asarray(got_H[0, 0, :, 0]), [0, 0, 0, 0, 1, 1, 1, 1])


_SPACING = 25e-9
_SHAPE = (12, 40, 40)


def _simulation(monitor_x: int):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING), time=20 * 0.99 / np.sqrt(3) * _SPACING / fdtdx.constants.c
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=_SHAPE)
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=2), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())
    core = fdtdx.UniformMaterialObject(
        name="core", partial_grid_shape=(None, 16, 8), material=fdtdx.Material(permittivity=12.0)
    )
    constraints += [core.place_at_center(volume, axes=(0, 1, 2)), core.same_size(volume, axes=(0,))]
    objects.append(core)
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    source = fdtdx.ModePlaneSource(
        name="src", partial_grid_shape=(1, None, None), wave_character=wave, direction="+", mode_index=0
    )
    normal = fdtdx.PhasorDetector(
        name="normal", partial_grid_shape=(1, None, None), wave_characters=(wave,), normal_axis_interpolation=True
    )
    full = fdtdx.PhasorDetector(name="full", partial_grid_shape=(1, None, None), wave_characters=(wave,))
    for obj, x in ((source, 3), (normal, monitor_x), (full, monitor_x)):
        constraints += [
            obj.same_size(volume, axes=(1, 2)),
            obj.place_at_center(volume, axes=(1, 2)),
            obj.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x,)),
        ]
        objects.append(obj)
    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key)
    return container, arrays, config


def test_detector_records_normal_axis_colocated_fields():
    container, arrays, config = _simulation(monitor_x=6)
    _, out = fdtdx.run_fdtd(arrays=arrays, objects=container, config=config, show_progress=False)
    normal = np.asarray(out.detector_states["normal"]["phasor"])
    full = np.asarray(out.detector_states["full"]["phasor"])
    assert normal.shape == full.shape and np.all(np.isfinite(normal))
    assert np.abs(normal).max() > 0
    # Same plane, different co-location: the tangential fields differ only by the transverse averaging.
    assert not np.allclose(normal, full)


@pytest.mark.parametrize("monitor_x", [0, 6])
def test_time_loop_colocation_matches_the_fork(monitor_x):
    # At x = 0 the cell below comes from the boundary padding: zeros next to a PML, as in the fork.
    container, _, config = _simulation(monitor_x=monitor_x)
    detector = next(d for d in container.detectors if d.name == "normal")
    rng = np.random.default_rng(monitor_x)
    E, H, H_prev = (jnp.asarray(rng.standard_normal((3, *_SHAPE)), dtype=jnp.float32) for _ in range(3))

    got_E, got_H = _colocate_along_normal(E, H, H_prev, detector, container, config)

    ref_E, ref_H = _fork_reference(E, (H_prev + H) / 2, 0)
    target = (slice(None), *detector.grid_slice)
    np.testing.assert_array_equal(np.asarray(got_E), np.asarray(ref_E[target]))
    np.testing.assert_array_equal(np.asarray(got_H), np.asarray(ref_H[target]))
