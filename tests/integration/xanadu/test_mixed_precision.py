"""XANADU: mixed precision with ``SimulationConfig.material_dtype``.

SilverHammer runs every job with jax x64 enabled, float64 accumulators (fields, PML, sources, detectors) and
float32 full-grid material arrays. These tests pin that split, that ``material_dtype`` defaults to ``dtype``
(upstream behavior), and that a float32 run stays float32 under x64 instead of being promoted.

x64 is enabled per test with ``jax.enable_x64``, so the rest of the suite is unaffected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_SPACING = 25e-9
_WAVELENGTH = 1.55e-6
_SHAPE = (12, 40, 40)
_STEPS = 20


def _build(dtype, material_dtype, core: fdtdx.SimulationObject | None = None):
    """A volume with PML, a +x mode source, a phasor plane and, optionally, one material object."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        time=_STEPS * 0.99 / np.sqrt(3) * _SPACING / fdtdx.constants.c,
        dtype=dtype,
        material_dtype=material_dtype,
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=_SHAPE)
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=2), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())
    if core is not None:
        constraints += [core.place_at_center(volume, axes=(0, 1, 2)), core.same_size(volume, axes=(0,))]
        objects.append(core)
    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.ModePlaneSource(
        name="src", partial_grid_shape=(1, None, None), wave_character=wave, direction="+", mode_index=0
    )
    phasor = fdtdx.PhasorDetector(
        name="phasor", partial_grid_shape=(1, None, None), wave_characters=(wave,), dtype=jnp.complex128
    )
    for obj, x in ((source, 3), (phasor, 6)):
        constraints += [
            obj.same_size(volume, axes=(1, 2)),
            obj.place_at_center(volume, axes=(1, 2)),
            obj.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x,)),
        ]
        objects.append(obj)
    return objects, constraints, config


def _core() -> fdtdx.UniformMaterialObject:
    # Diagonally anisotropic and lossy, like SilverHammer's materials.
    material = fdtdx.Material(permittivity=(12.0, 11.0, 10.0), electric_conductivity=2e3)
    return fdtdx.UniformMaterialObject(name="core", partial_grid_shape=(None, 16, 8), material=material)


def _place(objects, constraints, config, **kwargs):
    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key, **kwargs
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key)
    return container, arrays, config


def _run(container, arrays, config):
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=container, config=config, show_progress=False)
    return arrays


def _accumulators(arrays):
    psi = jax.tree_util.tree_leaves((arrays.fields.psi_E, arrays.fields.psi_H))
    return [arrays.fields.E, arrays.fields.H, *psi]


def test_material_dtype_defaults_to_dtype():
    grid = fdtdx.UniformGrid(spacing=_SPACING)
    assert fdtdx.SimulationConfig(grid=grid, time=1e-15).material_dtype == jnp.float32
    assert fdtdx.SimulationConfig(grid=grid, time=1e-15, dtype=jnp.float64).material_dtype == jnp.float64
    config = fdtdx.SimulationConfig(grid=grid, time=1e-15, dtype=jnp.float64, material_dtype=jnp.float32)
    assert config.material_dtype == jnp.float32


@pytest.mark.parametrize("precomputed", [False, True])
def test_float64_accumulators_float32_materials(precomputed):
    with jax.enable_x64(True):
        kwargs = {}
        if precomputed:
            _, ref_arrays, _ = _place(*_build(jnp.float32, jnp.float32, _core()))
            kwargs = {
                "precomputed_inv_permittivities": ref_arrays.inv_permittivities,
                "precomputed_electric_conductivity": ref_arrays.electric_conductivity,
            }
        container, arrays, config = _place(
            *_build(jnp.float64, jnp.float32, None if precomputed else _core()), **kwargs
        )

        assert arrays.inv_permittivities.dtype == jnp.float32
        assert arrays.electric_conductivity.dtype == jnp.float32
        assert all(a.dtype == jnp.float64 for a in _accumulators(arrays))
        for pml in container.pml_objects:
            assert pml.pml_a_E.dtype == pml.pml_b_E.dtype == pml.inv_kappa_E.dtype == jnp.float64
        src = next(o for o in container.objects if o.name == "src")
        assert src._E.dtype == src._H.dtype == jnp.complex128
        assert float(np.real(np.asarray(src._neff))) > 1.5

        out = _run(container, arrays, config)
        assert all(a.dtype == jnp.float64 for a in _accumulators(out))
        assert out.inv_permittivities.dtype == out.electric_conductivity.dtype == jnp.float32
        assert out.detector_states["phasor"]["phasor"].dtype == jnp.complex128
        assert float(jnp.abs(out.fields.E).max()) > 0


def _full_grid_64bit_values(container, arrays, config):
    """(primitive, dtype) of every 64-bit field-sized value in the traced time loop."""
    jaxpr = jax.make_jaxpr(lambda a: _run(container, a, config))(arrays).jaxpr
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            for v in eqn.outvars:
                if v.aval.shape == arrays.fields.E.shape and v.aval.dtype in (jnp.float64, jnp.complex128):
                    found.append((eqn.primitive.name, str(v.aval.dtype)))
            for param in eqn.params.values():
                for sub in param if isinstance(param, (list, tuple)) else (param,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        walk(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        walk(sub)

    walk(jaxpr)
    return found


def test_float32_run_stays_float32_under_x64():
    # Upstream promoted the PML coefficients to float64 under x64 and the time loop failed on the psi dtype.
    with jax.enable_x64(True):
        container, arrays, config = _place(*_build(jnp.float32, None, _core()))
        for pml in container.pml_objects:
            assert pml.pml_a_E.dtype == pml.pml_b_E.dtype == jnp.float32
        # The lossy mode source is complex128 even here; its increments must not promote E/H for the add.
        src = next(o for o in container.objects if o.name == "src")
        assert src._E.dtype == jnp.complex128
        assert _full_grid_64bit_values(container, arrays, config) == []
        out = _run(container, arrays, config)
        assert all(a.dtype == jnp.float32 for a in _accumulators(out))
        assert out.inv_permittivities.dtype == jnp.float32


def test_precomputed_dtype_must_match_material_dtype():
    with jax.enable_x64(True):
        objects, constraints, config = _build(jnp.float64, jnp.float32)
        inv_eps = jnp.full((3, *_SHAPE), 1 / 2.1, dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"config\.material_dtype is float32"):
            _place(objects, constraints, config, precomputed_inv_permittivities=inv_eps)
