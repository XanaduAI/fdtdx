"""XANADU: precomputed material arrays in ``place_objects``.

SilverHammer builds the whole volume's materials itself, subpixel averaging included, and hands them to
``place_objects``. These tests pin that the arrays are used exactly as given (no copy, no rescaling), that
sources are solved against them rather than against vacuum, and that the combinations which would silently
change them are refused.

Only mode solves run here, no FDTD time stepping.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_SPACING = 25e-9
_WAVELENGTH = 1.55e-6
_SHAPE = (8, 40, 40)
_CORE_CELLS = (16, 8)


def _build(core: fdtdx.SimulationObject | None = None):
    """A volume with PML, a +x mode source and, optionally, one material object."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_SPACING), time=5e-15, dtype=jnp.float32)
    volume = fdtdx.SimulationVolume(partial_grid_shape=_SHAPE)
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=2), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())
    if core is not None:
        constraints.append(core.place_at_center(volume, axes=(0, 1, 2)))
        if isinstance(core, fdtdx.UniformMaterialObject):
            constraints.append(core.same_size(volume, axes=(0,)))
        objects.append(core)
    source = fdtdx.ModePlaneSource(
        name="src",
        partial_grid_shape=(1, None, None),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction="+",
        mode_index=0,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(2,)),
        ]
    )
    objects.append(source)
    return objects, constraints, config


def _core(conductivity: float = 0.0) -> fdtdx.UniformMaterialObject:
    # Diagonally anisotropic, so the arrays carry three components like SilverHammer's.
    material = fdtdx.Material(permittivity=(12.0, 11.0, 10.0), electric_conductivity=conductivity)
    return fdtdx.UniformMaterialObject(name="core", partial_grid_shape=(None, *_CORE_CELLS), material=material)


def _source(container: fdtdx.ObjectContainer) -> fdtdx.ModePlaneSource:
    return next(obj for obj in container.objects if obj.name == "src")


@pytest.mark.parametrize("conductivity", [0.0, 2e3])
def test_precomputed_materials_match_object_built_ones(conductivity):
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build(_core(conductivity))
    ref_container, ref_arrays, _, _, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )

    objects, constraints, config = _build()
    container, arrays, _, _, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
        precomputed_inv_permittivities=ref_arrays.inv_permittivities,
        precomputed_electric_conductivity=ref_arrays.electric_conductivity,
    )

    # Used as given: the same buffers, so no second full-grid copy and no rescaling of the conductivity.
    assert arrays.inv_permittivities is ref_arrays.inv_permittivities
    assert arrays.electric_conductivity is ref_arrays.electric_conductivity

    # The source was solved against the precomputed waveguide, not vacuum, and matches exactly.
    ref_src, src = _source(ref_container), _source(container)
    assert float(np.real(np.asarray(src._neff))) > 1.5
    for attr in ("_neff", "_E", "_H"):
        np.testing.assert_array_equal(np.asarray(getattr(src, attr)), np.asarray(getattr(ref_src, attr)))


def _precomputed(shape=(3, *_SHAPE)):
    return jnp.full(shape, 1 / 2.1, dtype=jnp.float32)


def test_other_material_objects_are_refused():
    objects, constraints, config = _build(_core())
    with pytest.raises(ValueError, match="silently ignored: \\['core'\\]"):
        fdtdx.place_objects(
            object_list=objects, config=config, constraints=constraints, precomputed_inv_permittivities=_precomputed()
        )


def test_fdtdx_subpixel_smoothing_is_refused():
    materials = {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=12.0)}
    sphere = fdtdx.Sphere(
        name="sphere", materials=materials, radius=150e-9, material_name="si", subpixel_smoothing=True
    )
    objects, constraints, config = _build(sphere)
    with pytest.raises(ValueError, match="average a second time"):
        fdtdx.place_objects(
            object_list=objects, config=config, constraints=constraints, precomputed_inv_permittivities=_precomputed()
        )


def test_wrong_shape_is_refused():
    objects, constraints, config = _build()
    with pytest.raises(ValueError, match="precomputed_inv_permittivities must have shape"):
        fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            precomputed_inv_permittivities=_precomputed((3, 8, 40, 39)),
        )
