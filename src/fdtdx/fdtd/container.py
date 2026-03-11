"""Container module for managing collections of simulation objects and arrays.

This module provides container classes for organizing and managing simulation objects
and array data within FDTD simulations. It includes support for different object types
like sources, detectors, PML boundaries, periodic boundaries, and devices.
"""

from typing import Callable, Self

import jax

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.interfaces.state import RecordingState
from fdtdx.materials import Material
from fdtdx.objects.boundaries.boundary import BaseBoundary
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject

# Type alias for parameter dictionaries containing JAX arrays
ParameterContainer = dict[str, dict[str, jax.Array] | jax.Array]


@autoinit
class ObjectContainer(TreeClass):
    """Container for managing simulation objects and their relationships.

    This class provides a structured way to organize and access different types of simulation
    objects like sources, detectors, PML/periodic boundaries and devices. It maintains object lists
    and provides filtered access to specific object types.
    """

    #: List of all simulation objects in the container.
    object_list: list[SimulationObject]

    #: Index of the volume object in the object list.
    volume_idx: int = frozen_field()

    @property
    def volume(self) -> SimulationObject:
        return self.object_list[self.volume_idx]

    @property
    def objects(self) -> list[SimulationObject]:
        return self.object_list

    @property
    def static_material_objects(self) -> list[UniformMaterialObject | StaticMultiMaterialObject]:
        return [o for o in self.objects if isinstance(o, (UniformMaterialObject, StaticMultiMaterialObject))]

    @property
    def sources(self) -> list[Source]:
        return [o for o in self.objects if isinstance(o, Source)]

    @property
    def devices(self) -> list[Device]:
        return [o for o in self.objects if isinstance(o, Device)]

    @property
    def detectors(self) -> list[Detector]:
        return [o for o in self.objects if isinstance(o, Detector)]

    @property
    def forward_detectors(self) -> list[Detector]:
        return [o for o in self.detectors if not o.inverse]

    @property
    def backward_detectors(self) -> list[Detector]:
        return [o for o in self.detectors if o.inverse]

    @property
    def pml_objects(self) -> list[PerfectlyMatchedLayer]:
        return [o for o in self.objects if isinstance(o, PerfectlyMatchedLayer)]

    @property
    def periodic_objects(self) -> list[PeriodicBoundary]:
        return [o for o in self.objects if isinstance(o, PeriodicBoundary)]

    @property
    def boundary_objects(self) -> list[BaseBoundary]:
        return [o for o in self.objects if isinstance(o, (PerfectlyMatchedLayer, PeriodicBoundary))]

    @property
    def all_objects_non_magnetic(self) -> bool:
        def _fn(m: Material):
            return not m.is_magnetic

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_non_electrically_conductive(self) -> bool:
        def _fn(m: Material):
            return not m.is_electrically_conductive

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_non_magnetically_conductive(self) -> bool:
        def _fn(m: Material):
            return not m.is_magnetically_conductive

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_permittivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_permittivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_permeability(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_permeability

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_electric_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_electric_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_magnetic_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_magnetic_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_permittivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_permittivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_permeability(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_permeability

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_electric_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_electric_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_magnetic_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_magnetic_conductivity

        return self._is_material_fn_true_for_all(_fn)

    def _is_material_fn_true_for_all(
        self,
        fn: Callable[[Material], bool],
    ) -> bool:
        for o in self.objects:
            if isinstance(o, UniformMaterialObject):
                m = o.material
            elif isinstance(o, Device):
                m = o.materials
            elif isinstance(o, StaticMultiMaterialObject):
                m = o.materials
            else:
                continue
            if isinstance(m, Material):
                if not fn(m):
                    return False
            elif isinstance(m, dict):
                for v in m.values():
                    if not fn(v):
                        return False
        return True

    def __iter__(self):
        return iter(self.object_list)

    def __getitem__(
        self,
        key: str,
    ) -> SimulationObject:
        for o in self.objects:
            if o.name == key:
                return o
        raise ValueError(f"Key {key} does not exist in object list: {[o.name for o in self.objects]}")

    def __contains__(
        self,
        key: str,
    ) -> bool:
        for o in self.objects:
            if o.name == key:
                return True
        return False

    def __setitem__(
        self,
        key: str,
        val: SimulationObject,
    ):
        idx = -1
        for cur_idx, o in enumerate(self.objects):
            if o.name == key:
                idx = cur_idx
                break
        if idx == -1:
            ValueError(f"Key {key} does not exist in object list: {[o.name for o in self.objects]}")
        self.object_list[idx] = val

    def copy(
        self,
    ) -> "ObjectContainer":
        new_list = self.object_list.copy()
        return ObjectContainer(
            object_list=new_list,
            volume_idx=self.volume_idx,
        )

    def replace_sources(
        self,
        sources: list[Source],
    ) -> Self:
        new_objects = [o for o in self.objects if o not in self.sources] + sources
        self = self.aset("object_list", new_objects)
        return self


@autoinit
class ArrayContainer(TreeClass):
    """Container for simulation field arrays and states.

    This class holds the electromagnetic field arrays and various state information
    needed during FDTD simulation. It includes the E and H fields, material properties,
    and states for boundaries, detectors and recordings.

    PML coefficients (alpha, kappa, sigma) are stored as tuples of 6 pre-shaped 1D
    arrays for memory efficiency. Each component i corresponds to axis (i % 3) and is
    shaped for broadcasting: (Nx,1,1) for x-axis, (1,Ny,1) for y, (1,1,Nz) for z.
    Components 0-2 are E-field PML, components 3-5 are H-field PML.

    PML auxiliary fields (psi_E, psi_H) are stored sparsely as tuples of 6 (min, max)
    pairs, where each pair contains arrays covering only the PML boundary slabs on the
    min and max sides of the corresponding axis.
    """

    #: Electric field array.
    E: jax.Array

    #: Magnetic field array.
    H: jax.Array

    #: Sparse auxiliary E-field arrays for PML. Tuple of 6 (min_slab, max_slab) pairs.
    psi_E: tuple[tuple[jax.Array, jax.Array], ...]

    #: Sparse auxiliary H-field arrays for PML. Tuple of 6 (min_slab, max_slab) pairs.
    psi_H: tuple[tuple[jax.Array, jax.Array], ...]

    #: 1D alpha profiles for PML. Tuple of 6 arrays shaped for 3D broadcasting.
    alpha: tuple[jax.Array, ...]

    #: 1D kappa profiles for PML. Tuple of 6 arrays shaped for 3D broadcasting.
    kappa: tuple[jax.Array, ...]

    #: 1D sigma profiles for PML. Tuple of 6 arrays shaped for 3D broadcasting.
    sigma: tuple[jax.Array, ...]

    #: Inverse permittivity values array.
    inv_permittivities: jax.Array

    #: Inverse permeability values array.
    inv_permeabilities: jax.Array | float

    #: Dictionary mapping detector names to their states.
    detector_states: dict[str, DetectorState]

    #: Optional state for recording simulation data.
    recording_state: RecordingState | None

    #: field for electric conductivity terms. Defaults to None.
    electric_conductivity: jax.Array | None = None

    #: field for magnetic conductivity terms. Defaults to None.
    magnetic_conductivity: jax.Array | None = None


# time step and arrays
SimulationState = tuple[jax.Array, ArrayContainer]


def reset_array_container(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    reset_detector_states: bool = True,
    reset_recording_state: bool = False,
) -> ArrayContainer:
    """Reset an ArrayContainer's fields and optionally its states.

    This function creates a new ArrayContainer with zeroed E and H fields while preserving
    material properties. It can optionally reset detector and recording states.

    Args:
        arrays (ArrayContainer): The ArrayContainer to reset.
        objects (ObjectContainer): ObjectContainer with simulation objects.
        reset_detector_states (bool, optional): Whether to zero detector states. Defaults to True.
        reset_recording_state (bool, optional): Whether to zero recording state. Defaults to False.

    Returns:
        ArrayContainer: A new ArrayContainer with reset fields and optionally reset states.
    """
    E = arrays.E * 0
    arrays = arrays.aset("E", E)
    H = arrays.H * 0
    arrays = arrays.aset("H", H)

    psi_E = tuple((p_min * 0, p_max * 0) for p_min, p_max in arrays.psi_E)
    arrays = arrays.aset("psi_E", psi_E)
    psi_H = tuple((p_min * 0, p_max * 0) for p_min, p_max in arrays.psi_H)
    arrays = arrays.aset("psi_H", psi_H)

    detector_states = arrays.detector_states
    if reset_detector_states:
        detector_states = {k: {k2: v2 * 0 for k2, v2 in v.items()} for k, v in detector_states.items()}
    arrays = arrays.aset("detector_states", detector_states)

    recording_state = arrays.recording_state
    if reset_recording_state and arrays.recording_state is not None:
        recording_state = RecordingState(
            data={k: v * 0 for k, v in arrays.recording_state.data.items()},
            state={k: v * 0 for k, v in arrays.recording_state.state.items()},
        )
    arrays = arrays.aset("recording_state", recording_state)

    return arrays
