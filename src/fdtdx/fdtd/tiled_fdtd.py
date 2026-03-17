"""Tiled FDTD for CPU-GPU streaming on unified memory systems (e.g. GH200).

All large arrays (E, H, material properties) are kept in CPU memory. Z-axis
chunks are streamed to GPU for computation, then results are streamed back.
PML auxiliary fields and coefficients stay permanently on GPU.

This avoids the page-fault thrashing that occurs when CUDA unified memory
demand-pages individual 4-64 KB pages across the NVLink interconnect. Instead,
each chunk triggers a single bulk DMA transfer at full NVLink bandwidth.
"""

from functools import partial
import datetime

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.constants import eta0
from fdtdx.core.physics.curl import (
    PSI_COMPONENT_AXIS,
    PSI_E_COEFF_IDX,
    PSI_H_COEFF_IDX,
    SparsePsi,
    _compute_pml_ab,
    _extract_pml_slab,
    _scatter_psi_component,
)
from fdtdx.fdtd.container import (
    ArrayContainer,
    ObjectContainer,
    SimulationState,
)
from fdtdx.fdtd.update import get_periodic_axes

def print_timestamp():
    print(f"Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


# ---------------------------------------------------------------------------
# Memory placement helpers for GH200
# ---------------------------------------------------------------------------

def _free_jax_buffer(arr) -> None:
    """Explicitly free a JAX array's underlying XLA buffer.

    On GH200, the caller's reference to the input ArrayContainer keeps
    large JAX arrays alive on GPU even after tiled_fdtd copies them to
    numpy.  This function invalidates the buffer so GPU HBM is reclaimed
    immediately.  Any subsequent access to the array will raise an error.
    """
    if not isinstance(arr, jax.Array):
        return
    try:
        arr.delete()
    except Exception:
        pass


def _pin_memory_to_cpu():
    """On GH200 in NUMA mode, GPU HBM is exposed as a NUMA node and the OS
    can migrate malloc'd pages to HBM.  Call this early to bind all
    subsequent allocations to the CPU's LPDDR5X NUMA node.

    This is a no-op if libnuma is unavailable or the system has only one
    NUMA node (non-GH200).
    """
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("numa"))
        n_nodes = lib.numa_max_node() + 1
        if n_nodes <= 1:
            return
        lib.numa_set_preferred(0)
    except (OSError, TypeError):
        pass


def _cuda_pin(arr: np.ndarray) -> bool:
    """Pin a C-contiguous numpy array as CUDA page-locked host memory.

    On GH200, this pre-registers pages in the GPU's page table so that
    ``jax.device_put`` can DMA directly at full NVLink-C2C bandwidth
    instead of faulting every 4 KB page (~5 μs each).
    Returns True on success.
    """
    if not arr.flags['C_CONTIGUOUS']:
        return False
    try:
        import ctypes
        rt = ctypes.CDLL('libcudart.so')
        err = rt.cudaHostRegister(
            ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes),
            ctypes.c_uint(1),  # cudaHostRegisterPortable
        )
        return err == 0
    except (OSError, Exception):
        return False


def _to_zfirst(arr: np.ndarray) -> np.ndarray:
    """Transpose (C, Nx, Ny, Nz) → (Nz, C, Nx, Ny) and return C-contiguous."""
    return np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2)))


def _zf_to_gpu(arr_zf: np.ndarray, gpu) -> jax.Array:
    """Transfer a z-first numpy slice to GPU and transpose to kernel layout.

    Input:  (Cz, C, Nx, Ny)  — C-contiguous first-axis slice of pinned memory
    Output: (C, Nx, Ny, Cz)  — on GPU, ready for the kernel
    """
    return jnp.transpose(jax.device_put(arr_zf, gpu), (1, 2, 3, 0))


def _gpu_to_zf(arr_gpu: jax.Array) -> np.ndarray:
    """Transfer kernel-layout GPU array to z-first numpy.

    Input:  (C, Nx, Ny, Cz)  — on GPU, C-contiguous
    Output: (Cz, C, Nx, Ny)  — numpy view (NOT contiguous)

    The GPU array is copied to host as-is (fast DMA on C-contiguous data).
    The transpose is a zero-copy numpy view; the caller's assignment to a
    pinned z-first slice does the physical rearrangement on the CPU.
    """
    return np.asarray(jax.device_get(arr_gpu)).transpose(3, 0, 1, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_xy(field: jax.Array, periodic_axes: tuple[bool, bool, bool]) -> jax.Array:
    """Pad along x (axis 1) and y (axis 2) with +1 on each side.

    Z (axis 3) is NOT padded — the caller handles z-halos explicitly.
    """
    for i in range(2):
        mode = "wrap" if periodic_axes[i] else "constant"
        pw = [(0, 0)] * 4
        pw[i + 1] = (1, 1)
        field = jnp.pad(field, pw, mode=mode)
    return field


def _build_H_halo_for_E(
    H_np: np.ndarray,
    z0: int,
    z1: int,
    Nz: int,
    periodic_z: bool,
    gpu,
) -> jax.Array:
    """H chunk with 1-cell LEFT z-halo.

    H_np is z-first: (Nz, 3, Nx, Ny).
    Returns GPU array in kernel layout: (3, Nx, Ny, Cz+1).
    """
    if z0 > 0:
        halo_zf = H_np[z0 - 1 : z1]  # contiguous first-axis slice
    else:
        chunk = H_np[0:z1]
        left = H_np[-1:] if periodic_z else np.zeros_like(chunk[:1])
        halo_zf = np.concatenate([left, chunk], axis=0)
    return _zf_to_gpu(halo_zf, gpu)


def _build_E_halo_for_H(
    E_np: np.ndarray,
    z0: int,
    z1: int,
    Nz: int,
    periodic_z: bool,
    gpu,
) -> jax.Array:
    """E chunk with 1-cell RIGHT z-halo.

    E_np is z-first: (Nz, 3, Nx, Ny).
    Returns GPU array in kernel layout: (3, Nx, Ny, Cz+1).
    """
    if z1 < Nz:
        halo_zf = E_np[z0 : z1 + 1]  # contiguous first-axis slice
    else:
        chunk = E_np[z0:z1]
        right = E_np[:1] if periodic_z else np.zeros_like(chunk[:1])
        halo_zf = np.concatenate([chunk, right], axis=0)
    return _zf_to_gpu(halo_zf, gpu)


# ---------------------------------------------------------------------------
# PML chunk helpers (called inside JIT)
# ---------------------------------------------------------------------------

def _pml_update_xy(
    psi_min: jax.Array,
    psi_max: jax.Array,
    d_field: jax.Array,
    b_coeff: jax.Array,
    a_coeff: jax.Array,
    axis: int,
    z0: jax.Array,
    Cz: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update an x-axis or y-axis PML psi component for one z-chunk.

    Psi slabs for axes 0 and 1 have full Nz extent so they must be
    z-sliced with ``dynamic_slice`` / ``dynamic_update_slice``.

    Returns (updated_psi_min, updated_psi_max, psi_min_chunk, psi_max_chunk).
    The chunk arrays are what gets scattered into the curl.
    """
    L_min = psi_min.shape[axis]
    L_max = psi_max.shape[axis]

    z_dim = 2  # z is always the last spatial dim of the psi slab

    def _z_slice(arr: jax.Array) -> jax.Array:
        starts = [jnp.int32(0)] * 3
        starts[z_dim] = z0
        sizes = list(arr.shape)
        sizes[z_dim] = Cz
        return lax.dynamic_slice(arr, starts, sizes)

    def _z_write(arr: jax.Array, update: jax.Array) -> jax.Array:
        starts = [jnp.int32(0)] * 3
        starts[z_dim] = z0
        return lax.dynamic_update_slice(arr, update, starts)

    psi_min_c = _z_slice(psi_min)
    psi_max_c = _z_slice(psi_max)

    if L_min > 0:
        d_min = _extract_pml_slab(d_field, axis, L_min, "min")
        b_min = _extract_pml_slab(b_coeff, axis, L_min, "min")
        a_min = _extract_pml_slab(a_coeff, axis, L_min, "min")
        psi_min_c = b_min * psi_min_c + a_min * d_min
        psi_min = _z_write(psi_min, psi_min_c)

    if L_max > 0:
        d_max = _extract_pml_slab(d_field, axis, L_max, "max")
        b_max = _extract_pml_slab(b_coeff, axis, L_max, "max")
        a_max = _extract_pml_slab(a_coeff, axis, L_max, "max")
        psi_max_c = b_max * psi_max_c + a_max * d_max
        psi_max = _z_write(psi_max, psi_max_c)

    return psi_min, psi_max, psi_min_c, psi_max_c


def _pml_update_z(
    psi_min: jax.Array,
    psi_max: jax.Array,
    d_field: jax.Array,
    b_coeff: jax.Array,
    a_coeff: jax.Array,
    z0: jax.Array,
    Cz: int,
    Nz: int,
) -> tuple[jax.Array, jax.Array]:
    """Update a z-axis PML psi component for the current chunk.

    Uses gather-and-mask so the PML thickness and chunk size are independent.
    Each chunk updates only the portion of the psi slab that overlaps its
    z-range; non-overlapping entries are left unchanged.
    """
    L_min = psi_min.shape[2]
    L_max = psi_max.shape[2]

    if L_min > 0:
        b_min = _extract_pml_slab(b_coeff, 2, L_min, "min")
        a_min = _extract_pml_slab(a_coeff, 2, L_min, "min")
        # psi_min index p ↔ global z = p.  Overlap when z0 <= p < z0+Cz.
        psi_idx = jnp.arange(L_min)
        in_chunk = (psi_idx >= z0) & (psi_idx < z0 + Cz)
        chunk_local = jnp.clip(psi_idx - z0, 0, Cz - 1)
        d_gathered = d_field[:, :, chunk_local]  # (Nx, Ny, L_min)
        psi_min_new = b_min * psi_min + a_min * d_gathered
        psi_min = jnp.where(in_chunk[None, None, :], psi_min_new, psi_min)

    if L_max > 0:
        b_max = _extract_pml_slab(b_coeff, 2, L_max, "max")
        a_max = _extract_pml_slab(a_coeff, 2, L_max, "max")
        # psi_max index p ↔ global z = Nz - L_max + p.
        psi_global = Nz - L_max + jnp.arange(L_max)
        in_chunk = (psi_global >= z0) & (psi_global < z0 + Cz)
        chunk_local = jnp.clip(psi_global - z0, 0, Cz - 1)
        d_gathered = d_field[:, :, chunk_local]
        psi_max_new = b_max * psi_max + a_max * d_gathered
        psi_max = jnp.where(in_chunk[None, None, :], psi_max_new, psi_max)

    return psi_min, psi_max


def _scatter_z_psi(
    curl_comp: jax.Array,
    psi_min: jax.Array,
    psi_max: jax.Array,
    sign: float,
    z0: jax.Array,
    Cz: int,
    Nz: int,
) -> jax.Array:
    """Scatter z-axis PML psi into curl for the current chunk.

    Each chunk-local z position that falls inside a PML region picks up
    the corresponding psi value.  No constraint on PML thickness vs Cz.
    """
    L_min = psi_min.shape[2]
    L_max = psi_max.shape[2]
    local_z = jnp.arange(Cz)
    global_z = local_z + z0

    if L_min > 0:
        in_pml_min = global_z < L_min
        safe_idx = jnp.clip(global_z, 0, L_min - 1)
        psi_vals = psi_min[:, :, safe_idx]  # (Nx, Ny, Cz)
        curl_comp = curl_comp + jnp.where(
            in_pml_min[None, None, :], sign * psi_vals, 0.0,
        )

    if L_max > 0:
        pml_max_start = Nz - L_max
        in_pml_max = global_z >= pml_max_start
        safe_idx = jnp.clip(global_z - pml_max_start, 0, L_max - 1)
        psi_vals = psi_max[:, :, safe_idx]
        curl_comp = curl_comp + jnp.where(
            in_pml_max[None, None, :], sign * psi_vals, 0.0,
        )

    return curl_comp


# Scatter mapping: component index → (curl vector index, sign)
_CURL_IDX = (0, 0, 1, 1, 2, 2)
_CURL_SIGN = (+1.0, -1.0, +1.0, -1.0, +1.0, -1.0)


# ---------------------------------------------------------------------------
# JIT-compiled per-chunk E update
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("Cz", "Nz", "periodic_axes", "has_conductivity"))
def _update_E_chunk(
    E_chunk: jax.Array,
    H_halo: jax.Array,
    inv_eps_chunk: jax.Array,
    sigma_E_chunk: jax.Array,
    psi_E: SparsePsi,
    b_pml: tuple[jax.Array, ...],
    a_pml: tuple[jax.Array, ...],
    kappa_x: jax.Array,
    kappa_y: jax.Array,
    kappa_z_chunk: jax.Array,
    z0: jax.Array,
    courant_number: jax.Array,
    Nz: int,
    Cz: int,
    periodic_axes: tuple[bool, bool, bool],
    has_conductivity: bool,
) -> tuple[jax.Array, SparsePsi]:
    """Compute one z-chunk of the E-field update (curl_H → PML → material)."""

    # --- derivatives for curl_H: H[i] - H[i-1] --------------------------
    H_pad = _pad_xy(H_halo, periodic_axes)  # (3, Nx+2, Ny+2, Cz+1)

    # x/y derivs use roll on the padded axes, then crop xy padding and z-halo.
    dyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:]
    dyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:]
    dxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:]
    dxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:]
    # z derivs: explicit adjacent diff (roll would wrap incorrectly in z)
    dzHy = H_pad[1, 1:-1, 1:-1, 1:] - H_pad[1, 1:-1, 1:-1, :-1]
    dzHx = H_pad[0, 1:-1, 1:-1, 1:] - H_pad[0, 1:-1, 1:-1, :-1]

    d_fields = (dyHz, dzHy, dzHx, dxHz, dxHy, dyHx)

    # --- kappa-scaled curl ------------------------------------------------
    curl_x = (1.0 / kappa_y) * dyHz - (1.0 / kappa_z_chunk) * dzHy
    curl_y = (1.0 / kappa_z_chunk) * dzHx - (1.0 / kappa_x) * dxHz
    curl_z = (1.0 / kappa_x) * dxHy - (1.0 / kappa_y) * dyHx
    curls = [curl_x, curl_y, curl_z]

    # --- PML psi update & scatter into curl -------------------------------
    psi_list = list(psi_E)
    for i in range(6):
        axis = PSI_COMPONENT_AXIS[i]
        ci = PSI_E_COEFF_IDX[i]
        psi_min_i, psi_max_i = psi_list[i]
        cidx = _CURL_IDX[i]
        sign = _CURL_SIGN[i]

        if axis != 2:
            psi_min_i, psi_max_i, psi_min_c, psi_max_c = _pml_update_xy(
                psi_min_i, psi_max_i, d_fields[i],
                b_pml[ci], a_pml[ci], axis, z0, Cz,
            )
            curls[cidx] = _scatter_psi_component(
                curls[cidx], psi_min_c, psi_max_c, axis, sign,
            )
        else:
            psi_min_i, psi_max_i = _pml_update_z(
                psi_min_i, psi_max_i, d_fields[i],
                b_pml[ci], a_pml[ci], z0, Cz, Nz,
            )
            curls[cidx] = _scatter_z_psi(
                curls[cidx], psi_min_i, psi_max_i, sign, z0, Cz, Nz,
            )
        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=0)  # (3, Nx, Ny, Cz)

    # --- material update: E = factor*E + c*curl*inv_eps -------------------
    c = courant_number
    if has_conductivity:
        loss = c * sigma_E_chunk * eta0 * inv_eps_chunk / 2
        E_new = (1 - loss) * E_chunk + c * curl * inv_eps_chunk
        E_new = E_new / (1 + loss)
    else:
        E_new = E_chunk + c * curl * inv_eps_chunk

    return E_new, tuple(psi_list)


# ---------------------------------------------------------------------------
# JIT-compiled per-chunk H update
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("Cz", "Nz", "periodic_axes", "has_conductivity", "mu_is_scalar"))
def _update_H_chunk(
    H_chunk: jax.Array,
    E_halo: jax.Array,
    inv_mu_chunk: jax.Array,
    sigma_H_chunk: jax.Array,
    psi_H: SparsePsi,
    b_pml: tuple[jax.Array, ...],
    a_pml: tuple[jax.Array, ...],
    kappa_x: jax.Array,
    kappa_y: jax.Array,
    kappa_z_chunk: jax.Array,
    z0: jax.Array,
    courant_number: jax.Array,
    Nz: int,
    Cz: int,
    periodic_axes: tuple[bool, bool, bool],
    has_conductivity: bool,
    mu_is_scalar: bool,
) -> tuple[jax.Array, SparsePsi]:
    """Compute one z-chunk of the H-field update (curl_E → PML → material)."""

    # --- derivatives for curl_E: E[i+1] - E[i] ---------------------------
    E_pad = _pad_xy(E_halo, periodic_axes)  # (3, Nx+2, Ny+2, Cz+1)

    # x/y derivs: roll -1 gives E[i+1]; crop xy padding and right z-halo.
    dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, :-1]
    dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, :-1]
    dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, :-1]
    dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, :-1]
    # z derivs: explicit adjacent diff
    dzEy = E_pad[1, 1:-1, 1:-1, 1:] - E_pad[1, 1:-1, 1:-1, :-1]
    dzEx = E_pad[0, 1:-1, 1:-1, 1:] - E_pad[0, 1:-1, 1:-1, :-1]

    d_fields = (dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)

    # --- kappa-scaled curl ------------------------------------------------
    curl_x = (1.0 / kappa_y) * dyEz - (1.0 / kappa_z_chunk) * dzEy
    curl_y = (1.0 / kappa_z_chunk) * dzEx - (1.0 / kappa_x) * dxEz
    curl_z = (1.0 / kappa_x) * dxEy - (1.0 / kappa_y) * dyEx
    curls = [curl_x, curl_y, curl_z]

    # --- PML psi update & scatter -----------------------------------------
    psi_list = list(psi_H)
    for i in range(6):
        axis = PSI_COMPONENT_AXIS[i]
        ci = PSI_H_COEFF_IDX[i]
        psi_min_i, psi_max_i = psi_list[i]
        cidx = _CURL_IDX[i]
        sign = _CURL_SIGN[i]

        if axis != 2:
            psi_min_i, psi_max_i, psi_min_c, psi_max_c = _pml_update_xy(
                psi_min_i, psi_max_i, d_fields[i],
                b_pml[ci], a_pml[ci], axis, z0, Cz,
            )
            curls[cidx] = _scatter_psi_component(
                curls[cidx], psi_min_c, psi_max_c, axis, sign,
            )
        else:
            psi_min_i, psi_max_i = _pml_update_z(
                psi_min_i, psi_max_i, d_fields[i],
                b_pml[ci], a_pml[ci], z0, Cz, Nz,
            )
            curls[cidx] = _scatter_z_psi(
                curls[cidx], psi_min_i, psi_max_i, sign, z0, Cz, Nz,
            )
        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=0)

    # --- material update: H = factor*H - c*curl*inv_mu -------------------
    c = courant_number
    inv_mu = inv_mu_chunk if not mu_is_scalar else 1.0
    if has_conductivity:
        loss = c * sigma_H_chunk / eta0 * inv_mu / 2
        H_new = (1 - loss) * H_chunk - c * curl * inv_mu
        H_new = H_new / (1 + loss)
    else:
        H_new = H_chunk - c * curl * inv_mu

    return H_new, tuple(psi_list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _auto_chunk_size(
    Nz: int,
    Nx: int,
    Ny: int,
    C_eps: int,
    has_conductivity: bool,
    mu_is_scalar: bool,
    dtype_bytes: int,
    gpu_budget_bytes: int,
) -> int:
    """Pick the largest chunk_size that divides Nz and fits in GPU memory.

    Conservative estimate: per-chunk GPU allocation includes the input arrays
    (E, H_halo, inv_eps, optionally sigma, inv_mu) plus XLA intermediates
    for the curl + PML computation (~8x the raw input footprint).
    """
    cell_bytes = Nx * Ny * dtype_bytes
    # Input arrays per z-cell:
    #   E_chunk:     3 cells
    #   H_halo:      3 cells (Cz+1, so +3 cells total overhead)
    #   inv_eps:     C_eps cells
    #   sigma_E/H:   C_eps cells each (if present)
    #   inv_mu:      C_eps cells (if array)
    #   result:      3 cells
    arrays_per_z = 3 + 3 + C_eps + 3  # E, H_halo, eps, result
    if has_conductivity:
        arrays_per_z += C_eps * 2  # sigma_E + sigma_H
    if not mu_is_scalar:
        arrays_per_z += C_eps

    # XLA intermediate multiplier (curl derivatives, padding, PML scatter, etc.)
    xla_multiplier = 8
    bytes_per_z = arrays_per_z * cell_bytes * xla_multiplier
    # Fixed overhead for the H_halo extra slice (+1 in z)
    fixed_overhead = 3 * cell_bytes * xla_multiplier

    max_cz = max(1, int((gpu_budget_bytes - fixed_overhead) // bytes_per_z))

    # Find the largest divisor of Nz that is <= max_cz
    best = 1
    for d in range(1, min(max_cz, Nz) + 1):
        if Nz % d == 0:
            best = d
    return best


def tiled_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    chunk_size: int | None = None,
) -> SimulationState:
    """Run a forward-only FDTD simulation with CPU-GPU tiled streaming.

    Large field and material arrays live in CPU (Grace) memory.  Each time step
    is broken into z-axis chunks that are streamed to the GPU (Hopper) for
    computation, avoiding the page-fault thrashing of CUDA unified memory.

    Args:
        arrays: Initial simulation state (fields, materials, etc.).
        objects: Simulation objects (sources, detectors, boundaries, …).
        config: Simulation configuration.
        key: JAX PRNG key (unused in forward-only mode, kept for API compat).
        chunk_size: Number of z-cells per GPU chunk.  Must divide ``Nz``.
            Larger values increase GPU utilisation at the cost of GPU memory.
            If ``None`` (default), automatically selects the largest chunk size
            that fits in GPU memory.

    .. warning::
        This function **deletes the underlying XLA buffers** of the input
        ``arrays`` to free GPU memory.  The caller's reference to the input
        ``ArrayContainer`` will contain invalidated arrays after this call.
        Always use the returned ``arrays`` for subsequent work.

    Returns:
        ``(time_step, arrays)`` — same shape as ``checkpointed_fdtd``.
    """
    print_timestamp()
    print("Starting tiled_fdtd")

    import gc

    del key  # unused in forward-only

    # On GH200 in NUMA mode, ensure allocations land on CPU LPDDR5X
    _pin_memory_to_cpu()

    # ------------------------------------------------------------------
    # 0. Validate
    # ------------------------------------------------------------------
    inv_eps = arrays.inv_permittivities
    if inv_eps.shape[0] == 9:
        raise NotImplementedError(
            "Tiled FDTD does not yet support fully anisotropic materials "
            "(inv_permittivities shape[0] == 9).  Use diagonal anisotropy "
            "(shape[0] == 3) or isotropic (shape[0] == 1)."
        )

    # ------------------------------------------------------------------
    # 1. Device handles
    # ------------------------------------------------------------------
    gpu = jax.devices("gpu")[0]
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float64

    # ------------------------------------------------------------------
    # 2. Extract data from ArrayContainer into numpy / GPU arrays,
    #    then FREE the large JAX buffers to avoid holding 2x memory.
    #    Fields are zero (reset), so create numpy directly.
    #
    #    All large numpy arrays use Z-FIRST layout: (Nz, C, Nx, Ny).
    #    First-axis slices are C-contiguous → no strided copies needed.
    #    Arrays are CUDA-pinned so jax.device_put uses fast DMA.
    # ------------------------------------------------------------------
    field_shape = arrays.E.shape  # (3, Nx, Ny, Nz) — original layout
    _, Nx, Ny, Nz = field_shape

    # Z-first field arrays: (Nz, 3, Nx, Ny)
    E_np = np.zeros((Nz, 3, Nx, Ny), dtype=np_dtype)
    H_np = np.zeros((Nz, 3, Nx, Ny), dtype=np_dtype)

    # Material arrays: transpose from (C, Nx, Ny, Nz) → (Nz, C, Nx, Ny)
    inv_eps_np = _to_zfirst(np.array(jax.device_get(arrays.inv_permittivities)))

    has_sigma_E = arrays.electric_conductivity is not None
    has_sigma_H = arrays.magnetic_conductivity is not None
    sigma_E_np = _to_zfirst(np.array(jax.device_get(arrays.electric_conductivity))) if has_sigma_E else None
    sigma_H_np = _to_zfirst(np.array(jax.device_get(arrays.magnetic_conductivity))) if has_sigma_H else None

    inv_mu_val = arrays.inv_permeabilities
    mu_is_scalar = not isinstance(inv_mu_val, jax.Array) or inv_mu_val.ndim == 0
    inv_mu_np = None if mu_is_scalar else _to_zfirst(np.array(jax.device_get(inv_mu_val)))

    # PML → GPU (small, stay resident).  Zero the psi fields (reset).
    # Note: alpha/kappa/sigma are 1D coefficient arrays — already tiny.
    # psi arrays are zeroed.  jax.device_put is a no-op if already on GPU,
    # so we must NOT delete these source buffers (they'd be the same object).
    psi_E: SparsePsi = jax.device_put(
        tuple((p_min * 0, p_max * 0) for p_min, p_max in arrays.psi_E), gpu,
    )
    psi_H: SparsePsi = jax.device_put(
        tuple((p_min * 0, p_max * 0) for p_min, p_max in arrays.psi_H), gpu,
    )
    alpha = tuple(jax.device_put(a, gpu) for a in arrays.alpha)
    kappa = tuple(jax.device_put(k, gpu) for k in arrays.kappa)
    sigma_pml = tuple(jax.device_put(s, gpu) for s in arrays.sigma)

    b_pml, a_pml = _compute_pml_ab(alpha, kappa, sigma_pml, config)

    kappa_x = kappa[0]   # (Nx, 1, 1)
    kappa_y = kappa[1]   # (1, Ny, 1)
    kappa_z_full = kappa[2]  # (1, 1, Nz) — sliced per chunk

    has_detectors = bool(objects.forward_detectors)

    # Save recording_state before deleting JAX buffers
    _saved_recording_state = arrays.recording_state

    # Capture scalar inv_permeabilities value before freeing JAX buffers.
    if mu_is_scalar:
        inv_mu_scalar = float(jax.device_get(jnp.asarray(inv_mu_val)))
    else:
        inv_mu_scalar = None

    # Detector states — zero (reset), keep on GPU to avoid per-step transfers.
    # Free original detector state buffers after zeroing and moving to GPU.
    detector_states_gpu: dict = {}
    for k, v in arrays.detector_states.items():
        detector_states_gpu[k] = {}
        for k2, v2 in v.items():
            detector_states_gpu[k][k2] = jax.device_put(v2 * 0, gpu)
            _free_jax_buffer(v2)

    # FREE large JAX/GPU buffers.  The caller's reference to the original
    # ArrayContainer keeps the JAX arrays alive on GPU even after we
    # reassign our local `arrays`.  We must explicitly delete the
    # underlying XLA buffers so GPU HBM is reclaimed immediately.
    _free_jax_buffer(arrays.E)
    _free_jax_buffer(arrays.H)
    _free_jax_buffer(arrays.inv_permittivities)
    if has_sigma_E:
        _free_jax_buffer(arrays.electric_conductivity)
    if has_sigma_H:
        _free_jax_buffer(arrays.magnetic_conductivity)
    if not mu_is_scalar:
        _free_jax_buffer(inv_mu_val)
    del inv_mu_val
    del inv_eps
    gc.collect()

    # CUDA-pin the large z-first numpy arrays for fast DMA
    for _arr in [E_np, H_np, inv_eps_np]:
        _cuda_pin(_arr)
    if sigma_E_np is not None:
        _cuda_pin(sigma_E_np)
    if sigma_H_np is not None:
        _cuda_pin(sigma_H_np)
    if inv_mu_np is not None:
        _cuda_pin(inv_mu_np)

    # ------------------------------------------------------------------
    # 3. Chunk configuration
    # ------------------------------------------------------------------
    C_eps = inv_eps_np.shape[1]  # z-first: (Nz, C, Nx, Ny)
    dtype_bytes = 4 if np_dtype == np.float32 else 8

    if chunk_size is None:
        # Query GPU memory and subtract a safety margin for PML + JIT overhead
        gpu_mem = gpu.memory_stats()
        if gpu_mem and "bytes_limit" in gpu_mem:
            total_bytes = gpu_mem["bytes_limit"]
            in_use = gpu_mem.get("bytes_in_use", 0)
            available = int((total_bytes - in_use) * 0.75)  # 75% safety margin
        else:
            available = 80 * 1024**3  # conservative 80 GB fallback
        Cz = _auto_chunk_size(
            Nz, Nx, Ny, C_eps,
            has_conductivity=has_sigma_E,
            mu_is_scalar=mu_is_scalar,
            dtype_bytes=dtype_bytes,
            gpu_budget_bytes=available,
        )
        print(f"Auto chunk_size = {Cz}  ({Nz // Cz} chunks, "
              f"GPU budget {available / 1024**3:.1f} GB)")
    else:
        Cz = chunk_size

    if Nz % Cz != 0:
        raise ValueError(
            f"Nz ({Nz}) must be divisible by chunk_size ({Cz}).  "
            f"Try chunk_size={Nz // max(1, Nz // Cz)}."
        )
    n_chunks = Nz // Cz

    periodic_axes = get_periodic_axes(objects)
    periodic_z = periodic_axes[2]
    c_num = jnp.asarray(config.courant_number, dtype=config.dtype)

    _dummy_sigma = jnp.zeros((1,), dtype=config.dtype)

    print_timestamp()
    print(f"Time loop starting")

    # ------------------------------------------------------------------
    # 4. Time loop
    # ------------------------------------------------------------------
    for t in range(config.time_steps_total):
        if t % 1 == 0:
            print_timestamp()
            print(f"Time step {t} of {config.time_steps_total}")
        time_step = jnp.asarray(t, dtype=jnp.int32)

        # ==============================================================
        # Phase 1 — E update  (reads H, writes E)
        # ==============================================================
        print_timestamp()
        print(f"E update starting")
        import time as _time
        for iz in range(n_chunks):
            z0_int = iz * Cz
            z1_int = z0_int + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            _t0 = _time.perf_counter()
            H_halo = _build_H_halo_for_E(H_np, z0_int, z1_int, Nz, periodic_z, gpu)
            _t1 = _time.perf_counter()
            E_chunk = _zf_to_gpu(E_np[z0_int:z1_int], gpu)
            _t2 = _time.perf_counter()
            eps_chunk = _zf_to_gpu(inv_eps_np[z0_int:z1_int], gpu)
            _t3 = _time.perf_counter()
            sig_E_chunk = (
                _zf_to_gpu(sigma_E_np[z0_int:z1_int], gpu)  # type: ignore[index]
                if has_sigma_E else _dummy_sigma
            )
            kz_chunk = kappa_z_full[:, :, z0_int:z1_int]

            _t4 = _time.perf_counter()
            E_new, psi_E = _update_E_chunk(
                E_chunk, H_halo, eps_chunk, sig_E_chunk,
                psi_E, b_pml, a_pml,
                kappa_x, kappa_y, kz_chunk,
                z0_jax, c_num,
                Nz=Nz, Cz=Cz,
                periodic_axes=periodic_axes,
                has_conductivity=has_sigma_E,
            )
            _t5 = _time.perf_counter()

            E_np[z0_int:z1_int] = _gpu_to_zf(E_new)
            _t6 = _time.perf_counter()

            if t < 2 and iz < 3:
                print(f"  E chunk {iz}: halo={_t1-_t0:.3f}s  E_put={_t2-_t1:.3f}s  "
                      f"eps_put={_t3-_t2:.3f}s  sig+kz={_t4-_t3:.3f}s  "
                      f"kernel={_t5-_t4:.3f}s  get+write={_t6-_t5:.3f}s  "
                      f"total={_t6-_t0:.3f}s")

        # --- E-field sources (on GPU, small region only) ---
        print_timestamp()
        print(f"E-field sources starting")
        _apply_sources_E_gpu(
            E_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar,
            objects, time_step, config.dtype, gpu,
        )

        # ==============================================================
        # Phase 2 — H update  (reads E, writes H)
        # ==============================================================
        print_timestamp()
        print(f"H update starting")
        for iz in range(n_chunks):
            z0_int = iz * Cz
            z1_int = z0_int + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            _t0 = _time.perf_counter()
            E_halo = _build_E_halo_for_H(E_np, z0_int, z1_int, Nz, periodic_z, gpu)
            _t1 = _time.perf_counter()
            H_chunk = _zf_to_gpu(H_np[z0_int:z1_int], gpu)
            _t2 = _time.perf_counter()

            if mu_is_scalar:
                mu_chunk = jnp.asarray(1.0, dtype=config.dtype)
            else:
                mu_chunk = _zf_to_gpu(inv_mu_np[z0_int:z1_int], gpu)  # type: ignore[index]

            sig_H_chunk = (
                _zf_to_gpu(sigma_H_np[z0_int:z1_int], gpu)  # type: ignore[index]
                if has_sigma_H else _dummy_sigma
            )
            kz_chunk = kappa_z_full[:, :, z0_int:z1_int]

            _t3 = _time.perf_counter()
            H_new, psi_H = _update_H_chunk(
                H_chunk, E_halo, mu_chunk, sig_H_chunk,
                psi_H, b_pml, a_pml,
                kappa_x, kappa_y, kz_chunk,
                z0_jax, c_num,
                Nz=Nz, Cz=Cz,
                periodic_axes=periodic_axes,
                has_conductivity=has_sigma_H,
                mu_is_scalar=mu_is_scalar,
            )
            _t4 = _time.perf_counter()

            H_np[z0_int:z1_int] = _gpu_to_zf(H_new)
            _t5 = _time.perf_counter()

            if t < 2 and iz < 3:
                print(f"  H chunk {iz}: halo={_t1-_t0:.3f}s  H_put={_t2-_t1:.3f}s  "
                      f"mu+sig+kz={_t3-_t2:.3f}s  "
                      f"kernel={_t4-_t3:.3f}s  get+write={_t5-_t4:.3f}s  "
                      f"total={_t5-_t0:.3f}s")

        # --- H-field sources (on GPU, small region only) ---
        print_timestamp()
        print(f"H-field sources starting")
        _apply_sources_H_gpu(
            H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar,
            objects, time_step, config.dtype, gpu,
        )

        # ==============================================================
        # Detector update (on GPU, small sliced regions only)
        # ==============================================================
        print_timestamp()
        print(f"Detector update starting")
        if has_detectors:
            detector_states_gpu = _update_detectors_gpu(
                E_np, H_np, inv_eps_np, inv_mu_np,
                mu_is_scalar, inv_mu_scalar,
                detector_states_gpu, objects, time_step, gpu,
            )

    # ------------------------------------------------------------------
    # 5. Reconstruct output ArrayContainer
    #    Transpose z-first numpy → original (C, Nx, Ny, Nz) layout.
    #    We cannot use arrays.aset() because the input arrays' XLA
    #    buffers were deleted — tree_copy would crash.  Construct fresh.
    # ------------------------------------------------------------------
    final_E = jnp.asarray(np.ascontiguousarray(np.transpose(E_np, (1, 2, 3, 0))))
    final_H = jnp.asarray(np.ascontiguousarray(np.transpose(H_np, (1, 2, 3, 0))))
    final_inv_eps = jnp.asarray(np.ascontiguousarray(np.transpose(inv_eps_np, (1, 2, 3, 0))))

    final_sigma_E = (
        jnp.asarray(np.ascontiguousarray(np.transpose(sigma_E_np, (1, 2, 3, 0))))  # type: ignore[arg-type]
        if has_sigma_E else None
    )
    final_sigma_H = (
        jnp.asarray(np.ascontiguousarray(np.transpose(sigma_H_np, (1, 2, 3, 0))))  # type: ignore[arg-type]
        if has_sigma_H else None
    )
    if not mu_is_scalar:
        final_inv_mu = jnp.asarray(np.ascontiguousarray(np.transpose(inv_mu_np, (1, 2, 3, 0))))  # type: ignore[arg-type]
    else:
        final_inv_mu = jnp.asarray(inv_mu_scalar)

    out = ArrayContainer(
        E=final_E,
        H=final_H,
        psi_E=psi_E,
        psi_H=psi_H,
        alpha=alpha,
        kappa=kappa,
        sigma=sigma_pml,
        inv_permittivities=final_inv_eps,
        inv_permeabilities=final_inv_mu,
        detector_states=detector_states_gpu,
        recording_state=_saved_recording_state,
        electric_conductivity=final_sigma_E,
        magnetic_conductivity=final_sigma_H,
    )

    final_time = jnp.asarray(config.time_steps_total, dtype=jnp.int32)
    return (final_time, out)


# ---------------------------------------------------------------------------
# GPU source helpers — extract the source's small spatial region from numpy,
# send it to GPU, remap grid_slice to zero-based, and run source.update_*
# on GPU.  Only the tiny source region is transferred, not the full grid.
# ---------------------------------------------------------------------------

def _remap_to_gpu(
    obj,
    field_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_mu_np: np.ndarray | None,
    mu_is_scalar: bool,
    inv_mu_scalar: float | None,
    gpu,
):
    """Extract *obj*'s spatial region from z-first numpy arrays, send to GPU,
    and return (remapped_obj, field_gpu, eps_gpu, mu_gpu).

    field_np is (Nz, C, Nx, Ny).  The GPU arrays have kernel layout (C, dx, dy, dz).
    The remapped object has ``_grid_slice_tuple`` set to
    ``((0,dx),(0,dy),(0,dz))`` so ``grid_slice`` addresses the full
    small array.
    """
    gst = obj._grid_slice_tuple
    dx = gst[0][1] - gst[0][0]
    dy = gst[1][1] - gst[1][0]
    dz = gst[2][1] - gst[2][0]
    sx, sy, sz = obj.grid_slice

    # z-first: (Nz, C, Nx, Ny) → slice [sz, :, sx, sy] → (dz, C, dx, dy)
    # then transpose to kernel layout (C, dx, dy, dz)
    field_region = np.ascontiguousarray(
        np.transpose(field_np[sz, :, sx, sy], (1, 2, 3, 0))
    )
    field_gpu = jax.device_put(jnp.asarray(field_region), gpu)

    eps_region = np.ascontiguousarray(
        np.transpose(inv_eps_np[sz, :, sx, sy], (1, 2, 3, 0))
    )
    eps_gpu = jax.device_put(jnp.asarray(eps_region), gpu)

    if mu_is_scalar:
        mu_gpu = inv_mu_scalar
    else:
        mu_region = np.ascontiguousarray(
            np.transpose(inv_mu_np[sz, :, sx, sy], (1, 2, 3, 0))  # type: ignore[index]
        )
        mu_gpu = jax.device_put(jnp.asarray(mu_region), gpu)

    obj_remap = obj.aset("_grid_slice_tuple", ((0, dx), (0, dy), (0, dz)))
    return obj_remap, field_gpu, eps_gpu, mu_gpu


def _apply_sources_E_gpu(
    E_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_mu_np: np.ndarray | None,
    mu_is_scalar: bool,
    inv_mu_scalar: float | None,
    objects: ObjectContainer,
    time_step: jax.Array,
    dtype,
    gpu,
) -> None:
    """Apply E-field sources on GPU using only the source's spatial region."""
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))):
            continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice

        src_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu(
            source, E_np, inv_eps_np, inv_mu_np,
            mu_is_scalar, inv_mu_scalar, gpu,
        )
        E_updated = src_remap.update_E(
            E=E_gpu,
            inv_permittivities=eps_gpu,
            inv_permeabilities=mu_gpu,
            time_step=adj,
            inverse=False,
        )
        # kernel layout (3, dx, dy, dz) → z-first (dz, 3, dx, dy)
        E_np[sz, :, sx, sy] = np.transpose(
            np.asarray(jax.device_get(E_updated)), (3, 0, 1, 2)
        )


def _apply_sources_H_gpu(
    H_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_mu_np: np.ndarray | None,
    mu_is_scalar: bool,
    inv_mu_scalar: float | None,
    objects: ObjectContainer,
    time_step: jax.Array,
    dtype,
    gpu,
) -> None:
    """Apply H-field sources on GPU using only the source's spatial region."""
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))):
            continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice

        src_remap, H_gpu, eps_gpu, mu_gpu = _remap_to_gpu(
            source, H_np, inv_eps_np, inv_mu_np,
            mu_is_scalar, inv_mu_scalar, gpu,
        )
        H_updated = src_remap.update_H(
            H=H_gpu,
            inv_permittivities=eps_gpu,
            inv_permeabilities=mu_gpu,
            time_step=adj + 0.5,
            inverse=False,
        )
        H_np[sz, :, sx, sy] = np.transpose(
            np.asarray(jax.device_get(H_updated)), (3, 0, 1, 2)
        )


# ---------------------------------------------------------------------------
# GPU detector update — extract each detector's small spatial region, send
# to GPU, remap grid_slice, and run d.update() on GPU.  Detector states
# stay on GPU between time steps to avoid per-step transfers.
# ---------------------------------------------------------------------------

def _update_detectors_gpu(
    E_np: np.ndarray,
    H_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_mu_np: np.ndarray | None,
    mu_is_scalar: bool,
    inv_mu_scalar: float | None,
    detector_states: dict,
    objects: ObjectContainer,
    time_step: jax.Array,
    gpu,
) -> dict:
    """Update detector states on GPU using only each detector's spatial slice.

    Each detector's region is extracted from numpy (tiny), sent to GPU,
    and the detector's ``.update()`` runs entirely on GPU.  Detector
    states stay resident on GPU between time steps.
    """
    for d in objects.forward_detectors:
        is_on = bool(jax.device_get(d._is_on_at_time_step_arr[time_step]))
        if not is_on:
            continue

        d_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu(
            d, E_np, inv_eps_np, inv_mu_np,
            mu_is_scalar, inv_mu_scalar, gpu,
        )
        sx, sy, sz = d.grid_slice
        H_region = np.ascontiguousarray(
            np.transpose(H_np[sz, :, sx, sy], (1, 2, 3, 0))
        )
        H_gpu = jax.device_put(jnp.asarray(H_region), gpu)

        detector_states[d.name] = d_remap.update(
            time_step=time_step,
            E=E_gpu,
            H=H_gpu,
            state=detector_states[d.name],
            inv_permittivity=eps_gpu,
            inv_permeability=mu_gpu,
        )

    return detector_states
