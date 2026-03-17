"""Tiled FDTD for CPU-GPU streaming on unified memory systems (e.g. GH200).

All large arrays (E, H, material properties) are kept in CPU memory. Z-axis
chunks are streamed to GPU for computation, then results are streamed back.
PML auxiliary fields and coefficients stay permanently on GPU.

This avoids the page-fault thrashing that occurs when CUDA unified memory
demand-pages individual 4-64 KB pages across the NVLink interconnect. Instead,
each chunk triggers a single bulk DMA transfer at full NVLink bandwidth.
"""

from functools import partial

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
    """H chunk with 1-cell LEFT z-halo for curl_H.  Shape (3, Nx, Ny, Cz+1)."""
    if z0 > 0:
        return jax.device_put(H_np[:, :, :, z0 - 1 : z1], gpu)
    chunk = H_np[:, :, :, 0:z1]
    if periodic_z:
        left = H_np[:, :, :, -1:]
    else:
        left = np.zeros_like(chunk[:, :, :, :1])
    return jax.device_put(np.concatenate([left, chunk], axis=3), gpu)


def _build_E_halo_for_H(
    E_np: np.ndarray,
    z0: int,
    z1: int,
    Nz: int,
    periodic_z: bool,
    gpu,
) -> jax.Array:
    """E chunk with 1-cell RIGHT z-halo for curl_E.  Shape (3, Nx, Ny, Cz+1)."""
    if z1 < Nz:
        return jax.device_put(E_np[:, :, :, z0 : z1 + 1], gpu)
    chunk = E_np[:, :, :, z0:z1]
    if periodic_z:
        right = E_np[:, :, :, :1]
    else:
        right = np.zeros_like(chunk[:, :, :, :1])
    return jax.device_put(np.concatenate([chunk, right], axis=3), gpu)


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

def tiled_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    chunk_size: int = 64,
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

    Returns:
        ``(time_step, arrays)`` — same shape as ``checkpointed_fdtd``.
    """
    import gc

    del key  # unused in forward-only

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
    cpu = jax.devices("cpu")[0]
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float64

    # ------------------------------------------------------------------
    # 2. Extract data from ArrayContainer into numpy / GPU arrays,
    #    then FREE the large JAX buffers to avoid holding 2x memory.
    #    Fields are zero (reset), so create numpy directly.
    # ------------------------------------------------------------------
    field_shape = arrays.E.shape
    E_np = np.zeros(field_shape, dtype=np_dtype)
    H_np = np.zeros(field_shape, dtype=np_dtype)

    inv_eps_np = np.array(jax.device_get(arrays.inv_permittivities))

    has_sigma_E = arrays.electric_conductivity is not None
    has_sigma_H = arrays.magnetic_conductivity is not None
    sigma_E_np = np.array(jax.device_get(arrays.electric_conductivity)) if has_sigma_E else None
    sigma_H_np = np.array(jax.device_get(arrays.magnetic_conductivity)) if has_sigma_H else None

    inv_mu_val = arrays.inv_permeabilities
    mu_is_scalar = not isinstance(inv_mu_val, jax.Array) or inv_mu_val.ndim == 0
    inv_mu_np = None if mu_is_scalar else np.array(jax.device_get(inv_mu_val))

    # PML → GPU (small, stay resident).  Zero the psi fields (reset).
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

    # Detector states — zero (reset), keep on CPU
    detector_states: dict = {
        k: {k2: jax.device_put(v2 * 0, cpu) for k2, v2 in v.items()}
        for k, v in arrays.detector_states.items()
    }
    has_detectors = bool(objects.forward_detectors)

    # Cache inv_permeabilities for source calls (small scalar or freed below)
    inv_mu_for_sources = arrays.inv_permeabilities

    # FREE large JAX buffers — we've copied everything we need into
    # numpy / GPU arrays above.  This is critical to avoid 2× memory.
    _ph = jnp.zeros((1,), dtype=config.dtype)
    arrays = arrays.aset("E", _ph)
    arrays = arrays.aset("H", _ph)
    arrays = arrays.aset("inv_permittivities", _ph)
    if has_sigma_E:
        arrays = arrays.aset("electric_conductivity", _ph)
    if has_sigma_H:
        arrays = arrays.aset("magnetic_conductivity", _ph)
    if not mu_is_scalar:
        arrays = arrays.aset("inv_permeabilities", _ph)
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Chunk configuration
    # ------------------------------------------------------------------
    Nz = field_shape[3]
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

    # ------------------------------------------------------------------
    # 4. Time loop
    # ------------------------------------------------------------------
    for t in range(config.time_steps_total):
        if t % 1 == 0:
            print(f"Time step {t} of {config.time_steps_total}")
        time_step = jnp.asarray(t, dtype=jnp.int32)

        # ==============================================================
        # Phase 1 — E update  (reads H, writes E)
        # ==============================================================
        for iz in range(n_chunks):
            z0_int = iz * Cz
            z1_int = z0_int + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            H_halo = _build_H_halo_for_E(H_np, z0_int, z1_int, Nz, periodic_z, gpu)
            E_chunk = jax.device_put(E_np[:, :, :, z0_int:z1_int], gpu)
            eps_chunk = jax.device_put(inv_eps_np[:, :, :, z0_int:z1_int], gpu)
            sig_E_chunk = (
                jax.device_put(sigma_E_np[:, :, :, z0_int:z1_int], gpu)  # type: ignore[index]
                if has_sigma_E else _dummy_sigma
            )
            kz_chunk = kappa_z_full[:, :, z0_int:z1_int]

            E_new, psi_E = _update_E_chunk(
                E_chunk, H_halo, eps_chunk, sig_E_chunk,
                psi_E, b_pml, a_pml,
                kappa_x, kappa_y, kz_chunk,
                z0_jax, c_num,
                Nz=Nz, Cz=Cz,
                periodic_axes=periodic_axes,
                has_conductivity=has_sigma_E,
            )

            E_np[:, :, :, z0_int:z1_int] = np.asarray(jax.device_get(E_new))

        # --- E-field sources (on CPU, after all chunks) ---
        _apply_sources_E(
            E_np, inv_eps_np, inv_mu_for_sources,
            objects, time_step, config.dtype,
        )

        # ==============================================================
        # Phase 2 — H update  (reads E, writes H)
        # ==============================================================
        for iz in range(n_chunks):
            if iz % 1 == 0:
                print(f"H update chunk {iz} of {n_chunks}")
            z0_int = iz * Cz
            z1_int = z0_int + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            E_halo = _build_E_halo_for_H(E_np, z0_int, z1_int, Nz, periodic_z, gpu)
            H_chunk = jax.device_put(H_np[:, :, :, z0_int:z1_int], gpu)

            if mu_is_scalar:
                mu_chunk = jnp.asarray(1.0, dtype=config.dtype)
            else:
                mu_chunk = jax.device_put(inv_mu_np[:, :, :, z0_int:z1_int], gpu)  # type: ignore[index]

            sig_H_chunk = (
                jax.device_put(sigma_H_np[:, :, :, z0_int:z1_int], gpu)  # type: ignore[index]
                if has_sigma_H else _dummy_sigma
            )
            kz_chunk = kappa_z_full[:, :, z0_int:z1_int]

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

            H_np[:, :, :, z0_int:z1_int] = np.asarray(jax.device_get(H_new))

        # --- H-field sources (on CPU, after all chunks) ---
        _apply_sources_H(
            H_np, inv_eps_np, inv_mu_for_sources,
            objects, time_step, config.dtype,
        )

        # ==============================================================
        # Detector update — lightweight per-detector loop that avoids
        # the full-grid interpolate_fields() allocation.
        # ==============================================================
        if has_detectors:
            detector_states = _update_detectors_lightweight(
                E_np, H_np, inv_eps_np, inv_mu_for_sources,
                detector_states, objects, time_step,
            )

    # ------------------------------------------------------------------
    # 5. Reconstruct output ArrayContainer
    #    Keep large arrays on CPU — they don't fit on GPU.
    # ------------------------------------------------------------------
    final_E = jnp.asarray(E_np)
    final_H = jnp.asarray(H_np)
    final_inv_eps = jnp.asarray(inv_eps_np)

    out = arrays
    out = out.aset("E", final_E)
    out = out.aset("H", final_H)
    out = out.aset("inv_permittivities", final_inv_eps)
    out = out.aset("psi_E", psi_E)
    out = out.aset("psi_H", psi_H)
    out = out.aset("detector_states", detector_states)
    if has_sigma_E:
        out = out.aset("electric_conductivity", jnp.asarray(sigma_E_np))
    if has_sigma_H:
        out = out.aset("magnetic_conductivity", jnp.asarray(sigma_H_np))
    if not mu_is_scalar:
        out = out.aset("inv_permeabilities", jnp.asarray(inv_mu_np))

    final_time = jnp.asarray(config.time_steps_total, dtype=jnp.int32)
    return (final_time, out)


# ---------------------------------------------------------------------------
# Source helpers — use jnp.asarray (zero-copy view of numpy) so the only
# allocation is the new array from source.update_{E,H}'s .at[].set().
# ---------------------------------------------------------------------------

def _apply_sources_E(
    E_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_permeabilities,
    objects: ObjectContainer,
    time_step: jax.Array,
    dtype,
) -> None:
    """Apply all active E-field sources to *E_np* in-place."""
    active = [
        s for s in objects.sources
        if bool(jax.device_get(s.is_on_at_time_step(time_step)))
    ]
    if not active:
        return

    E_jax = jnp.asarray(E_np, dtype=dtype)
    inv_eps_jax = jnp.asarray(inv_eps_np, dtype=dtype)
    for source in active:
        adj = source.adjust_time_step_by_on_off(time_step)
        E_jax = source.update_E(
            E=E_jax,
            inv_permittivities=inv_eps_jax,
            inv_permeabilities=inv_permeabilities,
            time_step=adj,
            inverse=False,
        )
    E_np[:] = np.asarray(E_jax)
    del E_jax


def _apply_sources_H(
    H_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_permeabilities,
    objects: ObjectContainer,
    time_step: jax.Array,
    dtype,
) -> None:
    """Apply all active H-field sources to *H_np* in-place."""
    active = [
        s for s in objects.sources
        if bool(jax.device_get(s.is_on_at_time_step(time_step)))
    ]
    if not active:
        return

    H_jax = jnp.asarray(H_np, dtype=dtype)
    inv_eps_jax = jnp.asarray(inv_eps_np, dtype=dtype)
    for source in active:
        adj = source.adjust_time_step_by_on_off(time_step)
        H_jax = source.update_H(
            H=H_jax,
            inv_permittivities=inv_eps_jax,
            inv_permeabilities=inv_permeabilities,
            time_step=adj + 0.5,
            inverse=False,
        )
    H_np[:] = np.asarray(H_jax)
    del H_jax


# ---------------------------------------------------------------------------
# Lightweight detector update — avoids the full-grid interpolate_fields()
# that pads the ENTIRE E and H arrays, which would double memory usage.
# Instead, calls each detector's .update() directly with zero-copy views.
# ---------------------------------------------------------------------------

def _update_detectors_lightweight(
    E_np: np.ndarray,
    H_np: np.ndarray,
    inv_eps_np: np.ndarray,
    inv_permeabilities,
    detector_states: dict,
    objects: ObjectContainer,
    time_step: jax.Array,
) -> dict:
    """Update detector states without allocating full-grid temporaries.

    Each detector's .update() only reads a small spatial slice of E/H,
    so wrapping the numpy arrays as zero-copy JAX views is safe — the
    only allocations are the small per-detector slices.
    """
    E_jax = jnp.asarray(E_np)
    H_jax = jnp.asarray(H_np)
    inv_eps_jax = jnp.asarray(inv_eps_np)

    for d in objects.forward_detectors:
        is_on = bool(jax.device_get(d._is_on_at_time_step_arr[time_step]))
        if not is_on:
            continue
        detector_states[d.name] = d.update(
            time_step=time_step,
            E=E_jax,
            H=H_jax,
            state=detector_states[d.name],
            inv_permittivity=inv_eps_jax,
            inv_permeability=inv_permeabilities,
        )

    del E_jax, H_jax, inv_eps_jax
    return detector_states
