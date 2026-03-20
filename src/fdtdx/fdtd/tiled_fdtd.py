"""Tiled FDTD for CPU-GPU streaming on unified memory systems (e.g. GH200).

All large arrays (E, H, material properties) are kept in CPU memory. Z-axis
chunks are streamed directly to the GPU in their native z-first (Cz, C, Nx, Ny) 
layout. The JIT kernels operate natively on this layout, ensuring perfect 
buffer donation and avoiding XLA reallocation page faults.
"""

from functools import partial
import datetime
import gc
import ctypes

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
    _update_sparse_psi,
)
from fdtdx.fdtd.container import (
    ArrayContainer,
    ObjectContainer,
    SimulationState,
)
from fdtdx.fdtd.update import get_periodic_axes


def _print_timestamp():
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# --- Setup CUDA Runtime via ctypes ---
def _load_cudart():
    for lib in ['libcudart.so', 'libcudart.so.12', 'libcudart.so.11.0']:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            pass
    raise OSError("Could not find libcudart.so. Ensure CUDA is installed.")


cudart = _load_cudart()
cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
CUDA_HOST_REGISTER_DEFAULT = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _check_cuda(err, func):
    if err != 0:
        raise RuntimeError(f"CUDA Error {err} in {func}")


def _memcpy_gpu_to_cpu(gpu_arr: jax.Array, cpu_arr: np.ndarray) -> None:
    _check_cuda(
        cudart.cudaMemcpy(cpu_arr.ctypes.data, gpu_arr.unsafe_buffer_pointer(), gpu_arr.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST),
        "cudaMemcpy (D2H)"
    )


def _memcpy_cpu_to_gpu(cpu_arr: np.ndarray, gpu_arr: jax.Array) -> None:
    _check_cuda(
        cudart.cudaMemcpy(gpu_arr.unsafe_buffer_pointer(), cpu_arr.ctypes.data, cpu_arr.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE),
        "cudaMemcpy (H2D)"
    )


def _register_cpu_memory(cpu_arr: np.ndarray) -> None:
    _check_cuda(
        cudart.cudaHostRegister(cpu_arr.ctypes.data, cpu_arr.nbytes, CUDA_HOST_REGISTER_DEFAULT),
        "cudaHostRegister (cpu_arr)"
    )


def _unregister_cpu_memory(cpu_arr: np.ndarray) -> None:
    _check_cuda(
        cudart.cudaHostUnregister(cpu_arr.ctypes.data),
        "cudaHostUnregister (cpu_arr)"
    )


def _memcpy_to_cpu_and_transpose(gpu_arr: jax.Array, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    # 1. Allocate the final shape natively in pinned CPU memory.
    cpu_arr = np.empty((Nx, Ny, Nz, 3), dtype=np.float32)
    _register_cpu_memory(cpu_arr)

    # 2. Process the transpose on the GPU in safe, bite-sized chunks to prevent OOM
    chunk_size = 1  
    for i in range(0, Nx, chunk_size):
        end_i = min(i + chunk_size, Nx)
        
        # --- STEP A: Slice on GPU ---
        # Shape: (3, chunk_size, Ny, Nz)
        gpu_slice = gpu_arr[:, i:end_i, :, :]
        
        # --- STEP B: Transpose on GPU ---
        # JAX transpose is also just a view. We MUST force a physical copy 
        # on the GPU so the memory becomes physically contiguous for the DMA transfer.
        gpu_slice_T = jnp.transpose(gpu_slice, (1, 2, 3, 0))
        gpu_slice_T_contiguous = jnp.copy(gpu_slice_T)
        gpu_slice_T_contiguous.block_until_ready()
        
        # --- STEP C: DMA Transfer to Pinned CPU Memory ---
        src_ptr = gpu_slice_T_contiguous.unsafe_buffer_pointer()
        
        # Because cpu_arr is contiguous on axis 0, slicing it gives us the 
        # exact, contiguous byte offset we need for the destination pointer!
        dst_slice = cpu_arr[i:end_i, :, :, :]
        dst_ptr = dst_slice.ctypes.data
        size = dst_slice.nbytes
        
        _memcpy_gpu_to_cpu(gpu_slice_T_contiguous, dst_slice)
        
        # --- STEP D: Free transient GPU memory for the next loop ---
        del gpu_slice, gpu_slice_T, gpu_slice_T_contiguous
        
    return cpu_arr


def _pad_yz(field, periodic_axes):
    # Field layout is (Cx+1, Ny, Nz, 3). Pad axes 1 (y) and 2 (z).
    for i in (1, 2):
        mode = "wrap" if periodic_axes[i] else "constant"
        pw = [(0, 0)] * 4
        pw[i] = (1, 1)
        field = jnp.pad(field, pw, mode=mode)
    return field

_CURL_IDX = (0, 0, 1, 1, 2, 2)
_CURL_SIGN = (+1.0, -1.0, +1.0, -1.0, +1.0, -1.0)


def _pml_update_x_xf(psi_min, psi_max, d_field, b_coeff, a_coeff, x0_int, Cx, Nx):
    """Update psi for axis 0 (x, the chunked axis)."""
    L_min, L_max = psi_min.shape[0], psi_max.shape[0]
    x1_int = x0_int + Cx

    if L_min > 0:
        b_min = _extract_pml_slab(b_coeff, 0, L_min, "min")
        a_min = _extract_pml_slab(a_coeff, 0, L_min, "min")
        psi_idx = jnp.arange(L_min)
        in_chunk = (psi_idx >= x0_int) & (psi_idx < x1_int)
        chunk_local = jnp.clip(psi_idx - x0_int, 0, Cx - 1)
        d_gathered = d_field[chunk_local, :, :]
        psi_min_new = b_min * psi_min + a_min * d_gathered
        psi_min = jnp.where(in_chunk[:, None, None], psi_min_new, psi_min)

    if L_max > 0:
        b_max = _extract_pml_slab(b_coeff, 0, L_max, "max")
        a_max = _extract_pml_slab(a_coeff, 0, L_max, "max")
        psi_global = Nx - L_max + jnp.arange(L_max)
        in_chunk = (psi_global >= x0_int) & (psi_global < x1_int)
        chunk_local = jnp.clip(psi_global - x0_int, 0, Cx - 1)
        d_gathered = d_field[chunk_local, :, :]
        psi_max_new = b_max * psi_max + a_max * d_gathered
        psi_max = jnp.where(in_chunk[:, None, None], psi_max_new, psi_max)

    return psi_min, psi_max


def _scatter_x_psi_xf(curl_comp, psi_min, psi_max, sign, x0_int, Cx, Nx):
    """Scatter x-axis psi into chunk curl component."""
    L_min, L_max = psi_min.shape[0], psi_max.shape[0]
    local_x = jnp.arange(Cx)
    global_x = local_x + x0_int

    if L_min > 0:
        in_pml_min = global_x < L_min
        safe_idx = jnp.clip(global_x, 0, L_min - 1)
        curl_comp = curl_comp + jnp.where(in_pml_min[:, None, None], sign * psi_min[safe_idx, :, :], 0.0)

    if L_max > 0:
        pml_max_start = Nx - L_max
        in_pml_max = global_x >= pml_max_start
        safe_idx = jnp.clip(global_x - pml_max_start, 0, L_max - 1)
        curl_comp = curl_comp + jnp.where(in_pml_max[:, None, None], sign * psi_max[safe_idx, :, :], 0.0)

    return curl_comp


@partial(
    jax.jit, 
    donate_argnames=("E_chunk", "psi_E"),
    static_argnames=("Cx", "Nx", "periodic_axes", "courant_number"), 
)
def _update_E_chunk(
    E_chunk,
    H_halo,
    inv_eps_chunk,
    sigma_E_chunk,
    psi_E,
    courant_number,
    a_pml, b_pml,
    kappa_x, kappa_y, kappa_z,
    periodic_axes,
    x0_int, Cx, Nx,
):
    H_halo = _pad_yz(H_halo, periodic_axes) # shape: (Cx+1, Ny+2, Nz+2, 3)
    Hx, Hy, Hz = H_halo[:, :, :, 0], H_halo[:, :, :, 1], H_halo[:, :, :, 2]
    kappa_x_chunk = lax.dynamic_slice(kappa_x, (x0_int, 0, 0), (Cx, kappa_x.shape[1], kappa_x.shape[2]))

    dyHz = Hz[1:, 1:-1, 1:-1] - Hz[1:, :-2, 1:-1]
    dyHx = Hx[1:, 1:-1, 1:-1] - Hx[1:, :-2, 1:-1]
    dxHz = Hz[1:, 1:-1, 1:-1] - Hz[:-1, 1:-1, 1:-1]
    dxHy = Hy[1:, 1:-1, 1:-1] - Hy[:-1, 1:-1, 1:-1]
    dzHy = Hy[1:, 1:-1, 1:-1] - Hy[1:, 1:-1, :-2]
    dzHx = Hx[1:, 1:-1, 1:-1] - Hx[1:, 1:-1, :-2]

    d_fields = (dyHz, dzHy, dzHx, dxHz, dxHy, dyHx)

    curl_x = (1.0 / kappa_y) * dyHz - (1.0 / kappa_z) * dzHy
    curl_y = (1.0 / kappa_z) * dzHx - (1.0 / kappa_x_chunk) * dxHz
    curl_z = (1.0 / kappa_x_chunk) * dxHy - (1.0 / kappa_y) * dyHx
    curls = [curl_x, curl_y, curl_z]

    psi_list = list(psi_E)
    for i in range(6):
        axis = PSI_COMPONENT_AXIS[i]
        ci, cidx, sign = PSI_E_COEFF_IDX[i], _CURL_IDX[i], _CURL_SIGN[i]
        psi_min_i, psi_max_i = psi_list[i]

        if axis == 0:
            psi_min_i, psi_max_i = _pml_update_x_xf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci],
                x0_int, Cx, Nx,
            )
            curls[cidx] = _scatter_x_psi_xf(curls[cidx], psi_min_i, psi_max_i, sign, x0_int, Cx, Nx)
        else:
            psi_min_c = lax.dynamic_slice(psi_min_i, (x0_int, 0, 0), (Cx, psi_min_i.shape[1], psi_min_i.shape[2]))
            psi_max_c = lax.dynamic_slice(psi_max_i, (x0_int, 0, 0), (Cx, psi_max_i.shape[1], psi_max_i.shape[2]))
            psi_min_c, psi_max_c = _update_sparse_psi(
                psi_min_c, psi_max_c, b_pml[ci], a_pml[ci], d_fields[i], axis,
            )
            psi_min_i = lax.dynamic_update_slice(psi_min_i, psi_min_c, (x0_int, 0, 0))
            psi_max_i = lax.dynamic_update_slice(psi_max_i, psi_max_c, (x0_int, 0, 0))
            curls[cidx] = _scatter_psi_component(curls[cidx], psi_min_c, psi_max_c, axis, sign)

        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=3) # shape: (Cx, Ny, Nz, 3)
    c = courant_number
    if sigma_E_chunk is not None:
        loss = c * sigma_E_chunk * eta0 * inv_eps_chunk / 2
        E_new = (1 - loss) * E_chunk + c * curl * inv_eps_chunk
        E_new = E_new / (1 + loss)
    else:
        E_new = E_chunk + c * curl * inv_eps_chunk

    return E_new, tuple(psi_list)


@partial(
    jax.jit,
    donate_argnames=("H_chunk", "psi_H"),
    static_argnames=("Cx", "Nx", "periodic_axes", "courant_number"),
)
def _update_H_chunk(
    H_chunk,
    E_halo,
    inv_mu_chunk,
    sigma_H_chunk,
    psi_H,
    courant_number,
    a_pml, b_pml,
    kappa_x, kappa_y, kappa_z,
    periodic_axes,
    x0_int, Cx, Nx,
):
    E_halo = _pad_yz(E_halo, periodic_axes)  # shape: (Cx+1, Ny+2, Nz+2, 3)
    Ex, Ey, Ez = E_halo[:, :, :, 0], E_halo[:, :, :, 1], E_halo[:, :, :, 2]
    kappa_x_chunk = lax.dynamic_slice(kappa_x, (x0_int, 0, 0), (Cx, kappa_x.shape[1], kappa_x.shape[2]))

    dyEz = Ez[:-1, 2:, 1:-1] - Ez[:-1, 1:-1, 1:-1]
    dyEx = Ex[:-1, 2:, 1:-1] - Ex[:-1, 1:-1, 1:-1]
    dxEz = Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]
    dxEy = Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]
    dzEy = Ey[:-1, 1:-1, 2:] - Ey[:-1, 1:-1, 1:-1]
    dzEx = Ex[:-1, 1:-1, 2:] - Ex[:-1, 1:-1, 1:-1]

    d_fields = (dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)

    curl_x = (1.0 / kappa_y) * dyEz - (1.0 / kappa_z) * dzEy
    curl_y = (1.0 / kappa_z) * dzEx - (1.0 / kappa_x_chunk) * dxEz
    curl_z = (1.0 / kappa_x_chunk) * dxEy - (1.0 / kappa_y) * dyEx
    curls = [curl_x, curl_y, curl_z]

    psi_list = list(psi_H)
    for i in range(6):
        axis = PSI_COMPONENT_AXIS[i]
        ci, cidx, sign = PSI_H_COEFF_IDX[i], _CURL_IDX[i], _CURL_SIGN[i]
        psi_min_i, psi_max_i = psi_list[i]

        if axis == 0:
            psi_min_i, psi_max_i = _pml_update_x_xf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci],
                x0_int, Cx, Nx,
            )
            curls[cidx] = _scatter_x_psi_xf(curls[cidx], psi_min_i, psi_max_i, sign, x0_int, Cx, Nx)
        else:
            psi_min_c = lax.dynamic_slice(psi_min_i, (x0_int, 0, 0), (Cx, psi_min_i.shape[1], psi_min_i.shape[2]))
            psi_max_c = lax.dynamic_slice(psi_max_i, (x0_int, 0, 0), (Cx, psi_max_i.shape[1], psi_max_i.shape[2]))
            psi_min_c, psi_max_c = _update_sparse_psi(
                psi_min_c, psi_max_c, b_pml[ci], a_pml[ci], d_fields[i], axis,
            )
            psi_min_i = lax.dynamic_update_slice(psi_min_i, psi_min_c, (x0_int, 0, 0))
            psi_max_i = lax.dynamic_update_slice(psi_max_i, psi_max_c, (x0_int, 0, 0))
            curls[cidx] = _scatter_psi_component(curls[cidx], psi_min_c, psi_max_c, axis, sign)

        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=3)  # shape: (Cx, Ny, Nz, 3)
    c = courant_number
    inv_mu = inv_mu_chunk if inv_mu_chunk is not None else 1.0
    if sigma_H_chunk is not None:
        loss = c * sigma_H_chunk * inv_mu / eta0 / 2
        H_new = (1 - loss) * H_chunk - c * curl * inv_mu
        H_new = H_new / (1 + loss)
    else:
        H_new = H_chunk - c * curl * inv_mu

    return H_new, tuple(psi_list)


# ---------------------------------------------------------------------------
# Source and detector helpers (x-first layout: Nx, Ny, Nz, 3)
# ---------------------------------------------------------------------------

def _remap_to_gpu_xfirst(obj, field_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, gpu_device):
    """Extract object region from x-first (Nx, Ny, Nz, 3) arrays and remap for GPU update.
    Returns (obj_remap, field_gpu, eps_gpu, mu_gpu).
    inv_mu_np can be None when mu_is_scalar is True.
    """
    gst = obj._grid_slice_tuple
    sx, sy, sz = obj.grid_slice
    dx = gst[0][1] - gst[0][0]
    dy = gst[1][1] - gst[1][0]
    dz = gst[2][1] - gst[2][0]

    # field_np is (Nx, Ny, Nz, 3); slice gives (dx, dy, dz, 3)
    # Source/detector interface expects (3, Nx, Ny, Nz) -> transpose to (3, dx, dy, dz)
    field_slice = field_np[sx, sy, sz, :]
    field_gpu = jax.device_put(
        jnp.asarray(np.ascontiguousarray(np.transpose(field_slice, (3, 0, 1, 2)))),
        gpu_device,
    )
    eps_slice = inv_eps_np[sx, sy, sz, :]
    eps_gpu = jax.device_put(
        jnp.asarray(np.ascontiguousarray(np.transpose(eps_slice, (3, 0, 1, 2)))),
        gpu_device,
    )
    if mu_is_scalar:
        mu_gpu = inv_mu_scalar
    else:
        mu_slice = inv_mu_np[sx, sy, sz, :]
        mu_gpu = jax.device_put(
            jnp.asarray(np.ascontiguousarray(np.transpose(mu_slice, (3, 0, 1, 2)))),
            gpu_device,
        )
    obj_remap = obj.aset("_grid_slice_tuple", ((0, dx), (0, dy), (0, dz)))
    return obj_remap, field_gpu, eps_gpu, mu_gpu


def _apply_sources_E(
    E_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
    has_inv_permeabilities, inv_mu_scalar, objects, time_step, gpu_device,
):
    """Apply E-field sources. E_cpu and inv_permittivities_cpu are (Nx, Ny, Nz, 3).
    inv_permeabilities_cpu can be None when has_inv_permeabilities is False.
    """
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))):
            continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice
        src_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu_xfirst(
            source, E_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
            not has_inv_permeabilities, inv_mu_scalar, gpu_device,
        )
        E_updated = src_remap.update_E(
            E=E_gpu, inv_permittivities=eps_gpu, inv_permeabilities=mu_gpu,
            time_step=adj, inverse=False,
        )
        # E_updated is (3, dx, dy, dz); write back as (dx, dy, dz, 3)
        E_cpu[sx, sy, sz, :] = np.asarray(jax.device_get(E_updated)).transpose((1, 2, 3, 0))


def _apply_sources_H(
    H_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
    has_inv_permeabilities, inv_mu_scalar, objects, time_step, gpu_device,
):
    """Apply H-field sources. H_cpu and inv_permittivities_cpu are (Nx, Ny, Nz, 3).
    inv_permeabilities_cpu can be None when has_inv_permeabilities is False.
    """
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))):
            continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice
        src_remap, H_gpu, eps_gpu, mu_gpu = _remap_to_gpu_xfirst(
            source, H_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
            not has_inv_permeabilities, inv_mu_scalar, gpu_device,
        )
        H_updated = src_remap.update_H(
            H=H_gpu, inv_permittivities=eps_gpu, inv_permeabilities=mu_gpu,
            time_step=adj + 0.5, inverse=False,
        )
        H_cpu[sx, sy, sz, :] = np.asarray(jax.device_get(H_updated)).transpose((1, 2, 3, 0))


def _update_detectors(
    E_cpu, H_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
    has_inv_permeabilities, inv_mu_scalar, detector_states, objects, time_step, gpu_device,
):
    """Update detector states. E_cpu, H_cpu are (Nx, Ny, Nz, 3).
    inv_permeabilities_cpu can be None when has_inv_permeabilities is False.
    """
    for d in objects.forward_detectors:
        if not bool(jax.device_get(d._is_on_at_time_step_arr[time_step])):
            continue
        d_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu_xfirst(
            d, E_cpu, inv_permittivities_cpu, inv_permeabilities_cpu,
            not has_inv_permeabilities, inv_mu_scalar, gpu_device,
        )
        sx, sy, sz = d.grid_slice
        H_slice = H_cpu[sx, sy, sz, :]
        H_gpu = jax.device_put(
            jnp.asarray(np.ascontiguousarray(np.transpose(H_slice, (3, 0, 1, 2)))),
            gpu_device,
        )
        detector_states[d.name] = d_remap.update(
            time_step=time_step, E=E_gpu, H=H_gpu, state=detector_states[d.name],
            inv_permittivity=eps_gpu, inv_permeability=mu_gpu,
        )
    return detector_states


def tiled_fdtd(
    arrays: ArrayContainer, 
    objects: ObjectContainer, 
    config: SimulationConfig, 
    key: jax.Array, 
    chunk_size: int | None = None,
) -> SimulationState:

    try:
        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]
        print(f"Host Device: {cpu_device}")
        print(f"Compute Device: {gpu_device}")
    except IndexError:
        print("Error: Could not find both CPU and GPU. Check your JAX installation.")
        return

    _, Nx, Ny, Nz = arrays.E.shape 
    periodic_axes = get_periodic_axes(objects)
    Cx = chunk_size if chunk_size is not None else 1

    has_sources = bool(objects.sources)
    has_detectors = bool(objects.forward_detectors)
    inv_mu_val = arrays.inv_permeabilities
    mu_is_scalar = inv_mu_val is None or not isinstance(inv_mu_val, jax.Array) or inv_mu_val.ndim == 0
    inv_mu_scalar = (
        1.0 if (mu_is_scalar and inv_mu_val is None)
        else (float(jax.device_get(jnp.asarray(inv_mu_val))) if mu_is_scalar else None)
    )
    n_chunks = (Nx + Cx - 1) // Cx
    tail_size = Nx - (n_chunks - 1) * Cx
    x0_gpu_list = [jax.device_put(jnp.int32(ix * Cx), gpu_device) for ix in range(n_chunks)]
    print(f"Chunk size: {Cx}")
    print(f"Number of chunks: {n_chunks}")

    b_pml, a_pml = _compute_pml_ab(arrays.alpha, arrays.kappa, arrays.sigma, config)

    _print_timestamp()
    print("Copying arrays to CPU memory...")

    print("  Copying E to CPU memory...")
    arrays.E.block_until_ready()
    E_cpu = _memcpy_to_cpu_and_transpose(arrays.E, Nx, Ny, Nz)

    E_halo_buffer_cpu = np.empty((tail_size + 1, Ny, Nz, 3), dtype=np.float32)
    _check_cuda(
        cudart.cudaHostRegister(E_halo_buffer_cpu.ctypes.data, E_halo_buffer_cpu.nbytes, CUDA_HOST_REGISTER_DEFAULT),
        "cudaHostRegister (E_halo_buffer_cpu)"
    )

    E_chunk_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    E_chunk_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    E_halo_gpu_main = jax.device_put(jnp.zeros((Cx + 1, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    E_halo_gpu_tail = jax.device_put(jnp.zeros((tail_size + 1, Ny, Nz, 3), dtype=jnp.float32), gpu_device)

    print("  Copying H to CPU memory...")
    arrays.H.block_until_ready()
    H_cpu = _memcpy_to_cpu_and_transpose(arrays.H, Nx, Ny, Nz)

    H_halo_buffer_cpu = np.empty((Cx + 1, Ny, Nz, 3), dtype=np.float32)
    _check_cuda(
        cudart.cudaHostRegister(H_halo_buffer_cpu.ctypes.data, H_halo_buffer_cpu.nbytes, CUDA_HOST_REGISTER_DEFAULT),
        "cudaHostRegister (H_halo_buffer_cpu)"
    )

    H_chunk_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    H_chunk_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    H_halo_gpu_main = jax.device_put(jnp.zeros((Cx + 1, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    H_halo_gpu_tail = jax.device_put(jnp.zeros((tail_size + 1, Ny, Nz, 3), dtype=jnp.float32), gpu_device)

    print("  Copying inv_permittivities to CPU memory...")
    arrays.inv_permittivities.block_until_ready()
    inv_permittivities_cpu = _memcpy_to_cpu_and_transpose(arrays.inv_permittivities, Nx, Ny, Nz)

    inv_permittivities_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
    inv_permittivities_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)

    has_inv_permeabilities = False
    if arrays.inv_permeabilities is not None and isinstance(arrays.inv_permeabilities, jax.Array):
        print("  Copying inv_permeabilities to CPU memory...")
        arrays.inv_permeabilities.block_until_ready()
        inv_permeabilities_cpu = _memcpy_to_cpu_and_transpose(arrays.inv_permeabilities, Nx, Ny, Nz)
        inv_permeabilities_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        inv_permeabilities_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        has_inv_permeabilities = True

    has_electric_conductivity = False
    if arrays.electric_conductivity is not None:
        print("  Copying electric_conductivity to CPU memory...")
        arrays.electric_conductivity.block_until_ready()
        electric_conductivity_cpu = _memcpy_to_cpu_and_transpose(arrays.electric_conductivity, Nx, Ny, Nz)
        electric_conductivity_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        electric_conductivity_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        has_electric_conductivity = True

    has_magnetic_conductivity = False
    if arrays.magnetic_conductivity is not None:
        print("  Copying magnetic_conductivity to CPU memory...")
        arrays.magnetic_conductivity.block_until_ready()
        magnetic_conductivity_cpu = _memcpy_to_cpu_and_transpose(arrays.magnetic_conductivity, Nx, Ny, Nz)
        magnetic_conductivity_gpu_main = jax.device_put(jnp.zeros((Cx, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        magnetic_conductivity_gpu_tail = jax.device_put(jnp.zeros((tail_size, Ny, Nz, 3), dtype=jnp.float32), gpu_device)
        has_magnetic_conductivity = True

    new_arrays = ArrayContainer(
        E=None,
        H=None,
        psi_E=arrays.psi_E,
        psi_H=arrays.psi_H,
        alpha=arrays.alpha,
        kappa=arrays.kappa,
        sigma=arrays.sigma,
        inv_permittivities=None,
        inv_permeabilities=arrays.inv_permeabilities if not has_inv_permeabilities else None,
        detector_states=arrays.detector_states,
        recording_state=arrays.recording_state,
        electric_conductivity=None,
        magnetic_conductivity=None,
    )
    b_pml, a_pml = _compute_pml_ab(new_arrays.alpha, new_arrays.kappa, new_arrays.sigma, config)
    kappa_Ex = new_arrays.kappa[0]
    kappa_Ey = new_arrays.kappa[1]
    kappa_Ez = new_arrays.kappa[2]
    kappa_Hx = new_arrays.kappa[3]
    kappa_Hy = new_arrays.kappa[4]
    kappa_Hz = new_arrays.kappa[5]
    psi_E = new_arrays.psi_E
    psi_H = new_arrays.psi_H
    detector_states = new_arrays.detector_states

    del arrays
    gc.collect()

    _print_timestamp()
    print("Arrays copied to CPU memory")

    _print_timestamp()
    print(f"Time loop starting")

    for t in range(config.time_steps_total):
        time_step = jax.device_put(jnp.int32(t), gpu_device)
        if t % 50 == 0:
            _print_timestamp()
            print(f"Time step {t} of {config.time_steps_total}")

        # ==============================================================
        # Phase 1 — E update 
        # ==============================================================
        for ix in range(n_chunks):
            x0 = ix * Cx
            x1 = min((ix + 1) * Cx, Nx)
            actual_chunk_size = x1-x0
            x0_gpu = x0_gpu_list[ix]

            if actual_chunk_size == Cx:
                E_chunk_gpu = E_chunk_gpu_main
                inv_eps_chunk_gpu = inv_permittivities_gpu_main
                if has_electric_conductivity:
                    sigma_E_chunk_gpu = electric_conductivity_gpu_main
                H_halo_gpu = H_halo_gpu_main
            else:
                E_chunk_gpu = E_chunk_gpu_tail
                inv_eps_chunk_gpu = inv_permittivities_gpu_tail
                if has_electric_conductivity:
                    sigma_E_chunk_gpu = electric_conductivity_gpu_tail
                H_halo_gpu = H_halo_gpu_tail
            
            E_chunk_cpu = E_cpu[x0:x1, :, :, :]
            _memcpy_cpu_to_gpu(E_chunk_cpu, E_chunk_gpu)
            E_chunk_gpu.block_until_ready()
            inv_eps_chunk_cpu = inv_permittivities_cpu[x0:x1, :, :, :]
            _memcpy_cpu_to_gpu(inv_eps_chunk_cpu, inv_eps_chunk_gpu)
            inv_eps_chunk_gpu.block_until_ready()
            if has_electric_conductivity:
                sigma_E_chunk_cpu = electric_conductivity_cpu[x0:x1, :, :, :]
                _memcpy_cpu_to_gpu(sigma_E_chunk_cpu, sigma_E_chunk_gpu)
                sigma_E_chunk_gpu.block_until_ready()
            else:
                sigma_E_chunk_cpu = None
                sigma_E_chunk_gpu = None

            if x0 > 0:
                H_halo_cpu = H_cpu[x0-1:x1, :, :, :]
            else:
                if periodic_axes[0]:
                    H_halo_buffer_cpu[0, :, :, :] = H_cpu[-1, :, :, :]
                else:
                    H_halo_buffer_cpu[0, :, :, :] = 0
                H_halo_buffer_cpu[1:, :, :, :] = H_cpu[x0:x1, :, :, :]
                H_halo_cpu = H_halo_buffer_cpu
            _memcpy_cpu_to_gpu(H_halo_cpu, H_halo_gpu)
            H_halo_gpu.block_until_ready() 

            E_chunk_gpu, psi_E = _update_E_chunk(
                E_chunk_gpu,
                H_halo_gpu,
                inv_eps_chunk_gpu,
                sigma_E_chunk_gpu,
                psi_E,
                config.courant_number,
                a_pml, b_pml,
                kappa_Ex, kappa_Ey, kappa_Ez,
                periodic_axes,
                x0_gpu, 
                actual_chunk_size, 
                Nx,
            )
            E_chunk_gpu.block_until_ready()
            jax.block_until_ready(psi_E)

            _memcpy_gpu_to_cpu(E_chunk_gpu, E_chunk_cpu)

            if actual_chunk_size == Cx:
                E_chunk_gpu_main = E_chunk_gpu
                inv_eps_chunk_gpu_main = inv_eps_chunk_gpu
                if has_electric_conductivity:
                    sigma_E_chunk_gpu_main = sigma_E_chunk_gpu
                H_halo_gpu_main = H_halo_gpu
            else:
                E_chunk_gpu_tail = E_chunk_gpu
                inv_eps_chunk_gpu_tail = inv_eps_chunk_gpu
                if has_electric_conductivity:
                    sigma_E_chunk_gpu_tail = sigma_E_chunk_gpu
                H_halo_gpu_tail = H_halo_gpu

            del H_halo_cpu
            del E_chunk_cpu
            del inv_eps_chunk_cpu
            del sigma_E_chunk_cpu

        # ==============================================================
        # Phase 1a — E sources
        # ==============================================================
        if has_sources:
            _apply_sources_E(
                E_cpu, inv_permittivities_cpu,
                inv_permeabilities_cpu if has_inv_permeabilities else None,
                has_inv_permeabilities, inv_mu_scalar, objects, time_step, gpu_device,
            )

        # ==============================================================
        # Phase 2 — H update 
        # ==============================================================
        for ix in range(n_chunks):
            x0 = ix * Cx
            x1 = min((ix + 1) * Cx, Nx)
            actual_chunk_size = x1-x0
            x0_gpu = x0_gpu_list[ix]

            if actual_chunk_size == Cx:
                H_chunk_gpu = H_chunk_gpu_main
                if has_inv_permeabilities:
                    inv_mu_chunk_gpu = inv_permeabilities_gpu_main
                if has_magnetic_conductivity:
                    sigma_H_chunk_gpu = magnetic_conductivity_gpu_main
                E_halo_gpu = E_halo_gpu_main
            else:
                H_chunk_gpu = H_chunk_gpu_tail
                if has_inv_permeabilities:
                    inv_mu_chunk_gpu = inv_permeabilities_gpu_tail
                if has_magnetic_conductivity:
                    sigma_H_chunk_gpu = magnetic_conductivity_gpu_tail
                E_halo_gpu = E_halo_gpu_tail

            H_chunk_cpu = H_cpu[x0:x1, :, :, :]
            _memcpy_cpu_to_gpu(H_chunk_cpu, H_chunk_gpu)
            H_chunk_gpu.block_until_ready()
            if has_inv_permeabilities:
                inv_mu_chunk_cpu = inv_permeabilities_cpu[x0:x1, :, :, :]
                _memcpy_cpu_to_gpu(inv_mu_chunk_cpu, inv_mu_chunk_gpu)
                inv_mu_chunk_gpu.block_until_ready()
            else:
                inv_mu_chunk_cpu = None
                inv_mu_chunk_gpu = new_arrays.inv_permeabilities
            if has_magnetic_conductivity:
                sigma_H_chunk_cpu = magnetic_conductivity_cpu[x0:x1, :, :, :]
                _memcpy_cpu_to_gpu(sigma_H_chunk_cpu, sigma_H_chunk_gpu)
                sigma_H_chunk_gpu.block_until_ready()
            else:
                sigma_H_chunk_cpu = None
                sigma_H_chunk_gpu = None

            if x1 < Nx:
                E_halo_cpu = E_cpu[x0:x1+1, :, :, :]
            else:
                if periodic_axes[0]:
                    E_halo_buffer_cpu[-1, :, :, :] = E_cpu[0, :, :, :]
                else:
                    E_halo_buffer_cpu[-1, :, :, :] = 0
                E_halo_buffer_cpu[0:-1, :, :, :] = E_cpu[x0:x1, :, :, :]
                E_halo_cpu = E_halo_buffer_cpu
            _memcpy_cpu_to_gpu(E_halo_cpu, E_halo_gpu)
            E_halo_gpu.block_until_ready()

            H_chunk_gpu, psi_H = _update_H_chunk(
                H_chunk_gpu,
                E_halo_gpu,
                inv_mu_chunk_gpu,
                sigma_H_chunk_gpu,
                psi_H,
                config.courant_number,
                a_pml, b_pml,
                kappa_Hx, kappa_Hy, kappa_Hz,
                periodic_axes,
                x0_gpu, 
                actual_chunk_size, 
                Nx,
            )
            H_chunk_gpu.block_until_ready()
            jax.block_until_ready(psi_H)

            _memcpy_gpu_to_cpu(H_chunk_gpu, H_chunk_cpu)

            if actual_chunk_size == Cx:
                H_chunk_gpu_main = H_chunk_gpu
                if has_inv_permeabilities:
                    inv_mu_chunk_gpu_main = inv_mu_chunk_gpu
                if has_magnetic_conductivity:
                    sigma_H_chunk_gpu_main = sigma_H_chunk_gpu
                E_halo_gpu_main = E_halo_gpu
            else:
                H_chunk_gpu_tail = H_chunk_gpu
                if has_inv_permeabilities:
                    inv_mu_chunk_gpu_tail = inv_mu_chunk_gpu
                if has_magnetic_conductivity:
                    sigma_H_chunk_gpu_tail = sigma_H_chunk_gpu
                E_halo_gpu_tail = E_halo_gpu

            del E_halo_cpu
            del H_chunk_cpu
            del sigma_H_chunk_cpu

        # ==============================================================
        # Phase 2a — H sources
        # ==============================================================
        if has_sources:
            _apply_sources_H(
                H_cpu, inv_permittivities_cpu,
                inv_permeabilities_cpu if has_inv_permeabilities else None,
                has_inv_permeabilities, inv_mu_scalar, objects, time_step, gpu_device,
            )

        # ==============================================================
        # Phase 3 — Detectors
        # ==============================================================
        if has_detectors:
            detector_states = _update_detectors(
                E_cpu, H_cpu, inv_permittivities_cpu,
                inv_permeabilities_cpu if has_inv_permeabilities else None,
                has_inv_permeabilities, inv_mu_scalar, detector_states,
                objects, time_step, gpu_device,
            )

        #if t >= 10: break

    _print_timestamp()
    print(f"Time loop completed")

    # Delete GPU arrays
    E_chunk_gpu_main.delete()
    if Nx % Cx != 0: E_chunk_gpu_tail.delete()
    inv_eps_chunk_gpu_main.delete()
    if Nx % Cx != 0: inv_eps_chunk_gpu_tail.delete()
    if has_electric_conductivity:
        sigma_E_chunk_gpu_main.delete()
        if Nx % Cx != 0: sigma_E_chunk_gpu_tail.delete()
    H_chunk_gpu_main.delete()
    if Nx % Cx != 0: H_chunk_gpu_tail.delete()
    if has_inv_permeabilities:
        inv_mu_chunk_gpu_main.delete()
        if Nx % Cx != 0: inv_mu_chunk_gpu_tail.delete()
    if has_magnetic_conductivity:
        sigma_H_chunk_gpu_main.delete()
        if Nx % Cx != 0: sigma_H_chunk_gpu_tail.delete()
    E_halo_gpu_main.delete()
    if Nx % Cx != 0: E_halo_gpu_tail.delete()
    H_halo_gpu_main.delete()
    if Nx % Cx != 0: H_halo_gpu_tail.delete()

    gc.collect()

    # Reconstruct ArrayContainer — convert from (Nx, Ny, Nz, 3) to (3, Nx, Ny, Nz)
    final_inv_eps = jnp.asarray(np.ascontiguousarray(np.transpose(inv_permittivities_cpu, (3, 0, 1, 2))))
    final_sigma_E = (
        jnp.asarray(np.ascontiguousarray(np.transpose(electric_conductivity_cpu, (3, 0, 1, 2))))
        if has_electric_conductivity else None
    )
    final_sigma_H = (
        jnp.asarray(np.ascontiguousarray(np.transpose(magnetic_conductivity_cpu, (3, 0, 1, 2))))
        if has_magnetic_conductivity else None
    )
    final_inv_mu = (
        jnp.asarray(inv_mu_scalar)
        if mu_is_scalar
        else jnp.asarray(np.ascontiguousarray(np.transpose(inv_permeabilities_cpu, (3, 0, 1, 2))))
    )

    out = ArrayContainer(
        E=None,
        H=None,
        psi_E=None,
        psi_H=None,
        alpha=None,
        kappa=None,
        sigma=None,
        inv_permittivities=final_inv_eps,
        inv_permeabilities=final_inv_mu,
        detector_states=detector_states,
        recording_state=new_arrays.recording_state,
        electric_conductivity=final_sigma_E,
        magnetic_conductivity=final_sigma_H,
    )

    # Unregister CPU memory
    _unregister_cpu_memory(E_cpu)
    _unregister_cpu_memory(E_halo_buffer_cpu)
    _unregister_cpu_memory(H_cpu)
    _unregister_cpu_memory(H_halo_buffer_cpu)
    _unregister_cpu_memory(inv_permittivities_cpu)
    if has_inv_permeabilities:
        _unregister_cpu_memory(inv_permeabilities_cpu)
    if has_electric_conductivity:
        _unregister_cpu_memory(electric_conductivity_cpu)
    if has_magnetic_conductivity:
        _unregister_cpu_memory(magnetic_conductivity_cpu)

    
    return (jnp.asarray(t, dtype=jnp.int32), out)