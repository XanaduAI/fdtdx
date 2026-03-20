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
)
from fdtdx.fdtd.container import (
    ArrayContainer,
    ObjectContainer,
    SimulationState,
)
from fdtdx.fdtd.update import get_periodic_axes

def print_timestamp():
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------------
# Memory placement helpers for GH200
# ---------------------------------------------------------------------------

def _free_jax_buffer(arr) -> None:
    if not isinstance(arr, jax.Array):
        return
    try:
        arr.delete()
    except Exception:
        pass

def _cuda_pin(arr: np.ndarray, label: str = "") -> bool:
    if not arr.flags['C_CONTIGUOUS']:
        print(f"  [cuda_pin] SKIP {label}: not C-contiguous")
        return False
    rt = _get_cudart()
    if rt is None:
        return False
    err = rt.cudaHostRegister(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_size_t(arr.nbytes),
        ctypes.c_uint(1),  # cudaHostRegisterPortable
    )
    gb = arr.nbytes / (1024**3)
    if err == 0:
        print(f"  [cuda_pin] OK {label}: {gb:.2f} GB pinned")
    else:
        print(f"  [cuda_pin] FAIL {label}: {gb:.2f} GB, cuda error {err}")
    return err == 0

def _to_zfirst(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(arr, (3, 0, 1, 2)))

_cudart_dll: "ctypes.CDLL | None" = None
_cudart_tried = False

def _get_cudart():
    global _cudart_dll, _cudart_tried
    if not _cudart_tried:
        _cudart_tried = True
        try:
            rt = ctypes.CDLL('libcudart.so')
            rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            rt.cudaHostRegister.restype = ctypes.c_int
            rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            rt.cudaMemcpyAsync.restype = ctypes.c_int
            rt.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
            rt.cudaStreamSynchronize.restype = ctypes.c_int
            rt.cudaDeviceSynchronize.argtypes = []
            rt.cudaDeviceSynchronize.restype = ctypes.c_int
            rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            rt.cudaMemGetInfo.restype = ctypes.c_int
            _cudart_dll = rt
        except OSError:
            pass
    return _cudart_dll

def _gpu_mem_info() -> str:
    rt = _get_cudart()
    if rt is None: return ""
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    err = rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if err != 0: return ""
    return f"{free.value / (1024**3):.1f}/{total.value / (1024**3):.1f} GB free"


# ---------------------------------------------------------------------------
# Transfer Helpers (Zero-Allocation Direct DMA to persistent buffers)
# ---------------------------------------------------------------------------
# Use default stream (NULL) to avoid cross-stream conflicts with JAX/XLA on GH200
# unified memory. See docs/TILED_FDTD_DISPATCH_ANALYSIS.md
_USE_DEFAULT_STREAM = True  # Set False to use c_void_p(2) (legacy, may cause intermittency)

_STREAM_PER_THREAD = None

def _get_copy_stream():
    """Stream for H2D/D2H. Default stream avoids cross-stream conflicts with JAX."""
    global _STREAM_PER_THREAD
    if _USE_DEFAULT_STREAM:
        return ctypes.c_void_p(0)  # CUDA default stream (NULL)
    if _STREAM_PER_THREAD is None:
        _STREAM_PER_THREAD = ctypes.c_void_p(2)  # Legacy: invalid, may conflict
    return _STREAM_PER_THREAD

def _copy_h2d(src_np: np.ndarray, dst_gpu: jax.Array) -> jax.Array:
    """DMA directly from pinned CPU memory into an existing GPU JAX Array."""
    rt = _get_cudart()
    stream = _get_copy_stream()
    if rt is not None:
        try:
            try: gpu_ptr = dst_gpu.unsafe_buffer_pointer()
            except: gpu_ptr = dst_gpu.addressable_shards[0].data.unsafe_buffer_pointer()

            err = rt.cudaMemcpyAsync(
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_void_p(src_np.ctypes.data),
                ctypes.c_size_t(src_np.nbytes),
                ctypes.c_int(1),  # cudaMemcpyHostToDevice
                stream,
            )
            if err == 0:
                return dst_gpu
        except Exception: pass

    return jax.device_put(src_np, dst_gpu.device() if hasattr(dst_gpu, 'device') else jax.devices("gpu")[0])


def _copy_d2h(src_gpu: jax.Array, dst_np: np.ndarray, diag_label: str, t_start: float) -> None:
    """DMA raw memory directly from the JIT kernel output to the pinned numpy array."""
    import time as _time
    src_gpu.block_until_ready()
    _tb = _time.perf_counter()

    rt = _get_cudart()
    stream = _get_copy_stream()
    if rt is not None:
        try:
            try: gpu_ptr = src_gpu.unsafe_buffer_pointer()
            except: gpu_ptr = src_gpu.addressable_shards[0].data.unsafe_buffer_pointer()

            err = rt.cudaMemcpyAsync(
                ctypes.c_void_p(dst_np.ctypes.data),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(dst_np.nbytes),
                ctypes.c_int(2),  # cudaMemcpyDeviceToHost
                stream,
            )
            if err == 0:
                if diag_label:
                    rt.cudaStreamSynchronize(stream)
                    _tc = _time.perf_counter()
                    print(f"    D2H {diag_label}: kernel_wait={_tb-t_start:.4f}s  memcpy={_tc-_tb:.4f}s  total={_tc-t_start:.4f}s")
                return
        except Exception: pass

    dst_np[:] = np.asarray(jax.device_get(src_gpu))
    if diag_label:
        _tc = _time.perf_counter()
        print(f"    D2H {diag_label} (fallback): kernel_wait={_tb-t_start:.4f}s  memcpy={_tc-_tb:.4f}s  total={_tc-t_start:.4f}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_xy(field: jax.Array, periodic_axes: tuple[bool, bool, bool]) -> jax.Array:
    # Field layout is (Cz, C, Nx, Ny). Pad axes 2 (x) and 3 (y).
    for i in range(2):
        mode = "wrap" if periodic_axes[i] else "constant"
        pw = [(0, 0)] * 4
        pw[i + 2] = (1, 1)
        field = jnp.pad(field, pw, mode=mode)
    return field

_edge_halo_E: np.ndarray | None = None
_edge_halo_H: np.ndarray | None = None

def _get_edge_halo_buf(field_np: np.ndarray, Cz: int, tag: str) -> np.ndarray:
    global _edge_halo_E, _edge_halo_H
    buf = _edge_halo_E if tag == "E" else _edge_halo_H
    need = (Cz + 1,) + field_np.shape[1:]
    if buf is None or buf.shape != need or buf.dtype != field_np.dtype:
        buf = np.empty(need, dtype=field_np.dtype)
        _cuda_pin(buf, f"edge_halo_{tag}")
        if tag == "E": _edge_halo_E = buf
        else: _edge_halo_H = buf
    return buf

def _get_H_halo_np(H_np: np.ndarray, z0: int, z1: int, Nz: int, Cz: int, periodic_z: bool) -> np.ndarray:
    if z0 > 0:
        return H_np[z0 - 1 : z1]
    buf = _get_edge_halo_buf(H_np, Cz, "E")
    buf[0] = H_np[-1] if periodic_z else 0
    buf[1:] = H_np[0:z1]
    return buf

def _get_E_halo_np(E_np: np.ndarray, z0: int, z1: int, Nz: int, Cz: int, periodic_z: bool) -> np.ndarray:
    if z1 < Nz:
        return E_np[z0 : z1 + 1]
    buf = _get_edge_halo_buf(E_np, Cz, "H")
    buf[:-1] = E_np[z0:z1]
    buf[-1] = E_np[0] if periodic_z else 0
    return buf


# ---------------------------------------------------------------------------
# PML chunk helpers (Native Z-First)
# ---------------------------------------------------------------------------

def _extract_pml_slab_zf(arr, axis_zf, L, side):
    if side == "min":
        return lax.slice_in_dim(arr, 0, L, axis=axis_zf)
    else:
        dim_size = arr.shape[axis_zf]
        return lax.slice_in_dim(arr, dim_size - L, dim_size, axis=axis_zf)

def _pml_update_xy_zf(psi_min, psi_max, d_field, b_coeff, a_coeff, axis_zf, z0, Cz):
    L_min, L_max = psi_min.shape[axis_zf], psi_max.shape[axis_zf]

    def _z_slice(arr):
        starts, sizes = [jnp.int32(0)] * 3, list(arr.shape)
        starts[0], sizes[0] = z0, Cz
        return lax.dynamic_slice(arr, starts, sizes)

    def _z_write(arr, update):
        starts = [jnp.int32(0)] * 3
        starts[0] = z0
        return lax.dynamic_update_slice(arr, update, starts)

    psi_min_c, psi_max_c = _z_slice(psi_min), _z_slice(psi_max)

    if L_min > 0:
        d_min = _extract_pml_slab_zf(d_field, axis_zf, L_min, "min")
        b_min = _extract_pml_slab_zf(b_coeff, axis_zf, L_min, "min")
        a_min = _extract_pml_slab_zf(a_coeff, axis_zf, L_min, "min")
        psi_min_c = b_min * psi_min_c + a_min * d_min
        psi_min = _z_write(psi_min, psi_min_c)

    if L_max > 0:
        d_max = _extract_pml_slab_zf(d_field, axis_zf, L_max, "max")
        b_max = _extract_pml_slab_zf(b_coeff, axis_zf, L_max, "max")
        a_max = _extract_pml_slab_zf(a_coeff, axis_zf, L_max, "max")
        psi_max_c = b_max * psi_max_c + a_max * d_max
        psi_max = _z_write(psi_max, psi_max_c)

    return psi_min, psi_max, psi_min_c, psi_max_c

def _pml_update_z_zf(psi_min, psi_max, d_field, b_coeff, a_coeff, z0, Cz, Nz):
    L_min, L_max = psi_min.shape[0], psi_max.shape[0]

    if L_min > 0:
        b_min = _extract_pml_slab_zf(b_coeff, 0, L_min, "min")
        a_min = _extract_pml_slab_zf(a_coeff, 0, L_min, "min")
        psi_idx = jnp.arange(L_min)
        in_chunk = (psi_idx >= z0) & (psi_idx < z0 + Cz)
        chunk_local = jnp.clip(psi_idx - z0, 0, Cz - 1)
        d_gathered = d_field[chunk_local, :, :]
        psi_min_new = b_min * psi_min + a_min * d_gathered
        psi_min = jnp.where(in_chunk[:, None, None], psi_min_new, psi_min)

    if L_max > 0:
        b_max = _extract_pml_slab_zf(b_coeff, 0, L_max, "max")
        a_max = _extract_pml_slab_zf(a_coeff, 0, L_max, "max")
        psi_global = Nz - L_max + jnp.arange(L_max)
        in_chunk = (psi_global >= z0) & (psi_global < z0 + Cz)
        chunk_local = jnp.clip(psi_global - z0, 0, Cz - 1)
        d_gathered = d_field[chunk_local, :, :]
        psi_max_new = b_max * psi_max + a_max * d_gathered
        psi_max = jnp.where(in_chunk[:, None, None], psi_max_new, psi_max)

    return psi_min, psi_max

def _scatter_psi_zf(curl_comp, psi_min_c, psi_max_c, axis_zf, sign):
    L_min, L_max = psi_min_c.shape[axis_zf], psi_max_c.shape[axis_zf]
    
    if L_min > 0:
        starts = [0, 0, 0]
        curl_comp = lax.dynamic_update_slice(
            curl_comp, lax.dynamic_slice(curl_comp, starts, psi_min_c.shape) + sign * psi_min_c, starts
        )
        
    if L_max > 0:
        starts = [0, 0, 0]
        starts[axis_zf] = curl_comp.shape[axis_zf] - L_max
        curl_comp = lax.dynamic_update_slice(
            curl_comp, lax.dynamic_slice(curl_comp, starts, psi_max_c.shape) + sign * psi_max_c, starts
        )
    return curl_comp

def _scatter_z_psi_zf(curl_comp, psi_min, psi_max, sign, z0, Cz, Nz):
    L_min, L_max = psi_min.shape[0], psi_max.shape[0]
    local_z, global_z = jnp.arange(Cz), jnp.arange(Cz) + z0

    if L_min > 0:
        in_pml_min = global_z < L_min
        safe_idx = jnp.clip(global_z, 0, L_min - 1)
        curl_comp = curl_comp + jnp.where(in_pml_min[:, None, None], sign * psi_min[safe_idx, :, :], 0.0)

    if L_max > 0:
        pml_max_start = Nz - L_max
        in_pml_max = global_z >= pml_max_start
        safe_idx = jnp.clip(global_z - pml_max_start, 0, L_max - 1)
        curl_comp = curl_comp + jnp.where(in_pml_max[:, None, None], sign * psi_max[safe_idx, :, :], 0.0)

    return curl_comp

_CURL_IDX = (0, 0, 1, 1, 2, 2)
_CURL_SIGN = (+1.0, -1.0, +1.0, -1.0, +1.0, -1.0)


# ---------------------------------------------------------------------------
# Native Z-First JIT Updates (Zero Internal XLA Allocations)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("Cz", "Nz", "periodic_axes", "has_conductivity"), donate_argnums=(0, 4))
def _update_E_chunk(
    E_chunk, H_halo, inv_eps_chunk, sigma_E_chunk, psi_E, b_pml, a_pml,
    kappa_x, kappa_y, kappa_z_chunk, z0, courant_number, Nz, Cz, periodic_axes, has_conductivity
):
    H_pad = _pad_xy(H_halo, periodic_axes) # shape: (Cz+1, 3, Nx+2, Ny+2)
    Hz, Hx, Hy = H_pad[:, 2, :, :], H_pad[:, 0, :, :], H_pad[:, 1, :, :]
    
    dyHz = (Hz - jnp.roll(Hz, 1, axis=2))[1:, 1:-1, 1:-1]
    dyHx = (Hx - jnp.roll(Hx, 1, axis=2))[1:, 1:-1, 1:-1]
    dxHz = (Hz - jnp.roll(Hz, 1, axis=1))[1:, 1:-1, 1:-1]
    dxHy = (Hy - jnp.roll(Hy, 1, axis=1))[1:, 1:-1, 1:-1]
    dzHy = Hy[1:, 1:-1, 1:-1] - Hy[:-1, 1:-1, 1:-1]
    dzHx = Hx[1:, 1:-1, 1:-1] - Hx[:-1, 1:-1, 1:-1]
    
    d_fields = (dyHz, dzHy, dzHx, dxHz, dxHy, dyHx)

    curl_x = (1.0 / kappa_y) * dyHz - (1.0 / kappa_z_chunk) * dzHy
    curl_y = (1.0 / kappa_z_chunk) * dzHx - (1.0 / kappa_x) * dxHz
    curl_z = (1.0 / kappa_x) * dxHy - (1.0 / kappa_y) * dyHx
    curls = [curl_x, curl_y, curl_z]

    psi_list = list(psi_E)
    spatial_axis_map = {0: 1, 1: 2, 2: 0} # map physical (x,y,z) to z-first dims

    for i in range(6):
        orig_axis = PSI_COMPONENT_AXIS[i]
        axis_zf = spatial_axis_map[orig_axis]
        ci, cidx, sign = PSI_E_COEFF_IDX[i], _CURL_IDX[i], _CURL_SIGN[i]
        psi_min_i, psi_max_i = psi_list[i]
        
        if axis_zf != 0:
            psi_min_i, psi_max_i, p_min_c, p_max_c = _pml_update_xy_zf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci], axis_zf, z0, Cz)
            curls[cidx] = _scatter_psi_zf(curls[cidx], p_min_c, p_max_c, axis_zf, sign)
        else:
            psi_min_i, psi_max_i = _pml_update_z_zf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci], z0, Cz, Nz)
            curls[cidx] = _scatter_z_psi_zf(curls[cidx], psi_min_i, psi_max_i, sign, z0, Cz, Nz)
        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=1) # shape: (Cz, 3, Nx, Ny)
    c = courant_number
    if has_conductivity:
        loss = c * sigma_E_chunk * eta0 * inv_eps_chunk / 2
        E_new = (1 - loss) * E_chunk + c * curl * inv_eps_chunk
        E_new = E_new / (1 + loss)
    else:
        E_new = E_chunk + c * curl * inv_eps_chunk

    return E_new, tuple(psi_list)


@partial(jax.jit, static_argnames=("Cz", "Nz", "periodic_axes", "has_conductivity", "mu_is_scalar"), donate_argnums=(0, 4))
def _update_H_chunk(
    H_chunk, E_halo, inv_mu_chunk, sigma_H_chunk, psi_H, b_pml, a_pml,
    kappa_x, kappa_y, kappa_z_chunk, z0, courant_number, Nz, Cz, periodic_axes, has_conductivity, mu_is_scalar
):
    E_pad = _pad_xy(E_halo, periodic_axes) # shape: (Cz+1, 3, Nx+2, Ny+2)
    Ez, Ex, Ey = E_pad[:, 2, :, :], E_pad[:, 0, :, :], E_pad[:, 1, :, :]
    
    dyEz = (jnp.roll(Ez, -1, axis=2) - Ez)[:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(Ex, -1, axis=2) - Ex)[:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(Ez, -1, axis=1) - Ez)[:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(Ey, -1, axis=1) - Ey)[:-1, 1:-1, 1:-1]
    dzEy = Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]
    dzEx = Ex[1:, 1:-1, 1:-1] - Ex[:-1, 1:-1, 1:-1]
    
    d_fields = (dyEz, dzEy, dzEx, dxEz, dxEy, dyEx)

    curl_x = (1.0 / kappa_y) * dyEz - (1.0 / kappa_z_chunk) * dzEy
    curl_y = (1.0 / kappa_z_chunk) * dzEx - (1.0 / kappa_x) * dxEz
    curl_z = (1.0 / kappa_x) * dxEy - (1.0 / kappa_y) * dyEx
    curls = [curl_x, curl_y, curl_z]

    psi_list = list(psi_H)
    spatial_axis_map = {0: 1, 1: 2, 2: 0} # map physical (x,y,z) to z-first dims

    for i in range(6):
        orig_axis = PSI_COMPONENT_AXIS[i]
        axis_zf = spatial_axis_map[orig_axis]
        ci, cidx, sign = PSI_H_COEFF_IDX[i], _CURL_IDX[i], _CURL_SIGN[i]
        psi_min_i, psi_max_i = psi_list[i]
        
        if axis_zf != 0:
            psi_min_i, psi_max_i, p_min_c, p_max_c = _pml_update_xy_zf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci], axis_zf, z0, Cz)
            curls[cidx] = _scatter_psi_zf(curls[cidx], p_min_c, p_max_c, axis_zf, sign)
        else:
            psi_min_i, psi_max_i = _pml_update_z_zf(
                psi_min_i, psi_max_i, d_fields[i], b_pml[ci], a_pml[ci], z0, Cz, Nz)
            curls[cidx] = _scatter_z_psi_zf(curls[cidx], psi_min_i, psi_max_i, sign, z0, Cz, Nz)
        psi_list[i] = (psi_min_i, psi_max_i)

    curl = jnp.stack(curls, axis=1) # shape: (Cz, 3, Nx, Ny)
    c, inv_mu = courant_number, inv_mu_chunk if not mu_is_scalar else 1.0
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

def _auto_chunk_size(Nz, Nx, Ny, C_eps, has_conductivity, mu_is_scalar, dtype_bytes, gpu_budget_bytes) -> int:
    cell_bytes = Nx * Ny * dtype_bytes
    arrays_per_z = 3 + 3 + C_eps + 3 
    if has_conductivity: arrays_per_z += C_eps * 2
    if not mu_is_scalar: arrays_per_z += C_eps

    xla_multiplier = 8
    bytes_per_z = arrays_per_z * cell_bytes * xla_multiplier
    fixed_overhead = 3 * cell_bytes * xla_multiplier

    max_cz = max(1, int((gpu_budget_bytes - fixed_overhead) // bytes_per_z))
    best = 1
    for d in range(1, min(max_cz, Nz) + 1):
        if Nz % d == 0: best = d
    return best

def tiled_fdtd(
    arrays: ArrayContainer, 
    objects: ObjectContainer, 
    config: SimulationConfig, 
    key: jax.Array, 
    chunk_size: int | None = None,
    warmup: bool = False
) -> SimulationState:
    print_timestamp()
    print("Starting tiled_fdtd")

    del key 
    inv_eps = arrays.inv_permittivities
    if inv_eps.shape[0] == 9:
        raise NotImplementedError("Tiled FDTD does not yet support fully anisotropic materials.")

    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float64
    field_shape = arrays.E.shape 
    _, Nx, Ny, Nz = field_shape

    gpu = jax.devices("gpu")[0]

    _saved_recording_state = arrays.recording_state
    has_detectors = bool(objects.forward_detectors)
    has_sigma_E = arrays.electric_conductivity is not None
    has_sigma_H = arrays.magnetic_conductivity is not None

    E_np = np.zeros((Nz, 3, Nx, Ny), dtype=np_dtype)
    H_np = np.zeros((Nz, 3, Nx, Ny), dtype=np_dtype)
    _free_jax_buffer(arrays.E)
    _free_jax_buffer(arrays.H)
    gc.collect()

    inv_eps_np = _to_zfirst(np.array(jax.device_get(arrays.inv_permittivities)))
    _free_jax_buffer(arrays.inv_permittivities)
    
    sigma_E_np = _to_zfirst(np.array(jax.device_get(arrays.electric_conductivity))) if has_sigma_E else None
    if has_sigma_E: _free_jax_buffer(arrays.electric_conductivity)

    sigma_H_np = _to_zfirst(np.array(jax.device_get(arrays.magnetic_conductivity))) if has_sigma_H else None
    if has_sigma_H: _free_jax_buffer(arrays.magnetic_conductivity)

    inv_mu_val = arrays.inv_permeabilities
    mu_is_scalar = not isinstance(inv_mu_val, jax.Array) or inv_mu_val.ndim == 0
    inv_mu_scalar = float(jax.device_get(jnp.asarray(inv_mu_val))) if mu_is_scalar else None
    inv_mu_np = None if mu_is_scalar else _to_zfirst(np.array(jax.device_get(inv_mu_val)))
    if not mu_is_scalar: _free_jax_buffer(inv_mu_val)
    gc.collect()

    # Detector states — zero (reset), keep on GPU to avoid per-step transfers.
    detector_states_gpu: dict = {}
    for k, v in arrays.detector_states.items():
        detector_states_gpu[k] = {}
        for k2, v2 in v.items():
            detector_states_gpu[k][k2] = jax.device_put(v2 * 0, gpu)
            _free_jax_buffer(v2)

    # ------------------------------------------------------------------
    # Transpose PML grids into z-first natively before uploading
    # ------------------------------------------------------------------
    def _tp_psi(p):
        return np.transpose(np.array(jax.device_get(p)), (2, 0, 1)) * 0

    cpu_psi_E = tuple((_tp_psi(p_min), _tp_psi(p_max)) for p_min, p_max in arrays.psi_E)
    cpu_psi_H = tuple((_tp_psi(p_min), _tp_psi(p_max)) for p_min, p_max in arrays.psi_H)
    
    psi_E = jax.tree_util.tree_map(lambda x: jax.device_put(x, gpu), cpu_psi_E)
    psi_H = jax.tree_util.tree_map(lambda x: jax.device_put(x, gpu), cpu_psi_H)
    
    # 1D arrays, no transposes needed
    alpha = tuple(jax.device_put(a, gpu) for a in arrays.alpha)
    kappa = tuple(jax.device_put(k, gpu) for k in arrays.kappa)
    sigma_pml = tuple(jax.device_put(s, gpu) for s in arrays.sigma)

    b_pml, a_pml = _compute_pml_ab(alpha, kappa, sigma_pml, config)
    
    # Pre-transpose PML coefficients and kappa arrays so they perfectly broadcast with z-first inside the kernel
    b_pml = tuple(jnp.transpose(b, (2, 0, 1)) for b in b_pml)
    a_pml = tuple(jnp.transpose(a, (2, 0, 1)) for a in a_pml)
    
    kappa_x = jnp.transpose(kappa[0], (2, 0, 1))
    kappa_y = jnp.transpose(kappa[1], (2, 0, 1))
    kappa_z_full = jnp.transpose(kappa[2], (2, 0, 1))

    print("  [tiled_fdtd] Pinning host arrays...")
    _cuda_pin(E_np, "E_np")
    _cuda_pin(H_np, "H_np")
    _cuda_pin(inv_eps_np, "inv_eps_np")
    if sigma_E_np is not None: _cuda_pin(sigma_E_np, "sigma_E_np")
    if sigma_H_np is not None: _cuda_pin(sigma_H_np, "sigma_H_np")
    if inv_mu_np is not None: _cuda_pin(inv_mu_np, "inv_mu_np")

    C_eps = inv_eps_np.shape[1]
    dtype_bytes = 4 if np_dtype == np.float32 else 8
    gpu_mem = gpu.memory_stats()
    available = int((gpu_mem["bytes_limit"] - gpu_mem.get("bytes_in_use", 0)) * 0.75) if gpu_mem and "bytes_limit" in gpu_mem else 80 * 1024**3
    
    Cz = chunk_size if chunk_size is not None else _auto_chunk_size(Nz, Nx, Ny, C_eps, has_sigma_E, mu_is_scalar, dtype_bytes, available)
    if Nz % Cz != 0: raise ValueError(f"Nz ({Nz}) must be divisible by chunk_size ({Cz}).")
    n_chunks = Nz // Cz

    # Pre-allocate and pin edge halos BEFORE the loop so we never allocate during chunks
    _get_edge_halo_buf(E_np, Cz, "E")
    _get_edge_halo_buf(H_np, Cz, "H")

    periodic_axes = get_periodic_axes(objects)
    periodic_z = periodic_axes[2]
    c_num = jnp.asarray(config.courant_number, dtype=config.dtype)
    _dummy_sigma = jnp.zeros((1,), dtype=config.dtype)

    # ------------------------------------------------------------------
    # ZERO-ALLOCATION SETUP: Pre-allocate exactly one set of GPU buffers
    # ------------------------------------------------------------------
    print("  [tiled_fdtd] Allocating persistent GPU chunk buffers...")
    
    # Use jnp.zeros to natively allocate and zero-fill directly on the GPU
    buf_E = jnp.zeros((Cz, 3, Nx, Ny), dtype=np_dtype)
    buf_H_halo = jnp.zeros((Cz+1, 3, Nx, Ny), dtype=np_dtype)
    buf_eps = jnp.zeros((Cz, C_eps, Nx, Ny), dtype=np_dtype)
    buf_sig_E = jnp.zeros((Cz, C_eps, Nx, Ny), dtype=np_dtype) if has_sigma_E else _dummy_sigma

    buf_H = jnp.zeros((Cz, 3, Nx, Ny), dtype=np_dtype)
    buf_E_halo = jnp.zeros((Cz+1, 3, Nx, Ny), dtype=np_dtype)
    buf_mu = jnp.zeros((Cz, C_eps, Nx, Ny), dtype=np_dtype) if not mu_is_scalar else None
    buf_sig_H = jnp.zeros((Cz, C_eps, Nx, Ny), dtype=np_dtype) if has_sigma_H else _dummy_sigma

    # Wait for the GPU kernels to finish initializing the memory
    for b in (buf_E, buf_H_halo, buf_eps, buf_sig_E, buf_H, buf_E_halo, buf_mu, buf_sig_H):
        if isinstance(b, jax.Array):
            b.block_until_ready()

    print_timestamp()
    print(f"Time loop starting")

    _rt = _get_cudart()
    _copy_stream = _get_copy_stream()

    for t in range(config.time_steps_total):
        if t % 1 == 0:
            print_timestamp()
            print(f"Time step {t} of {config.time_steps_total}")
        time_step = jnp.asarray(t, dtype=jnp.int32)

        # ==============================================================
        # Phase 1 — E update 
        # ==============================================================
        print_timestamp()
        if t < 5: print(f"E update starting  gpu_mem={_gpu_mem_info()}")
        else: print(f"E update starting")
        import time as _time
        _chunk_times_E = []
        for iz in range(n_chunks):
            z0_int, z1_int = iz * Cz, iz * Cz + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            _t0 = _time.perf_counter()

            # Wait for D2H from previous chunk to finish before overwriting input buffers
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_sync_prev = _time.perf_counter()

            # Direct DMA from CPU memory into the existing physical GPU buffers
            H_halo_np = _get_H_halo_np(H_np, z0_int, z1_int, Nz, Cz, periodic_z)
            buf_H_halo = _copy_h2d(H_halo_np, buf_H_halo)
            buf_E = _copy_h2d(E_np[z0_int:z1_int], buf_E)
            buf_eps = _copy_h2d(inv_eps_np[z0_int:z1_int], buf_eps)
            if has_sigma_E: buf_sig_E = _copy_h2d(sigma_E_np[z0_int:z1_int], buf_sig_E)
            kz_chunk = kappa_z_full[z0_int:z1_int, :, :]

            # Sync copy stream so H2D memory loads finish BEFORE kernel starts reading
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_h2d_done = _time.perf_counter()

            # Run Kernel. XLA reuses the donated buf_E. Other buffers are safely preserved.
            buf_E_out, psi_E = _update_E_chunk(
                buf_E, buf_H_halo, buf_eps, buf_sig_E, psi_E, b_pml, a_pml,
                kappa_x, kappa_y, kz_chunk, z0_jax, c_num, Nz=Nz, Cz=Cz,
                periodic_axes=periodic_axes, has_conductivity=has_sigma_E,
            )
            buf_E_out.block_until_ready()
            _t_kernel_done = _time.perf_counter()

            _diag = f"E{iz}" if t < 5 and (iz < 5 or iz % 5 == 0) else ""
            _copy_d2h(buf_E_out, E_np[z0_int:z1_int], "", _t0)  # no internal print, we do per-phase below
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_d2h_done = _time.perf_counter()

            # Per-phase timing to pinpoint rolling slowdowns
            if _diag:
                sync_prev = _t_sync_prev - _t0
                h2d = _t_h2d_done - _t_sync_prev
                kernel = _t_kernel_done - _t_h2d_done
                d2h = _t_d2h_done - _t_kernel_done
                total = _t_d2h_done - _t0
                slow = " <<< SLOW" if total > 0.1 else ""
                print(f"    {_diag}: sync_prev={sync_prev:.3f}s h2d={h2d:.3f}s kernel={kernel:.3f}s d2h={d2h:.3f}s total={total:.3f}s{slow}")
            
            # Keep the resulting buffer alive for the next iteration's H2D overwrite!
            buf_E = buf_E_out

        # --- E-field sources ---
        print_timestamp()
        print(f"E-field sources starting")
        _apply_sources_E_gpu(E_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, objects, time_step, config.dtype, gpu)

        gc.collect()
        if _rt: _rt.cudaDeviceSynchronize()

        # ==============================================================
        # Phase 2 — H update
        # ==============================================================
        print_timestamp()
        if t < 5: print(f"H update starting  gpu_mem={_gpu_mem_info()}")
        else: print(f"H update starting")
        _chunk_times_H = []
        for iz in range(n_chunks):
            z0_int, z1_int = iz * Cz, iz * Cz + Cz
            z0_jax = jnp.asarray(z0_int, dtype=jnp.int32)

            _t0 = _time.perf_counter()

            # Wait for D2H from previous chunk to finish before overwriting input buffers
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_sync_prev = _time.perf_counter()

            E_halo_np = _get_E_halo_np(E_np, z0_int, z1_int, Nz, Cz, periodic_z)
            buf_E_halo = _copy_h2d(E_halo_np, buf_E_halo)
            buf_H = _copy_h2d(H_np[z0_int:z1_int], buf_H)
            if not mu_is_scalar: buf_mu = _copy_h2d(inv_mu_np[z0_int:z1_int], buf_mu)
            if has_sigma_H: buf_sig_H = _copy_h2d(sigma_H_np[z0_int:z1_int], buf_sig_H)
            kz_chunk = kappa_z_full[z0_int:z1_int, :, :]

            # Sync copy stream so H2D memory loads finish BEFORE kernel starts reading
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_h2d_done = _time.perf_counter()

            buf_H_out, psi_H = _update_H_chunk(
                buf_H, buf_E_halo, 
                buf_mu if not mu_is_scalar else _dummy_sigma, 
                buf_sig_H, psi_H, b_pml, a_pml,
                kappa_x, kappa_y, kz_chunk, z0_jax, c_num, Nz=Nz, Cz=Cz,
                periodic_axes=periodic_axes, has_conductivity=has_sigma_H, mu_is_scalar=mu_is_scalar,
            )
            buf_H_out.block_until_ready()
            _t_kernel_done = _time.perf_counter()

            _diag = f"H{iz}" if t < 5 and (iz < 5 or iz % 5 == 0) else ""
            _copy_d2h(buf_H_out, H_np[z0_int:z1_int], "", _t0)
            if _rt and _copy_stream: _rt.cudaStreamSynchronize(_copy_stream)
            _t_d2h_done = _time.perf_counter()

            # Per-phase timing to pinpoint rolling slowdowns
            if _diag:
                sync_prev = _t_sync_prev - _t0
                h2d = _t_h2d_done - _t_sync_prev
                kernel = _t_kernel_done - _t_h2d_done
                d2h = _t_d2h_done - _t_kernel_done
                total = _t_d2h_done - _t0
                slow = " <<< SLOW" if total > 0.1 else ""
                print(f"    {_diag}: sync_prev={sync_prev:.3f}s h2d={h2d:.3f}s kernel={kernel:.3f}s d2h={d2h:.3f}s total={total:.3f}s{slow}")

            buf_H = buf_H_out

        # --- H-field sources ---
        print_timestamp()
        print(f"H-field sources starting")
        _apply_sources_H_gpu(H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, objects, time_step, config.dtype, gpu)

        # --- Detectors ---
        print_timestamp()
        print(f"Detector update starting")
        if has_detectors:
            detector_states_gpu = _update_detectors_gpu(
                E_np, H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar,
                detector_states_gpu, objects, time_step, gpu,
            )
        
        gc.collect()
        if _rt: _rt.cudaDeviceSynchronize()

    # Reconstruct ArrayContainer
    final_E = jnp.asarray(np.ascontiguousarray(np.transpose(E_np, (1, 2, 3, 0))))
    final_H = jnp.asarray(np.ascontiguousarray(np.transpose(H_np, (1, 2, 3, 0))))
    final_inv_eps = jnp.asarray(np.ascontiguousarray(np.transpose(inv_eps_np, (1, 2, 3, 0))))
    final_sigma_E = jnp.asarray(np.ascontiguousarray(np.transpose(sigma_E_np, (1, 2, 3, 0)))) if has_sigma_E else None
    final_sigma_H = jnp.asarray(np.ascontiguousarray(np.transpose(sigma_H_np, (1, 2, 3, 0)))) if has_sigma_H else None
    final_inv_mu = jnp.asarray(inv_mu_scalar) if mu_is_scalar else jnp.asarray(np.ascontiguousarray(np.transpose(inv_mu_np, (1, 2, 3, 0))))

    def _inv_tp_psi(p):
        return jnp.transpose(p, (1, 2, 0))

    final_psi_E = tuple((_inv_tp_psi(p_min), _inv_tp_psi(p_max)) for p_min, p_max in psi_E)
    final_psi_H = tuple((_inv_tp_psi(p_min), _inv_tp_psi(p_max)) for p_min, p_max in psi_H)

    out = ArrayContainer(
        E=final_E, H=final_H, psi_E=final_psi_E, psi_H=final_psi_H, alpha=alpha, kappa=kappa, sigma=sigma_pml,
        inv_permittivities=final_inv_eps, inv_permeabilities=final_inv_mu, detector_states=detector_states_gpu,
        recording_state=_saved_recording_state, electric_conductivity=final_sigma_E, magnetic_conductivity=final_sigma_H,
    )
    return (jnp.asarray(config.time_steps_total, dtype=jnp.int32), out)


# ---------------------------------------------------------------------------
# GPU source and detector helpers 
# ---------------------------------------------------------------------------

def _remap_to_gpu(obj, field_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, gpu):
    gst, (sx, sy, sz) = obj._grid_slice_tuple, obj.grid_slice
    dx, dy, dz = gst[0][1] - gst[0][0], gst[1][1] - gst[1][0], gst[2][1] - gst[2][0]

    field_gpu = jax.device_put(jnp.asarray(np.ascontiguousarray(np.transpose(field_np[sz, :, sx, sy], (1, 2, 3, 0)))), gpu)
    eps_gpu = jax.device_put(jnp.asarray(np.ascontiguousarray(np.transpose(inv_eps_np[sz, :, sx, sy], (1, 2, 3, 0)))), gpu)
    
    if mu_is_scalar:
        mu_gpu = inv_mu_scalar 
    else:
        mu_gpu = jax.device_put(jnp.asarray(np.ascontiguousarray(np.transpose(inv_mu_np[sz, :, sx, sy], (1, 2, 3, 0)))), gpu)

    obj_remap = obj.aset("_grid_slice_tuple", ((0, dx), (0, dy), (0, dz)))
    return obj_remap, field_gpu, eps_gpu, mu_gpu

def _apply_sources_E_gpu(E_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, objects, time_step, dtype, gpu):
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))): continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice
        src_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu(source, E_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, gpu)
        E_updated = src_remap.update_E(E=E_gpu, inv_permittivities=eps_gpu, inv_permeabilities=mu_gpu, time_step=adj, inverse=False)
        E_np[sz, :, sx, sy] = np.transpose(np.asarray(jax.device_get(E_updated)), (3, 0, 1, 2))

def _apply_sources_H_gpu(H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, objects, time_step, dtype, gpu):
    for source in objects.sources:
        if not bool(jax.device_get(source.is_on_at_time_step(time_step))): continue
        adj = source.adjust_time_step_by_on_off(time_step)
        sx, sy, sz = source.grid_slice
        src_remap, H_gpu, eps_gpu, mu_gpu = _remap_to_gpu(source, H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, gpu)
        H_updated = src_remap.update_H(H=H_gpu, inv_permittivities=eps_gpu, inv_permeabilities=mu_gpu, time_step=adj + 0.5, inverse=False)
        H_np[sz, :, sx, sy] = np.transpose(np.asarray(jax.device_get(H_updated)), (3, 0, 1, 2))

def _update_detectors_gpu(E_np, H_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, detector_states, objects, time_step, gpu):
    for d in objects.forward_detectors:
        if not bool(jax.device_get(d._is_on_at_time_step_arr[time_step])): continue
        d_remap, E_gpu, eps_gpu, mu_gpu = _remap_to_gpu(d, E_np, inv_eps_np, inv_mu_np, mu_is_scalar, inv_mu_scalar, gpu)
        sx, sy, sz = d.grid_slice
        H_gpu = jax.device_put(jnp.asarray(np.ascontiguousarray(np.transpose(H_np[sz, :, sx, sy], (1, 2, 3, 0)))), gpu)
        detector_states[d.name] = d_remap.update(
            time_step=time_step, E=E_gpu, H=H_gpu, state=detector_states[d.name], inv_permittivity=eps_gpu, inv_permeability=mu_gpu
        )
    return detector_states