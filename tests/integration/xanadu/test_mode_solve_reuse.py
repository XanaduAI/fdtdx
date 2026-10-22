"""XANADU: mode-solve reuse and the polarization-count check in ``compute_mode``.

SilverHammer needs every mode index in both directions on each monitor plane. With the same ``num_modes`` for all
of them, the forward tidy3d solve of a plane is done once and reused, and the results are exactly those of solving
each time. An index past the modes of the requested polarization raises instead of returning the other one.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics import modes

_FREQUENCY = 3e8 / 1.55e-6


def _cross_section(core_permittivity: float = 12.0) -> jnp.ndarray:
    # x-normal plane of a strip waveguide, wide in y and thin in z: TE0, then TM0, ...
    permittivity = np.full((3, 1, 40, 24), 2.1)
    permittivity[:, :, 12:28, 9:15] = core_permittivity
    return jnp.asarray(1 / permittivity, dtype=jnp.float32)


def _solve(inv_permittivities, **kwargs):
    return modes.compute_mode(
        frequency=_FREQUENCY, inv_permittivities=inv_permittivities, inv_permeabilities=1.0, resolution=25e-9, **kwargs
    )


@pytest.fixture
def solver_calls(monkeypatch):
    """Clears the reused solve and records the num_modes of every real tidy3d solve."""
    calls = []
    solve = modes._compute_modes

    def counting_solve(**kwargs):
        calls.append(kwargs["mode_spec"].num_modes)
        return solve(**kwargs)

    modes._last_forward_solve.clear()
    monkeypatch.setattr(modes, "_compute_modes", counting_solve)
    yield calls
    modes._last_forward_solve.clear()


def test_one_solve_per_plane_with_identical_results(solver_calls):
    inv_eps = _cross_section()
    requests = [(direction, k) for direction in ("+", "-") for k in range(4)]
    num_modes = 2 * (3 + 1) + 10

    reused = {(d, k): _solve(inv_eps, direction=d, mode_index=k, num_modes=num_modes) for d, k in requests}
    assert solver_calls == [num_modes]

    for d, k in requests:
        modes._last_forward_solve.clear()
        fresh = _solve(inv_eps, direction=d, mode_index=k, num_modes=num_modes)
        for got, want in zip(reused[(d, k)], fresh, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert len(solver_calls) == 1 + len(requests)


def test_new_plane_or_mode_count_solves_again(solver_calls):
    _solve(_cross_section(), mode_index=0)
    _solve(_cross_section(), mode_index=0)
    _solve(_cross_section(11.0), mode_index=0)
    _solve(_cross_section(11.0), mode_index=2)
    # Without num_modes the upstream default 2 * (mode_index + 1) + 10 is kept.
    assert solver_calls == [12, 12, 16]


def test_index_past_the_polarization_raises(solver_calls):
    inv_eps = _cross_section()
    _, _, neff = _solve(inv_eps, mode_index=0, num_modes=2, filter_pol="te")
    assert float(np.real(np.asarray(neff))) > 1.6  # guided: above the cladding index 1.45
    # The two solved modes are TE0 and TM0, so a second TE mode doesn't exist.
    with pytest.raises(Exception, match="only 1 of the 2 solved modes are TE"):
        _solve(inv_eps, mode_index=1, num_modes=2, filter_pol="te")
    with pytest.raises(ValueError, match="must exceed mode_index"):
        _solve(inv_eps, mode_index=2, num_modes=2)


def test_single_mode_solve(solver_calls):
    _, _, neff = _solve(_cross_section(), mode_index=0, num_modes=1)
    assert solver_calls == [1]
    assert float(np.real(np.asarray(neff))) > 1.6  # guided: above the cladding index 1.45
