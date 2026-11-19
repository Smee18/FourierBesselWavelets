import numpy as np
import pytest
import scipy.special

from fbscatnet.math_utils import (
    _find_neumann_root_muller,
    _first_kind_bessel,
    _first_kind_bessel_deriv,
    _generate_fourier_bessel_wavelet,
    _generate_fourier_low_pass_filter,
    _solid_harmonic_wavelet_fourier,
    _solid_harmonics_low_pass_filter
)

SIZE = 193
FREQ_LIMIT = 20
SIGMA = 1
SH_LOW_PASS_PARAM = 4

# Helpers


def _shifted_meshgrid(size=SIZE):
    """Frequency meshgrid in radians/sample with DC at index size // 2."""
    axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size))
    return np.meshgrid(axis, axis)


def _fb_radius_grid(size=SIZE, freq_limit=FREQ_LIMIT):
    """Radius grid assuming the Fourier-Bessel grid spans [-freq_limit, freq_limit)
    with DC at index size // 2 (the same assumption the L2 test makes)."""
    dx = 2 * freq_limit / size
    axis = (np.arange(size) - size // 2) * dx
    X, Y = np.meshgrid(axis, axis)
    return np.hypot(X, Y), dx


def _fb_wavelet(m, k, norm="l2"):
    _, _, Z = _generate_fourier_bessel_wavelet(
        m, k, size=SIZE, sigma=SIGMA, norm=norm, freq_limit=FREQ_LIMIT
    )
    return Z


def _sh_wavelet(m, k):
    X, Y = _shifted_meshgrid()
    return _solid_harmonic_wavelet_fourier(m, k, X, Y)


def _wavelet(impl, m, k):
    return _fb_wavelet(m, k) if impl == "fourier_bessel" else _sh_wavelet(m, k)


IMPLEMENTATIONS = ["fourier_bessel", "solid_harmonic"]


# Bessel wrappers


@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_first_kind_bessel_matches_scipy_for_positive_orders(order):
    x = np.linspace(0.1, 20, 50)
    expected = scipy.special.jv(order, x)
    # atol guards against rtol blowing up near zero crossings.
    np.testing.assert_allclose(_first_kind_bessel(x, order), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_first_kind_bessel_negative_order_identity(order):
    """J_{-n}(x) = (-1)^n J_n(x) for integer n."""
    x = np.linspace(0.1, 20, 50)
    positive = _first_kind_bessel(x, order)
    negative = _first_kind_bessel(x, -order)
    np.testing.assert_allclose(
        negative, ((-1) ** order) * np.asarray(positive), rtol=1e-10, atol=1e-12
    )


def test_first_kind_bessel_scalar_returns_python_float():
    result = _first_kind_bessel(1.5, 0)
    assert isinstance(result, float)


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_first_kind_bessel_deriv_matches_scipy_jvp(order):
    x = np.linspace(0.5, 15, 30)
    expected = scipy.special.jvp(order, x, n=1)
    np.testing.assert_allclose(
        _first_kind_bessel_deriv(x, order), expected, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_first_kind_bessel_deriv_satisfies_recurrence(order):
    """J_n'(x) = (J_{n-1}(x) - J_{n+1}(x)) / 2, independent of scipy's jvp."""
    x = np.linspace(0.5, 15, 30)
    expected = 0.5 * (scipy.special.jv(order - 1, x) - scipy.special.jv(order + 1, x))
    np.testing.assert_allclose(
        _first_kind_bessel_deriv(x, order), expected, rtol=1e-8, atol=1e-10
    )


# Root finder


def test_root_finder_returns_zero_for_low_pass_case():
    assert _find_neumann_root_muller(0, 0) == 0.0


@pytest.mark.parametrize("m,k", [(1, 0), (2, 1), (3, 2)])
def test_root_finder_returns_zero_when_k_less_than_m(m, k):
    """By convention there is no k-th root of J_m' when k < m."""
    assert _find_neumann_root_muller(m, k) == 0.0


@pytest.mark.parametrize(
    "m,k",
    [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (3, 4)],
)
def test_root_finder_matches_scipy_reference_roots(m, k):
    scipy_index = k if m == 0 else k - m + 1
    expected = scipy.special.jnp_zeros(m, scipy_index)[-1]
    found = _find_neumann_root_muller(m, k)
    assert found == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("m,k", [(0, 2), (1, 3), (2, 4)])
def test_root_finder_result_is_actually_a_root(m, k):
    root = _find_neumann_root_muller(m, k)
    assert root > 0  # a vacuous 0.0 return must not pass (J_m'(0) = 0 for m >= 1)
    assert abs(_first_kind_bessel_deriv(root, m)) < 1e-8


@pytest.mark.parametrize("m", [0, 1, 2])
def test_root_finder_roots_increase_strictly_with_k(m):
    ks = range(max(m, 1), max(m, 1) + 4)
    roots = [_find_neumann_root_muller(m, k) for k in ks]
    assert all(r > 0 for r in roots)
    assert all(b > a for a, b in zip(roots, roots[1:]))


# Wavelet / filter mathematical properties


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("m,k", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_wavelet_output_shape_and_finite(impl, m, k):
    Z = _wavelet(impl, m, k)
    assert Z.shape == (SIZE, SIZE)
    assert np.all(np.isfinite(Z))
    assert np.max(np.abs(Z)) > 0  # guards against an all-zero output


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("m,k", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_bandpass_wavelet_is_approximately_zero_mean(impl, m, k):
    Z = _wavelet(impl, m, k)
    center = SIZE // 2
    peak = np.max(np.abs(Z))
    assert peak > 0
    assert np.abs(Z[center, center]) < 1e-6 * peak


@pytest.mark.parametrize("m,k", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_fourier_bessel_wavelet_energy_is_away_from_dc(m, k):
    """Bandpass: peak is not at DC and little energy sits in the lowest frequencies."""
    Z = _fb_wavelet(m, k)
    radius, dx = _fb_radius_grid()
    energy = np.abs(Z) ** 2
    peak_idx = np.unravel_index(np.argmax(energy), energy.shape)
    assert radius[peak_idx] > 0
    near_dc_fraction = energy[radius <= 2 * dx].sum() / energy.sum()
    assert near_dc_fraction < 0.05


@pytest.mark.parametrize("m,k", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_solid_harmonic_wavelet_energy_is_away_from_dc(m, k):
    X, Y = _shifted_meshgrid()
    Z = _solid_harmonic_wavelet_fourier(m, k, X, Y)
    radius = np.hypot(X, Y)
    dx = 2 * np.pi / SIZE
    energy = np.abs(Z) ** 2
    peak_idx = np.unravel_index(np.argmax(energy), energy.shape)
    assert radius[peak_idx] > 0
    near_dc_fraction = energy[radius <= 2 * dx].sum() / energy.sum()
    assert near_dc_fraction < 0.05


def test_low_pass_filter_peaks_at_zero_frequency():
    X, Y = _shifted_meshgrid()
    _, Z_fb = _generate_fourier_low_pass_filter(
        size=SIZE, sigma=SIGMA, norm="l1", freq_limit=FREQ_LIMIT
    )
    Z_sh = _solid_harmonics_low_pass_filter(SH_LOW_PASS_PARAM, X, Y)
    c = SIZE // 2
    for Z in (Z_fb, Z_sh):
        mag = np.abs(Z)
        assert mag[c, c] == pytest.approx(np.max(mag), rel=1e-6)
        for neighbour in (mag[c - 1, c], mag[c + 1, c], mag[c, c - 1], mag[c, c + 1]):
            assert mag[c, c] > neighbour


@pytest.mark.parametrize("m,k", [(0, 1), (1, 1), (1, 2), (2, 2)])
def test_l1_norm_gives_unit_peak_magnitude_fourier_bessel(m, k):
    Z = _fb_wavelet(m, k, norm="l1")
    assert np.max(np.abs(Z)) == pytest.approx(1.0, rel=1e-6)


def test_solid_harmonic_wavelet_has_unit_peak_magnitude():
    Z = _sh_wavelet(1, 1)
    assert np.max(np.abs(Z)) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("m,k", [(0, 1), (1, 1), (2, 2)])
def test_l2_norm_gives_approximately_unit_energy(m, k):
    Z = _fb_wavelet(m, k, norm="l2")
    dx = (2 * FREQ_LIMIT) / SIZE
    energy = np.sum(np.abs(Z) ** 2) * (dx * dx)
    assert energy == pytest.approx(1.0, rel=0.05)