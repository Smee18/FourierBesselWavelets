import numpy as np
import pytest
import scipy.special

from fbscatnet.generate_bank import FourierBesselWaveletBank
from fbscatnet.math_utils import (
    _find_neumann_root_muller,
    _first_kind_bessel,
    _first_kind_bessel_deriv,
    _generate_fourier_bessel_wavelet,
    _generate_fourier_low_pass_filter,
)

# Bessel wrappers


@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_first_kind_bessel_matches_scipy_for_positive_orders(order):
    x = np.linspace(0.1, 20, 50)
    expected = scipy.special.jv(order, x)
    np.testing.assert_allclose(_first_kind_bessel(x, order), expected, rtol=1e-10)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_first_kind_bessel_negative_order_identity(order):
    """J_{-n}(x) = (-1)^n J_n(x) for integer n."""
    x = np.linspace(0.1, 20, 50)
    positive = _first_kind_bessel(x, order)
    negative = _first_kind_bessel(x, -order)
    np.testing.assert_allclose(negative, ((-1) ** order) * np.asarray(positive), rtol=1e-10)


def test_first_kind_bessel_scalar_returns_python_float():
    result = _first_kind_bessel(1.5, 0)
    assert isinstance(result, float)


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_first_kind_bessel_deriv_matches_scipy_jvp(order):
    x = np.linspace(0.5, 15, 30)
    expected = scipy.special.jvp(order, x, n=1)
    np.testing.assert_allclose(_first_kind_bessel_deriv(x, order), expected, rtol=1e-8)


# Root finder


def test_root_finder_returns_zero_for_low_pass_case():
    assert _find_neumann_root_muller(0, 0) == 0.0


@pytest.mark.parametrize("m,k", [(1, 0), (2, 1), (3, 2)])
def test_root_finder_returns_zero_when_k_less_than_m(m, k):
    """By convention there is no k-th root of J_m' when k < m."""
    assert _find_neumann_root_muller(m, k) == 0.0


@pytest.mark.parametrize(
    "m,k,scipy_index",
    [
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),
        (2, 2, 1),
        (2, 3, 2),
        (3, 3, 1),
        (3, 4, 2),
    ],
)
def test_root_finder_matches_scipy_reference_roots(m, k, scipy_index):
    expected = scipy.special.jnp_zeros(m, scipy_index)[-1]
    found = _find_neumann_root_muller(m, k)
    assert found == pytest.approx(expected, abs=1e-8)


def test_root_finder_result_is_actually_a_root():
    for m, k in [(0, 2), (1, 3), (2, 4)]:
        root = _find_neumann_root_muller(m, k)
        assert abs(_first_kind_bessel_deriv(root, m)) < 1e-8


# Wavelet / filter mathematical properties


@pytest.mark.parametrize("m,k", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_bandpass_wavelet_is_approximately_zero_mean(m, k):
    _, _, Z = _generate_fourier_bessel_wavelet(m, k, size=129, sigma=0.3, norm="l2", freq_limit=20)
    size = Z.shape[0]
    center = size // 2
    # Value at the zero-frequency (DC) point should be tiny relative to the peak.
    assert np.abs(Z[center, center]) < 1e-6 * np.max(np.abs(Z))


def test_low_pass_filter_peaks_at_zero_frequency():
    _, Z = _generate_fourier_low_pass_filter(size=64, sigma=0.3, norm="l1", freq_limit=20)
    center = Z.shape[0] // 2
    assert Z[center, center] == pytest.approx(np.max(np.abs(Z)), rel=1e-6)


def test_l1_norm_gives_unit_peak_magnitude():
    _, _, Z = _generate_fourier_bessel_wavelet(1, 1, size=64, sigma=0.3, norm="l1", freq_limit=20)
    assert np.max(np.abs(Z)) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("m,k", [(0, 1), (1, 1), (2, 2)])
def test_l2_norm_gives_approximately_unit_energy(m, k):
    size = 256
    freq_limit = 20
    _, _, Z = _generate_fourier_bessel_wavelet(
        m, k, size=size, sigma=0.3, norm="l2", freq_limit=freq_limit
    )
    dx = (2 * freq_limit) / size
    energy = np.sum(np.abs(Z) ** 2) * (dx * dx)
    assert energy == pytest.approx(1.0, rel=0.05)


def test_bank_rejects_norm_other_than_l1_l2_but_accepts_both():
    for norm in ("l1", "l2"):
        bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1, norm=norm)
        assert bank.norm == norm
