from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.special

from .logger_config import setup_logger

logger = setup_logger(__name__)

### BESSEL WRAPPERS VIA SCIPY ###


def _first_kind_bessel(X: npt.ArrayLike, order: int | np.integer) -> float | np.ndarray:
    """Vectorised Bessel function supporting both scalar points and arrays."""
    X_arr = np.asarray(X)
    abs_order = int(abs(int(order)))

    if order < 0:
        val = (-1) ** abs_order * scipy.special.jv(abs_order, X_arr)
    else:
        val = scipy.special.jv(int(order), X_arr)

    # If the input was a scalar, return a Python float; otherwise the array
    if np.isscalar(X):
        return float(np.asarray(val).item())
    return cast(np.ndarray, np.asarray(val))


def _first_kind_bessel_deriv(X: npt.ArrayLike, order: int | np.integer) -> float | np.ndarray:
    """Derivative supporting both scalar points and arrays using recurrence relations."""
    val = 0.5 * (_first_kind_bessel(X, order - 1) - _first_kind_bessel(X, order + 1))
    return val


def _first_modified_bessel(
    X: npt.ArrayLike, order: int | np.integer, scaled: bool = False
) -> float | np.ndarray:
    """Modified Bessel function supporting both scalar points, arrays, and exponential scaling."""
    X_arr = np.asarray(X)

    if scaled:
        val = scipy.special.ive(order, X_arr)
    else:
        val = scipy.special.iv(order, X_arr)

    if np.isscalar(X):
        return float(np.asarray(val).item())
    return cast(np.ndarray, np.asarray(val))


### MULLER'S METHOD TO FIND EIGENVALUE ###


def _mcmahon_seed(m: int | np.integer, k: int | np.integer) -> float:
    """Approximate the k-th positive root using McMahon's asymptotic expansion."""
    m = int(m)
    k = int(k)
    beta: float
    if m == 0:
        s = k
        nu = 1
        beta = (s + nu / 2.0 - 0.25) * np.pi
        return beta - (4.0 * nu**2 - 1.0) / (8.0 * beta)
    else:
        s = k - m + 1
        beta = (s + m / 2.0 - 0.75) * np.pi
        if beta <= 0:
            return float(m) + 0.5
        return beta - (4.0 * m**2 + 3.0) / (8.0 * beta)


def _find_neumann_root_muller(
    m: int | np.integer, k: int | np.integer, thresh: float = 1e-12, max_iter: int = 200
) -> float:
    """Find the k-th positive root of using Muller's method."""

    x3 = 0.0

    if k < m or (m == 0 and k == 0):
        return x3

    x_seed = _mcmahon_seed(m, k)
    x0 = max(0.01, x_seed - 0.15)
    x1 = x_seed
    x2 = x_seed + 0.15

    def f(x: npt.ArrayLike) -> float | np.ndarray:
        return _first_kind_bessel_deriv(x, m)

    d0: float = float(f(x0))
    d1: float = float(f(x1))
    d2: float = float(f(x2))

    converged = False

    for _ in range(max_iter):
        h1 = x1 - x0
        h2 = x2 - x1
        if h1 == 0 or h2 == 0:
            break
        delta1 = (d1 - d0) / h1
        delta2 = (d2 - d1) / h2
        if (h2 + h1) == 0:
            break
        d_coef = (delta2 - delta1) / (h2 + h1)
        a = d_coef
        b = delta2 + h2 * d_coef
        c = d2
        disc = np.lib.scimath.sqrt(b**2 - 4 * a * c)

        dx = -2 * c / (b + disc) if np.real(b) >= 0 else -2 * c / (b - disc)

        x3 = np.real(x2 + dx).item()
        d3 = float(f(x3))
        if abs(d3) < thresh or abs(dx) < thresh:
            converged = True
            break
        x0, x1, x2 = x1, x2, x3
        d0, d1, d2 = d1, d2, d3

    if not converged:
        logger.warning(
            "Root finder did not converge for m=%d, k=%d after %d iterations "
            "(residual=%.2e). Consider increasing max_iter.",
            int(m),
            int(k),
            max_iter,
            abs(d3) if "d3" in locals() else float("nan"),
        )
    return float(x3)


### FOURIER WAVELET ###


def _generate_fourier_bessel_wavelet(
    m: int | np.integer,
    k: int | np.integer,
    Q: float | np.ndarray,
    Psi: float | np.ndarray,
    dx: float,
    sigma: float = 1.0,
    norm: str = "l1",
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a complex Fourier-Bessel wavelet in the frequency domain.
    Refer to Eq(127) in "Notes on Fourier-Bessel wavelets"

    Args:
        m: Angular order.
        k: Radial eigenvalue index.
        Q: Radial frequency grid.
        Psi: Angular frequency grid.
        dx: Frequency grid spacing. for norm verbose
        sigma: Standard deviation of the Gaussian envelope.
        norm: Normalisation method ('l1' or 'l2').
        verbose: Whether to log diagnostic information.

    Returns:
        The complex wavelet.
    """

    abs_m = np.abs(m)
    eigenvalue = _find_neumann_root_muller(abs_m, k)
    sigma2 = sigma**2
    eig2 = eigenvalue**2
    angular_profile = np.exp(1j * m * Psi)
    K = np.exp(-(eig2 * sigma2) / 2)  # Eq(79) in "Notes on Fourier-Bessel wavelets"
    Z: np.ndarray

    mod_bessel = _first_modified_bessel((eig2 * sigma2) / 2, m, True)
    mod_bessel_freq = _first_modified_bessel(eigenvalue * sigma2 * Q, m, True)

    start = ((-1j) ** m) * angular_profile
    left_hand = (
        sigma2 * np.exp(-(sigma2 * (eig2 + Q**2)) / 2 + eigenvalue * sigma2 * Q) * mod_bessel_freq
    )

    if m == 0:
        bracket = mod_bessel - 2 * np.exp(-(3 * sigma2 * eig2) / 4) + np.exp(-sigma2 * eig2)
        norm_term = 1 / (np.sqrt((np.pi * sigma2) * bracket))

    else:
        norm_term = 1 / (np.sqrt(np.pi * sigma2 * mod_bessel))

    if m == 0:
        Z = start * norm_term * (left_hand - K * sigma2 * np.exp(-(sigma2 * Q**2) / 2))
    else:
        Z = start * norm_term * left_hand

    if norm == "l1":
        z_max = np.max(np.abs(Z))
        Z /= z_max

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        mean_val = np.abs(np.mean(Z))
        if norm == "l2":
            post_norm_energy = np.sum(np.abs(Z) ** 2) * (dx * dx)
        else:
            post_norm_energy = np.max(np.abs(Z))

        logger.info(
            f"Wavelet (m={m}, k={k}) | "
            f"Mean (~0): {mean_val:.2e} | "
            f"{norm.upper()}: {post_norm_energy:.4f} | "
            f"Eigenvalue: {eigenvalue:.4f}"
        )
    # -------------------------

    return Z


def _generate_fourier_low_pass_filter(
    Q: float | np.ndarray,
    dx: float,
    sigma: float = 1.0,
    norm: str = "l1",
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a complex Fourier-Bessel low pass filter in the frequency domain.
    This is equivalent to the m=0, k=0 wavelet.
    Refer to Eq(127) in "Notes on Fourier-Bessel wavelets"

    Args:
        Q: Radial frequency grid.
        dx: Frequency grid spacing. for norm verbose
        sigma: Standard deviation of the Gaussian envelope.
        norm: Normalisation method ('l1' or 'l2').
        verbose: Whether to log diagnostic information.

    Returns:
        The real low pass filter.
    """

    sigma2 = sigma**2
    Z = np.exp(-(sigma2) * (Q**2) / 2)

    if norm == "l1":
        z_max = np.max(np.abs(Z))
        Z /= z_max

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        mean_val = np.abs(np.mean(Z))
        if norm == "l2":
            post_norm_energy = np.sum(np.abs(Z) ** 2) * (dx * dx)
        else:
            post_norm_energy = np.max(np.abs(Z))

        logger.info(
            f"Low pass filter | Mean (>0): {mean_val:.2e} | {norm.upper()}: {post_norm_energy:.4f}"
        )
    # -------------------------

    return Z


### SOLID HARMONICS ###


def _solid_harmonic_wavelet_fourier(
    j: int,
    l: int,
    lambda_val: np.ndarray,
    alpha: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a complex Solid Harmonic wavelet in the frequency domain.
    Refer to Section 2.2 of Solid Harmonic Wavelet Scattering (NIPS 2017)

    Args:
        j: Scale
        l: Angular order
        lambda_val: 2D frequency polar coordinates
        alpha_val: 2D frequency polar coordinates
        verbose: Whether to log diagnostic information.

    Returns:
        The complex wavelet.
    """

    # Apply dyadic dilation (2^j) to the radial frequency
    lambda_scaled = lambda_val * (2.0**j)

    # Construct the Fourier components
    prefactor = (-1j) ** l
    gaussian_envelope = np.exp(-0.5 * lambda_scaled**2)
    polynomial = lambda_scaled**l
    complex_phase = np.exp(1j * l * alpha)

    # Combine to form the final complex wavelet
    Z = prefactor * gaussian_envelope * polynomial * complex_phase

    peak_magnitude_pre = np.max(np.abs(Z))
    final: np.ndarray = Z / peak_magnitude_pre

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        mean_val = np.abs(np.mean(final))
        post_norm_energy = np.max(np.abs(final))

        logger.info(
            f"Wavelet (j={j}, l={l}) | "
            f"Mean (~0): {mean_val:.2e} | "
            f"{'L1'}: {post_norm_energy:.4f}"
        )

    return final


def _solid_harmonics_low_pass_filter(
    j: int,
    lambda_val: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    low_pass_filter: np.ndarray = np.exp(-0.5 * (lambda_val * (2.0**j)) ** 2)
    """
    Generate a complex Solid Harmonic low pass filter in the frequency domain.
    Refer to Section 2.2 of Solid Harmonic Wavelet Scattering (NIPS 2017)

    Args:
        m: Largest scale
        lambda_val: 2D frequency polar coordinates
        verbose: Whether to log diagnostic information.

    Returns:
        The complex low pass filter.
    """

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        mean_val = np.abs(np.mean(low_pass_filter))
        post_norm_energy = np.max(np.abs(low_pass_filter))

        logger.info(f"Low pass filter | Mean (>0): {mean_val:.2e} | {'L1'}: {post_norm_energy:.4f}")
    return low_pass_filter
