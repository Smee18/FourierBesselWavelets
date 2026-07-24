from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.special

### BESSEL WRAPPERS VIA SCIPY ###


def _first_kind_bessel(X: npt.ArrayLike, order: int | np.integer) -> float | np.ndarray:
    """Vectorized Bessel function supporting both scalar points and arrays."""
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


def _first_modified_bessel(X: npt.ArrayLike, order: int | np.integer) -> float | np.ndarray:
    """Modified Bessel function supporting both scalar points and arrays."""

    X_arr = np.asarray(X)
    val = scipy.special.iv(order, X_arr)

    # If the input was a scalar, return a Python float; otherwise the array
    if np.isscalar(X):
        return float(np.asarray(val).item())
    return cast(np.ndarray, np.asarray(val))


### MULLER'S METHOD TO FIND EIGENVALUE ###


def _mcmahon_seed(m: int | np.integer, k: int | np.integer) -> float:
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
            break
        x0, x1, x2 = x1, x2, x3
        d0, d1, d2 = d1, d2, d3

    return float(x3)


### FOURIER WAVELET ###


def _generate_fourier_bessel_wavelet(
    m: int | np.integer,
    k: int | np.integer,
    size: int = 50,
    sigma: float = 0.1,
    freq_limit: int = 20,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    abs_m = np.abs(m)
    freq = np.linspace(-freq_limit, freq_limit, size)
    Kx: np.ndarray
    Ky: np.ndarray
    Kx, Ky = np.meshgrid(freq, freq, indexing="ij")

    Q = np.sqrt(Kx**2 + Ky**2)
    Psi = np.arctan2(Ky, Kx)

    eigenvalue = _find_neumann_root_muller(abs_m, k)
    sigma2 = sigma**2
    eig2 = eigenvalue**2
    angular_profile = np.exp(1j * m * Psi)
    K = np.exp(-(eig2 * sigma2) / 2)

    mod_bessel = _first_modified_bessel((eig2 * sigma2) / 2, m)
    mod_bessel_freq = _first_modified_bessel(eigenvalue * sigma2 * Q, m)

    start = (1j**m) * angular_profile
    left_hand = sigma2 * np.exp(-(sigma2 * (eig2 + Q**2)) / 2) * mod_bessel_freq

    if m == 0:
        bracket = K * mod_bessel - 2 * np.exp(-(3 * sigma2 * eig2) / 4) + np.exp(-sigma2 * eig2)
        norm_term = 1 / (np.sqrt((np.pi * sigma2) * bracket))
        Z = start * norm_term * (left_hand - K * sigma2 * np.exp(-(sigma2 * Q**2) / 2))
    else:
        norm_term = 1 / (np.sqrt(np.pi * sigma2 * K * mod_bessel))
        Z = start * norm_term * left_hand

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        dx = freq[1] - freq[0]
        mean_val = np.abs(np.mean(Z))
        post_norm_energy = np.sum(np.abs(Z) ** 2) * (dx * dx)

        print(
            f"Wavelet (m={m}, k={k}) | "
            f"Mean (~0): {mean_val:.2e} | "
            f"L2: {post_norm_energy:.4f} "
            f"Eigenvalue: {eigenvalue:.4f}"
        )
    # -------------------------

    return Kx, Ky, Z


def _generate_fourier_low_pass_filter(
    size: int = 50, sigma: float = 0.1, freq_limit: int = 20, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    freq = np.linspace(-freq_limit, freq_limit, size)
    Kx: np.ndarray
    Ky: np.ndarray
    Kx, Ky = np.meshgrid(freq, freq, indexing="ij")
    Q = np.sqrt(Kx**2 + Ky**2)

    sigma2 = sigma**2
    Z = np.exp(-(sigma2) * (Q**2) / 2)

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        dx = freq[1] - freq[0]
        mean_val = np.abs(np.mean(Z))
        post_norm_energy = np.sum(np.abs(Z)) * (dx * dx)

        print(f"Low pass  (m=0, k=0) | Mean (~0): {mean_val:.2e} | L1: {post_norm_energy:.4f}")
    # -------------------------

    return Q, Z
