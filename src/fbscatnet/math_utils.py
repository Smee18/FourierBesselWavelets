import numpy as np
import scipy.special

### BESSEL WRAPPERS VIA SCIPY ###


def _first_kind_bessel(X, order: int):
    """Vectorized Bessel function of the first kind supporting negative orders."""

    if order < 0:
        return (-1) ** abs(order) * scipy.special.jv(abs(order), X)
    return scipy.special.jv(order, X)


def _first_kind_bessel_deriv(X, order: int):
    """Vectorized derivative using recurrence relations."""
    X_arr = np.atleast_1d(X).astype(float)
    return 0.5 * (_first_kind_bessel(X_arr, order - 1) - _first_kind_bessel(X_arr, order + 1))


def _first_modified_bessel(X, order: int):
    """Vectorized modified Bessel function of the first kind."""

    return scipy.special.iv(order, X)


### MULLER'S METHOD TO FIND EIGENVALUE ###


def _mcmahon_seed(m, k):
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


def _find_neumann_root_muller(m, k, thresh=1e-12, max_iter=200) -> float:

    x3 = 0.0

    if k < m or (m == 0 and k == 0):
        return x3

    x_seed = _mcmahon_seed(m, k)
    x0 = max(0.01, x_seed - 0.15)
    x1 = x_seed
    x2 = x_seed + 0.15

    def f(x):
        return _first_kind_bessel_deriv(x, m)

    d0, d1, d2 = f(x0), f(x1), f(x2)

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
        d3 = f(x3)
        if abs(d3) < thresh or abs(dx) < thresh:
            break
        x0, x1, x2 = x1, x2, x3
        d0, d1, d2 = d1, d2, d3

    return x3


### FOURIER WAVELET ###


def _generate_fourier_bessel_wavelet(m, k, size=50, sigma=0.1, freq_limit=20, verbose=False):
    abs_m = np.abs(m)
    freq = np.linspace(-freq_limit, freq_limit, size)
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
            f"Wavelet  (m={m}, k={k}) | Mean (~0): {mean_val:.2e} | \
              L2: {post_norm_energy:.4f} Eigenvalue: {eigenvalue:.4f}"
        )
    # -------------------------

    return Kx, Ky, Z


def _generate_fourier_low_pass_filter(size=50, sigma=0.1, freq_limit=20, verbose=False):
    freq = np.linspace(-freq_limit, freq_limit, size)
    Kx, Ky = np.meshgrid(freq, freq, indexing="ij")
    Q = np.sqrt(Kx**2 + Ky**2)

    sigma2 = sigma**2
    Z = np.exp(-2 * (np.pi**2) * sigma2 * (Q**2))

    # --- DIAGNOSTIC PRINTS ---
    if verbose:
        dx = freq[1] - freq[0]
        mean_val = np.abs(np.mean(Z))
        post_norm_energy = np.sum(np.abs(Z)) * (dx * dx)

        print(f"Low pass  (m=0, k=0) | Mean (~0): {mean_val:.2e} | L1: {post_norm_energy:.4f}")
    # -------------------------

    return Q, Z
