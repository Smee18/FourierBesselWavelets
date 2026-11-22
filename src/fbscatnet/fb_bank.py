from collections.abc import Iterator

import numpy as np

from .bank_class import WaveletBank
from .math_utils import (
    _generate_fourier_bessel_wavelet,
    _generate_fourier_low_pass_filter,
    _find_neumann_root_muller
)


class FourierBesselWaveletBank(WaveletBank):
    family_name = "Fourier Bessel"

    def __init__(
        self, size: int, m: int, k: int, sigma: float = 1.0, norm: str = "l1", verbose: bool = False
    ):
        if size <= 0:
            raise ValueError("Size must be positive")
        if m < 0 or k < 0 or sigma <= 0:
            raise ValueError("Cannot accept negative parameters")
        if m > k:
            raise ValueError("m <= k condition is not respected")
        if norm not in ("l1", "l2"):
            raise ValueError(f"Invalid norm: must be 'l1' or 'l2', not {norm}")

        self.m, self.k = m, k
        self.sigma, self.norm = sigma, norm
        self._m_values = np.arange(0, m)
        self._k_values = np.arange(0, k)
        lambda_max = _find_neumann_root_muller(m, k)
        frequency_scale = np.pi / (lambda_max + 6 / sigma)
        axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size))
        fx, fy = np.meshgrid(axis, axis, indexing="ij")
        self.q = np.hypot(fx, fy) / frequency_scale
        self.psi = np.arctan2(fy, fx)
        self.dx = (axis[1] - axis[0]) / frequency_scale
        del fx, fy
        super().__init__(size, verbose)

    def _indices(self) -> Iterator[tuple[int, int]]:
        for k in self._k_values:
            for m in self._m_values:
                if abs(m) <= k:
                    yield (int(m), int(k))

    def _generate(self, index: tuple[int, int]) -> np.ndarray:
        m, k = index
        if k == 0:
            Z = _generate_fourier_low_pass_filter(
                Q=self.q,
                dx=self.dx,
                sigma=self.sigma,
                norm=self.norm,
                verbose=self.verbose,
            )
        else:
            Z = _generate_fourier_bessel_wavelet(
                m=m,
                k=k,
                Q=self.q,
                Psi=self.psi,
                dx=self.dx,
                sigma=self.sigma,
                norm=self.norm,
                verbose=self.verbose,
            )
        return Z

    def _params(self) -> dict[str, int | float | str]:
        return {"m": self.m, "k": self.k, "sigma": self.sigma, "norm": self.norm}

    def _grid_layout(self) -> tuple[list[int], list[int], str, str]:
        return list(self._k_values), list(self._m_values), "k", "m"

    def _grid_index(self, row: int, col: int) -> tuple[int, int]:
        return (col, row)

    def _make_key(self, index: tuple[int, int]) -> str:
        m, k = index
        return f"m_{m}_k_{k}"
