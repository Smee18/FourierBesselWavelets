from collections.abc import Iterable

import numpy as np

from .bank_class import WaveletBank
from .math_utils import (
    _choose_fourier_bound,
    _find_neumann_root_muller,
    _generate_fourier_bessel_wavelet,
    _generate_fourier_low_pass_filter,
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
        super().__init__(size, verbose)

    def _prepare(self) -> None:
        k_max = int(self._k_values.max()) if len(self._k_values) else 0
        self.lambda_max = _find_neumann_root_muller(0, k_max)
        self._freq_limit = _choose_fourier_bound(
            sigma=self.sigma, eigenvalue_max=self.lambda_max, size=self.size
        )

    @property
    def freq_limit(self) -> float:
        return self._freq_limit

    @property
    def m_values(self) -> np.ndarray:
        return self._m_values

    @property
    def k_values(self) -> np.ndarray:
        return self._k_values

    def _indices(self) -> Iterable[tuple[int, int]]:
        for k in self._k_values:
            for m in self._m_values:
                if abs(m) <= k:
                    yield (int(m), int(k))

    def _generate(self, index: tuple[int, int]) -> np.ndarray:
        m, k = index
        if k == 0:
            _, Z = _generate_fourier_low_pass_filter(
                size=self.size,
                sigma=self.sigma,
                norm=self.norm,
                freq_limit=self.freq_limit,
                verbose=self.verbose,
            )
        else:
            _, _, Z = _generate_fourier_bessel_wavelet(
                m,
                k,
                size=self.size,
                sigma=self.sigma,
                norm=self.norm,
                freq_limit=self.freq_limit,
                verbose=self.verbose,
            )
        return Z

    def _params(self) -> dict[str, float | str]:
        return {"m": self.m, "k": self.k, "sigma": self.sigma, "norm": self.norm}
