from collections.abc import Iterable

import numpy as np

from .bank_class import WaveletBank
from .math_utils import _solid_harmonic_wavelet_fourier, _solid_harmonics_low_pass_filter


class SolidHarmonicsWaveletBank(WaveletBank):
    family_name = "Solid Harmonics"

    def __init__(self, size: int, m: int, k: int, verbose: bool = False):
        if m < 1 or k < 0:
            raise ValueError("Require m >= 1 and k >= 0")
        self.m, self.k = m, k
        self._m_values = np.arange(0, m + 1)
        self._k_values = np.arange(0, k + 1)
        self.verbose = verbose
        axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size))
        self.X, self.Y = np.meshgrid(axis, axis)  # Create frequency meshgrid
        super().__init__(size, verbose)

    def _prepare(self) -> None:
        self._freq_limit = np.pi

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
        yield (0, 0)  # kow pass filter
        for m in self._m_values[1:]:
            for k in self._k_values:
                yield (int(m), int(k))

    def _generate(self, index: tuple[int, int]) -> np.ndarray:
        m, k = index
        if m == 0:  # kow pass
            return _solid_harmonics_low_pass_filter(
                self.m, X=self.X, Y=self.Y, verbose=self.verbose
            )
        return _solid_harmonic_wavelet_fourier(m, k, X=self.X, Y=self.Y, verbose=self.verbose)

    def _params(self) -> dict:
        return {"m": self.m, "k": self.k}
