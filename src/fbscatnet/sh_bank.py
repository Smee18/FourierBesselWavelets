from collections.abc import Iterator

import numpy as np

from .bank_class import WaveletBank
from .math_utils import _solid_harmonic_wavelet_fourier, _solid_harmonics_low_pass_filter


class SolidHarmonicsWaveletBank(WaveletBank):
    family_name = "Solid Harmonics"

    def __init__(self, size: int, j: int, l: int, verbose: bool = False):
        if j < 1 or l < 0:
            raise ValueError("Require j >= 1 and l >= 0")
        self.j, self.l = j, l
        self._j_values = np.arange(0, j + 1)
        self._l_values = np.arange(0, l + 1)
        axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size))
        fx, fy = np.meshgrid(axis, axis, indexing="ij")  # Create frequency meshgrid
        self.lambda_val = np.sqrt(fx**2 + fy**2)
        self.alpha = np.arctan2(fy, fx)
        del fx, fy
        super().__init__(size, verbose)

    def _indices(self) -> Iterator[tuple[int, int]]:
        yield (0, 0)  # kow pass filter
        for j in self._j_values[1:]:
            for l in self._l_values:
                yield (int(j), int(l))

    def _generate(self, index: tuple[int, int]) -> np.ndarray:
        j, l = index
        if j == 0:  # Low pass
            return _solid_harmonics_low_pass_filter(
                j=j, lambda_val=self.lambda_val, verbose=self.verbose
            )
        return _solid_harmonic_wavelet_fourier(
            j=j, l=l, lambda_val=self.lambda_val, alpha=self.alpha, verbose=self.verbose
        )

    def _params(self) -> dict:
        return {"j": self.j, "l": self.l}

    def _grid_layout(self) -> tuple[list[int], list[int], str, str]:
        return list(self._j_values[1:]), list(self._l_values), "j", "l"

    def _grid_index(self, row: int, col: int) -> tuple[int, int]:
        return (row, col)

    def _make_key(self, index: tuple[int, int]) -> str:
        j, l = index
        return f"j_{j}_l_{l}"
