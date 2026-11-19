from abc import ABC, abstractmethod
from collections.abc import Iterable, KeysView, ValuesView

import matplotlib.pyplot as plt
import numpy as np

from .logger_config import setup_logger

logger = setup_logger(__name__)


class WaveletBank(ABC):
    """Abstract base for a bank of wavelets indexed by an integer tuple."""

    family_name: str = "Wavelet"

    def __init__(self, size: int, verbose: bool = False):

        self.size = size
        self.verbose = verbose

        self.wavelet_bank: dict[str, np.ndarray] = {}
        self.index_to_key: dict[tuple, str] = {}
        self._build()

    # Children abstract methods to be implemented
    @abstractmethod
    def _prepare(self) -> None:
        """Optional: compute family-specific quantities before building."""

    @abstractmethod
    def _indices(self) -> Iterable[tuple[int, int]]:
        """Yield every valid index tuple, (m, k)"""

    @abstractmethod
    def _generate(self, index: tuple) -> np.ndarray:
        """Construct the wavelet for one index."""

    @abstractmethod
    def _params(self) -> dict:
        """Parameters to show in summary function."""

    # Shared wavelet construction function
    def _build(self) -> None:
        self._prepare()  # Get wavelet specific parameters
        for index in self._indices():
            key = self._make_key(index)  # Create str key for dico
            self.index_to_key[index] = key
            self.wavelet_bank[key] = self._generate(index)

    # Shared retrieval function
    def __getitem__(self, key_or_indices: str | tuple) -> np.ndarray:
        if isinstance(key_or_indices, str):
            try:
                return self.wavelet_bank[key_or_indices]
            except KeyError:
                raise KeyError(f"Wavelet key '{key_or_indices}' not found.") from None
        if isinstance(key_or_indices, tuple):
            key = self.index_to_key.get(key_or_indices)
            if key is None:
                raise KeyError(f"Wavelet with index {key_or_indices} not found.")
            return self.wavelet_bank[key]
        raise TypeError("Invalid index type. Use a string key or an index tuple.")

    def __len__(self) -> int:
        return len(self.wavelet_bank)

    def __iter__(self) -> Iterable[tuple[str, np.ndarray]]:
        return iter(self.wavelet_bank.items())

    def get_keys(self) -> KeysView[str]:
        return self.wavelet_bank.keys()

    def get_values(self) -> ValuesView[np.ndarray]:
        return self.wavelet_bank.values()

    def _make_key(self, index: tuple[int, int]) -> str:
        m, k = index
        return f"m_{m}_k_{k}"

    def summary(self, verbose: bool = True) -> dict:
        params = self._params()  # Extract wavelet specific parameters
        if verbose:
            logger.info("%s Bank summary:", self.family_name)
            logger.info("Parameters: %s", ", ".join(f"{k} = {v}" for k, v in params.items()))
            logger.info("Total wavelets: %d", len(self))
        return params

    # Plotting function to be implemented by children
    @property
    @abstractmethod
    def freq_limit(self) -> float:
        """Return the frequency limit for plotting the wavelets."""

    @property
    @abstractmethod
    def m_values(self) -> np.ndarray:
        """Return the list of m values"""

    @property
    @abstractmethod
    def k_values(self) -> np.ndarray:
        """Return the list of k values"""

    # Shared plotting function

    def _grid_layout(self) -> tuple[list[int], list[int], str, str]:
        return list(self.k_values), list(self.m_values), "k", "m"

    def _grid_index(self, row: int, col: int) -> tuple[int, int]:
        return (int(col), int(row))

    def _to_2d(self, Z: np.ndarray) -> np.ndarray:
        return np.real(Z)

    def plot_bank(self) -> None:
        rows, cols, row_label, col_label = self._grid_layout()
        lim = self.freq_limit
        _, axes = plt.subplots(len(rows), len(cols), figsize=(8, 6), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        for r, row_val in enumerate(rows):
            for c, col_val in enumerate(cols):
                ax = axes[r, c]
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.axis("off")

                if r == 0:
                    ax.set_title(f"{col_label} = {col_val}", fontsize=12, fontweight="bold")
                if c == 0:
                    ax.text(
                        -lim * 1.2,
                        0,
                        f"{row_label} = {row_val}",
                        fontsize=12,
                        fontweight="bold",
                        ha="right",
                        va="center",
                    )

                key = self.index_to_key.get(self._grid_index(row_val, col_val))
                if key is None:
                    continue

                img = self._to_2d(self.wavelet_bank[key])
                z_max = float(np.max(np.abs(img)))
                ax.imshow(
                    img,
                    extent=[-lim, lim, -lim, lim],
                    cmap="inferno",
                    origin="lower",
                    vmin=-z_max,
                    vmax=z_max,
                )

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
