import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, KeysView, ValuesView
from .math_utils import (
    _find_neumann_root_muller, 
    _generate_fourier_bessel_wavelet, 
    _generate_fourier_low_pass_filter
)

class FourierBesselWaveletBank:
    """A bank of Fourier-Bessel wavelets indexed by parameters m and k.
    
    Attributes:
        size (int): Image size.
        m (int): Maximum order .
        k (int): Maximum angular index.
        sigma (float): Scale parameter for the wavelets.
    """

    def __init__(self, size: int, m: int, k: int, sigma: float, verbose: bool = False) -> None:

        self.size = size
        self.m = m
        self.k = k
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.m_values = np.arange(0, m)
        self.k_values = np.arange(0, k)
        self.verbose = verbose

        assert self.m <= self.k, "m <= k condition is not respected"

        self.lambda_max = _find_neumann_root_muller(0, int(self.k_values.max()) if len(self.k_values) > 0 else 0)
        wavelet_bank: Dict[str, np.ndarray] = {}

        self.freq_limit = int(self.lambda_max + 2 / self.sigma)

        for k_val in self.k_values:
            for m_val in self.m_values:
                if np.abs(m_val) > k_val:
                    continue

                if k_val == 0:
                    _, Z = _generate_fourier_low_pass_filter(size=self.size, sigma=sigma, freq_limit=self.freq_limit, verbose=self.verbose)
                else:
                    _, _, Z = _generate_fourier_bessel_wavelet(m_val, k_val, size=self.size, sigma=sigma, freq_limit=self.freq_limit, verbose=self.verbose)

                key_name = f'm_{m_val}_k_{k_val}_s{self.sigma}'
                wavelet_bank[key_name] = Z

        self.wavelet_bank = wavelet_bank
        
    def __getitem__(self, m_index: int, k_index: int) -> np.ndarray:
        """Retrieve a specific wavelet by its parameters.
        
        Args:
            m_index (int): Order.
            k_index (int): Angular index.
            
        Returns:
            np.ndarray: The 2D array representing the requested wavelet.
        """

        key = f'm_{m_index}_k_{k_index}_s{self.sigma}'
        if key not in self.wavelet_bank:
            raise KeyError(f"Wavelet with m={m_index}, k={k_index} not found in bank.")
        return self.wavelet_bank[key]

    def __len__(self):

        return len(self.wavelet_bank)

    def get_keys(self) -> KeysView[str]:
        """Return the dictionary keys representing individual wavelets."""

        return self.wavelet_bank.keys()

    def get_values(self) -> ValuesView[np.ndarray]:
        """Return the collection of wavelet arrays."""

        return self.wavelet_bank.values()

    def summary(self, verbose: bool = True):
        """Print a summary of the wavelet bank parameters and size."""
        if verbose:
            print("Fourier-Bessel Wavelet bank summary:\n")
            print(f"Parameters: m = {self.m}, k = {self.k}, sigma = {self.sigma}\n")
            print(f"Total wavelets: {len(self.wavelet_bank)}")
            print(f"Frequency limit: {self.freq_limit:.2f}")
            print("Key naming structure: {m_{m_val}_k_{k_val}_s{sigma}}")

        return self.m, self.k, self.sigma
        
    def plot_bank(self) -> None:
        """Plot the grid of wavelets using matplotlib."""
        _, axes = plt.subplots(len(self.k_values), len(self.m_values), figsize=(8, 6),
                                sharex=True, sharey=True)

        axes = np.atleast_2d(axes)
        for row_idx, k_val in enumerate(self.k_values):
            for col_idx, m_val in enumerate(self.m_values):
                ax = axes[row_idx, col_idx]
                ax.set_xlim(-self.freq_limit, self.freq_limit)
                ax.set_ylim(-self.freq_limit, self.freq_limit)
                ax.axis('off')

                if row_idx == 0:
                    ax.set_title(f'm = {m_val}', fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.text(-self.freq_limit*1.2, 0, f'k = {k_val}', fontsize=12, fontweight='bold',
                            ha='right', va='center')

                if np.abs(m_val) > k_val:
                    continue

                if f'm_{m_val}_k_{k_val}_s{self.sigma}' not in self.wavelet_bank:
                    continue

                Z = self.wavelet_bank[f'm_{m_val}_k_{k_val}_s{self.sigma}']
                z_max = np.max(np.abs(Z))
                ax.imshow(np.real(Z),
                        extent=[-self.freq_limit, self.freq_limit, -self.freq_limit, self.freq_limit],
                        cmap='inferno', origin='lower', vmin=-z_max, vmax=z_max)  

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()