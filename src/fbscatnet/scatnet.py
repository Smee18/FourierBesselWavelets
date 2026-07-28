import os
import re
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as spfft
from joblib import Parallel, delayed
from rainbow_tqdm import tqdm

from .generate_bank import FourierBesselWaveletBank
from .logger_config import setup_logger

logger = setup_logger(__name__)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import cupy as cp

    try:
        HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
    except Exception as exc:
        logger.debug("CuPy is installed but no usable GPU was detected: %s", exc)
        HAS_CUPY = False
except ImportError:
    cp = None
    HAS_CUPY = False


def _extract_k(key_str: str) -> int:
    """Extract the integer 'k' index encoded in a filter bank key string."""
    match = re.search(r"k[=_ ]*(\d+)", key_str)
    return int(match.group(1)) if match else 0


class FourierBesselScatNet:
    """A scattering network based on Fourier-Bessel wavelets for generating image embeddings.

    Attributes:
        size (int): Image spatial size.
        bank (FourierBesselWaveletBank): The bank of Fourier-Bessel wavelets.
        num_filters (int): Total number of filters in the wavelet bank.
        bank_keys (list[str]): List of string keys representing the filters.
        low_pass (np.ndarray): Low-pass filter array retrieved from the bank.
        final_features (np.ndarray): Generated feature embeddings from the last run.
    """

    def __init__(self, bank: FourierBesselWaveletBank, backend: str = "cpu") -> None:
        """Initialise the FourierBesselScatNet.

        Args:
            bank (FourierBesselWaveletBank): A configured bank of Fourier-Bessel wavelets.
            backend (str): Device to operate on

        Raises:
            ImportError: If CuPy requested but no GPU.
            ValueError: If the bank contains no band-pass wavelets (i.e. only
                the (0, 0) low-pass filter), since the scattering cascade has
                nothing to filter with in that case.
        """
        self.backend = backend.lower()
        if self.backend == "gpu" and not HAS_CUPY:
            raise ImportError(
                "GPU backend was requested, but 'cupy' is not"
                ""
                "installed or no compatible GPU was found."
            )
        if len(bank) <= 1:
            raise ValueError(
                "The wavelet bank contains no band-pass wavelets (only the "
                "low-pass filter). FourierBesselScatNet needs at least one "
                "wavelet - try constructing the bank with m > 1 and/or k > 1."
            )
        self.xp: Any = cp if self.backend == "gpu" else np
        self.size = bank[0, 0].shape[0]
        self.bank = bank
        self.num_filters = len(bank)
        self.bank_keys = list(bank.get_keys())
        self.low_pass = self.xp.asarray(bank[0, 0], dtype=self.xp.float32)
        self.final_features: Any | None = None

        # Precompute everything that is constant across batches
        xp = self.xp
        self.order_1_keys = self.bank_keys[1:]
        self.num_order_1_maps = len(self.order_1_keys)

        self._k_map = {key: _extract_k(key) for key in self.order_1_keys}

        self._order2_children: dict[str, list[str]] = {
            key1: [key2 for key2 in self.order_1_keys if self._k_map[key2] < self._k_map[key1]]
            for key1 in self.order_1_keys
        }
        self.num_order_2_maps = sum(len(v) for v in self._order2_children.values())

        self._filters_1 = {
            key: xp.asarray(bank[key], dtype=xp.complex64) for key in self.order_1_keys
        }
        self._low_pass_c = xp.asarray(bank[0, 0], dtype=xp.complex64)

        # Stacked filter tensors for vectorised filtering.
        # Shape: (num_order_1_maps, H, W)
        self._filters_1_stack = xp.stack([self._filters_1[k] for k in self.order_1_keys])

        # Per-key1 stacked children filters for vectorised order-2 filtering.
        # Shape per entry: (num_children, H, W), or None if no valid children.
        self._filters_2_stack_by_key1: dict[str, Any] = {
            key1: (xp.stack([self._filters_1[k2] for k2 in children]) if children else None)
            for key1, children in self._order2_children.items()
        }

    def _fft2(self, x: Any, xp: Any) -> Any:
        """Backend-aware 2D FFT (uses scipy on CPU for multi-threaded speed, cupy on GPU)."""
        if self.backend == "gpu":
            return xp.fft.fft2(x, axes=(-2, -1))
        return spfft.fft2(x, axes=(-2, -1), workers=-1)

    def _ifft2(self, x: Any, xp: Any) -> Any:
        """Backend-aware inverse 2D FFT, mirroring `_fft2`."""
        if self.backend == "gpu":
            return xp.fft.ifft2(x, axes=(-2, -1))
        return spfft.ifft2(x, axes=(-2, -1), workers=-1)

    def _filter_and_modulus(self, xp: Any, freq_signal: Any, filters: Any) -> Any:
        """Apply wavelet filter(s) in the frequency domain and take the spatial modulus.

        Args:
            xp: The array module to use (numpy or cupy) for this batch.
            freq_signal: Input already in fftshifted frequency domain,
                broadcastable against `filters`.
            filters: One or a stack of frequency-domain filters to apply.

        Returns:
            The fftshifted frequency-domain modulus.
        """
        filtered_fft = freq_signal * filters
        shifted_freq = xp.fft.ifftshift(filtered_fft, axes=(-2, -1))
        spatial_complex = self._ifft2(shifted_freq, xp)
        modulus_spatial = xp.abs(spatial_complex).astype(xp.float32)
        return xp.fft.fftshift(self._fft2(modulus_spatial, xp), axes=(-2, -1)).astype(xp.complex64)

    def _smooth_and_pool(
        self,
        xp: Any,
        modulus_fft: Any,
        low_pass_c: Any,
        batch_size: int,
        n_maps: int,
        d_size: int,
        downsize: int,
    ) -> Any:
        """Low-pass smooth a modulus map and block-mean-pool it down to `d_size`.

        Args:
            xp: The array module to use (numpy or cupy) for this batch.
            modulus_fft: Output of `_filter_and_modulus`, shape (batch, n_maps, H, W).
            low_pass_c: Broadcastable complex low-pass filter.
            batch_size: Number of samples in this batch.
            n_maps: Number of filter channels being pooled.
            d_size: Output spatial size after downsampling (H // downsize).
            downsize: Block-mean pooling factor.

        Returns:
            Real-valued pooled features of shape (batch_size, d_size, d_size, n_maps).
        """
        filtered_low_pass = modulus_fft * low_pass_c
        smoothed_shifted = xp.fft.ifftshift(filtered_low_pass, axes=(-2, -1))
        smoothed_spatial = self._ifft2(smoothed_shifted, xp)
        return (
            xp.real(smoothed_spatial)
            .reshape(batch_size, n_maps, d_size, downsize, d_size, downsize)
            .mean(axis=(3, 5))
            .transpose(0, 2, 3, 1)
        )

    def generate_embeddings(
        self,
        data: np.ndarray,
        downsize: int,
        batch_size: int = 32,
        use_multiprocessing: bool = False,
    ) -> np.ndarray:
        """Generate scattering network feature embeddings for a given dataset.

        Args:
            data (np.ndarray): Input image dataset of shape (num_samples, height, width).
            downsize (int): Spatial downsampling factor via block mean pooling.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            use_multiprocessing (bool, optional): Whether to use multiple CPU cores
                (Ignored if backend='gpu'). Defaults to False.

        Returns:
            np.ndarray: Flattened feature embeddings of shape (num_samples, feature_dim).
        """
        xp = self.xp
        data = np.asarray(data, dtype=np.float32)

        num_samples = data.shape[0]
        d_size = int(self.size / downsize)

        order_1_keys = self.order_1_keys
        num_order_1_maps = self.num_order_1_maps
        num_order_2_maps = self.num_order_2_maps

        def _process_batch(batch_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            b_xp: Any = cp if self.backend == "gpu" else np
            b_data = (
                b_xp.asarray(batch_data, dtype=b_xp.float32)
                if self.backend == "gpu"
                else batch_data
            )
            curr_batch_size = b_data.shape[0]

            batch_fft = b_xp.fft.fftshift(self._fft2(b_data, b_xp), axes=(-2, -1)).astype(
                b_xp.complex64
            )

            low_pass_c = self._low_pass_c

            # ORDER 0: plain low-pass response, no modulus non-linearity.
            low_pass_spatial = self._ifft2(
                b_xp.fft.ifftshift(batch_fft * low_pass_c, axes=(-2, -1)), b_xp
            )
            low_pass_down = (
                b_xp.real(low_pass_spatial)
                .reshape(-1, d_size, downsize, d_size, downsize)
                .mean(axis=(2, 4))
            )
            order_0_res = low_pass_down.reshape(curr_batch_size, -1)

            # ORDER 1: filter with every wavelet, take modulus, smooth, pool.
            bank_1_stack = self._filters_1_stack  # (num_filters, H, W)
            modulus_fft_1 = self._filter_and_modulus(
                b_xp, batch_fft[:, None, :, :], bank_1_stack[None, :, :, :]
            )
            batch_pooled_order_1 = self._smooth_and_pool(
                b_xp,
                modulus_fft_1,
                low_pass_c[None, None, :, :],
                curr_batch_size,
                num_order_1_maps,
                d_size,
                downsize,
            )

            # ORDER 2: for each order-1 channel, filter again with only the
            # wavelets of strictly lower angular index k
            batch_pooled_order_2 = b_xp.zeros(
                (curr_batch_size, d_size, d_size, num_order_2_maps), dtype=b_xp.float32
            )
            order_2_idx = 0

            for i, key1 in enumerate(order_1_keys):
                children = self._order2_children[key1]
                if not children:
                    continue

                filters_2_stack = self._filters_2_stack_by_key1[key1]  # (n_children, H, W)
                mod_fft_1_single = modulus_fft_1[:, i]  # (batch, H, W)

                modulus_fft_2 = self._filter_and_modulus(
                    b_xp, mod_fft_1_single[:, None, :, :], filters_2_stack[None, :, :, :]
                )
                n_children = len(children)
                down = self._smooth_and_pool(
                    b_xp,
                    modulus_fft_2,
                    low_pass_c[None, None, :, :],
                    curr_batch_size,
                    n_children,
                    d_size,
                    downsize,
                )
                batch_pooled_order_2[..., order_2_idx : order_2_idx + n_children] = down
                order_2_idx += n_children

            return (
                order_0_res,
                batch_pooled_order_1.reshape(curr_batch_size, -1),
                batch_pooled_order_2.reshape(curr_batch_size, -1),
            )

        # Build batches index ranges
        batches = [
            data[start : min(start + batch_size, num_samples)]
            for start in range(0, num_samples, batch_size)
        ]

        # Execution Selection
        if self.backend == "cpu" and use_multiprocessing:
            nb_cpu: int = os.cpu_count() or 1
            n_jobs = max(1, nb_cpu - 1) if os.cpu_count() and nb_cpu > 1 else 1
            results = self._run_multiprocess(batches, _process_batch, n_jobs)
        else:
            results = []
            for batch in tqdm(
                batches, desc=f"Generating Embeddings on {self.backend.upper()}", unit="batch"
            ):
                results.append(_process_batch(batch))

        # Re-assemble outputs from batch results
        order_0_list, order_1_list, order_2_list = zip(*results)

        order_0_features = xp.concatenate(order_0_list, axis=0)
        first_order_features = xp.concatenate(order_1_list, axis=0)
        second_order_features = xp.concatenate(order_2_list, axis=0)

        final_features = xp.concatenate(
            (order_0_features, first_order_features, second_order_features), axis=1
        )
        self.final_features = final_features

        if self.backend == "gpu":
            return np.asarray(final_features.get())

        return np.asarray(final_features)

    def _run_multiprocess(
        self, batches: list[np.ndarray], _process_batch: Any, n_jobs: int
    ) -> list[Any]:
        """Run batches across multiple processes with a progress bar"""
        try:
            # Joblib generator evaluation wrapped inside tqdm
            with tqdm(
                total=len(batches), desc="Generating Embeddings (Multiprocessing)", unit="batch"
            ) as pbar:
                job_results = Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(_process_batch)(b) for b in batches
                )
                results = []
                for res in job_results:
                    results.append(res)
                    pbar.update(1)
        except TypeError:
            # Fallback for older joblib versions without generator support
            results = []
            for res in tqdm(
                Parallel(n_jobs=n_jobs)(delayed(_process_batch)(b) for b in batches),
                desc="Generating Embeddings (Multiprocessing)",
                unit="batch",
            ):
                results.append(res)

        return results

    def save_embeddings(self, path: str) -> None:
        """Save the generated feature embeddings to a compressed .npz file."""
        if self.final_features is None:
            raise ValueError("No embeddings found. Run generate_embeddings() first.")

        m, k, sigma = self.bank.summary(verbose=False)
        os.makedirs(path, exist_ok=True)
        save_path = rf"{path}/embedding_m{m}_k{k}_sigma{sigma}.npz"

        features = self.final_features.get() if self.backend == "gpu" else self.final_features
        np.savez_compressed(save_path, embedding=features)

        logger.info("Embedding successfully saved to '%s'", save_path)

    def visualise_maps(self, image: np.ndarray, downsize: int) -> None:
        """
        Visualises Order 0 and Order 1 scattering maps.

        Args:
            image (np.ndarray): A single 2D image array (height, width).
            downsize (int): Spatial downsampling factor. Set to 1 for no downsampling.
        """

        if image.ndim != 2:
            raise ValueError("Please provide a single 2D image array of shape (height, width).")

        d_size = int(self.size / downsize)

        # Add fake batch dimension to match FFT logic
        batch = image[None, ...]
        batch_fft = np.fft.fftshift(np.fft.fft2(batch, axes=(-2, -1)), axes=(-2, -1))

        # Dictionaries to hold maps for plotting
        maps = {}

        # ORDER 0
        low_pass_coeffs = batch_fft * self.low_pass
        low_pass_spatial = np.real(
            np.fft.ifft2(np.fft.ifftshift(low_pass_coeffs, axes=(-2, -1)), axes=(-2, -1))
        )[0]

        low_pass_down = low_pass_spatial.reshape(d_size, downsize, d_size, downsize).mean(
            axis=(1, 3)
        )

        maps["Order 0 (Low Pass)"] = low_pass_down

        # ORDER 1
        order_1_keys = self.bank_keys[1:]

        for key1 in order_1_keys:
            wavelet_fft = self.bank[key1]

            # Standard scattering cascade
            filtered_fft = batch_fft * wavelet_fft
            spatial_complex = np.fft.ifft2(
                np.fft.ifftshift(filtered_fft, axes=(-2, -1)), axes=(-2, -1)
            )
            modulus_spatial = np.abs(spatial_complex)

            modulus_fft = np.fft.fftshift(
                np.fft.fft2(modulus_spatial, axes=(-2, -1)), axes=(-2, -1)
            )
            filtered_low_pass = modulus_fft * self.low_pass

            smoothed_spatial = np.real(
                np.fft.ifft2(np.fft.ifftshift(filtered_low_pass, axes=(-2, -1)), axes=(-2, -1))
            )[0]

            # Downsample
            smoothed_down = smoothed_spatial.reshape(d_size, downsize, d_size, downsize).mean(
                axis=(1, 3)
            )

            maps[f"Order 1 ({key1})"] = smoothed_down

        # DYNAMIC GRID CALCULATION
        num_maps = len(maps)

        # Find the best exact integer factors
        best_factor = 1
        for i in range(1, int(np.sqrt(num_maps)) + 1):
            if num_maps % i == 0:
                best_factor = i

        rows = best_factor
        cols = num_maps // best_factor

        # Fallback for prime numbers or extremely stretched grids
        # If the aspect ratio is wider than 3:1, use a square-ish grid instead.
        if cols / rows > 3:
            cols = int(np.ceil(np.sqrt(num_maps)))
            rows = int(np.ceil(num_maps / cols))

        # PLOTTING
        _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

        # Flatten the axes array so we can iterate through it easily
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]  # Catches the edge case where num_maps == 1

        for idx, (title, map_data) in enumerate(maps.items()):
            axes[idx].imshow(map_data, cmap="inferno")
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis("off")

        # Turn off the axes for any leftover empty subplots from the prime-number fallback
        for empty_idx in range(num_maps, len(axes)):
            axes[empty_idx].axis("off")

        plt.tight_layout()
        plt.show()
