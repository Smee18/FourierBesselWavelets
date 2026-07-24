import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .generate_bank import FourierBesselWaveletBank


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

    def __init__(self, bank: FourierBesselWaveletBank) -> None:
        """Initialise the FourierBesselScatNet.

        Args:
            bank (FourierBesselWaveletBank): A configured bank of Fourier-Bessel wavelets.
        """
        self.size = bank[0, 0].shape[0]
        self.bank = bank
        self.num_filters = len(bank)
        self.bank_keys = list(bank.get_keys())
        self.low_pass = bank[0, 0]
        self.final_features: np.ndarray | None = None

    def generate_embeddings(
        self, data: np.ndarray, downsize: int, batch_size: int = 32
    ) -> np.ndarray:
        """Generate scattering network feature embeddings for a given dataset.

        Args:
            data (np.ndarray): Input image dataset of shape (num_samples, height, width).
            downsize (int): Spatial downsampling factor via block mean pooling.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.

        Returns:
            np.ndarray: Flattened feature embeddings of shape (num_samples, feature_dim).
        """

        num_samples = data.shape[0]
        d_size = int(self.size / downsize)

        # Helper function to safely extract k from
        def get_k(key_str: str) -> int:
            match = re.search(r"k[=_ ]*(\d+)", key_str)
            return int(match.group(1)) if match else 0

        # Determine memory allocations
        order_1_keys = self.bank_keys[1:]
        num_order_1_maps = len(order_1_keys)

        # Order 2 valid paths
        valid_order_2_paths = []
        for key1 in order_1_keys:
            k1 = get_k(key1)
            for key2 in order_1_keys:
                k2 = get_k(key2)
                if k2 < k1:
                    valid_order_2_paths.append((key1, key2))

        num_order_2_maps = len(valid_order_2_paths)

        # Memory allocation per order
        order_0_features = np.zeros((num_samples, d_size * d_size), dtype=np.float32)
        first_order_features = np.zeros(
            (num_samples, d_size * d_size * num_order_1_maps), dtype=np.float32
        )
        second_order_features = np.zeros(
            (num_samples, d_size * d_size * num_order_2_maps), dtype=np.float32
        )

        for start in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            end = min(start + batch_size, num_samples)
            batch = data[start:end]

            # Convert batch to frequency domain
            batch_fft = np.fft.fftshift(np.fft.fft2(batch, axes=(-2, -1)), axes=(-2, -1))

            # ==============================
            # ORDER 0
            # ==============================
            low_pass_coeffs = batch_fft * self.low_pass
            low_pass_shifted = np.fft.ifftshift(low_pass_coeffs, axes=(-2, -1))
            low_pass_spatial = np.fft.ifft2(low_pass_shifted, axes=(-2, -1))

            low_pass_down = (
                np.real(low_pass_spatial)
                .reshape(-1, d_size, downsize, d_size, downsize)
                .mean(axis=(2, 4))
            )
            order_0_features[start:end] = low_pass_down.reshape(end - start, -1)

            # ==============================
            # ORDER 1
            # ==============================
            batch_pooled_order_1 = np.zeros(
                (end - start, d_size, d_size, num_order_1_maps), dtype=np.float32
            )
            batch_pooled_order_2 = np.zeros(
                (end - start, d_size, d_size, num_order_2_maps), dtype=np.float32
            )

            # Flat counter for Order 2 to avoid array index out-of-bounds
            order_2_idx = 0

            for i, key1 in enumerate(order_1_keys):
                wavelet_fft_1 = self.bank[key1]
                k1 = get_k(key1)

                # Convolution in frequency domain
                filtered_fft_1 = batch_fft * wavelet_fft_1

                # IFFT to bring to SPATIAL domain
                shifted_freq_1 = np.fft.ifftshift(filtered_fft_1, axes=(-2, -1))
                spatial_complex_1 = np.fft.ifft2(shifted_freq_1, axes=(-2, -1))

                # Extract SPATIAL amplitude envelope
                modulus_spatial_1 = np.abs(spatial_complex_1)

                # FFT back to frequency domain
                modulus_fft_1 = np.fft.fftshift(
                    np.fft.fft2(modulus_spatial_1, axes=(-2, -1)), axes=(-2, -1)
                )

                # Low-Pass to get invariant Order 1 features
                filtered_low_pass_1 = modulus_fft_1 * self.low_pass
                smoothed_shifted_1 = np.fft.ifftshift(filtered_low_pass_1, axes=(-2, -1))
                smoothed_spatial_1 = np.fft.ifft2(smoothed_shifted_1, axes=(-2, -1))

                batch_pooled_order_1[..., i] = (
                    np.real(smoothed_spatial_1)
                    .reshape(-1, d_size, downsize, d_size, downsize)
                    .mean(axis=(2, 4))
                )

                # ==============================
                # ORDER 2
                # ==============================
                for key2 in order_1_keys:
                    k2 = get_k(key2)

                    # Only cascade downward in frequency
                    if k2 < k1:
                        wavelet_fft_2 = self.bank[key2]

                        # Convolve Order 1 modulus with the Order 2 wavelet
                        filtered_fft_2 = modulus_fft_1 * wavelet_fft_2

                        shifted_freq_2 = np.fft.ifftshift(filtered_fft_2, axes=(-2, -1))
                        spatial_complex_2 = np.fft.ifft2(shifted_freq_2, axes=(-2, -1))

                        # Spatial amplitude envelope
                        modulus_spatial_2 = np.abs(spatial_complex_2)

                        # FFT back and apply low-pass to get invariant Order 2 features =
                        modulus_fft_2 = np.fft.fftshift(
                            np.fft.fft2(modulus_spatial_2, axes=(-2, -1)), axes=(-2, -1)
                        )
                        filtered_low_pass_2 = modulus_fft_2 * self.low_pass

                        smoothed_shifted_2 = np.fft.ifftshift(filtered_low_pass_2, axes=(-2, -1))
                        smoothed_spatial_2 = np.fft.ifft2(smoothed_shifted_2, axes=(-2, -1))

                        batch_pooled_order_2[..., order_2_idx] = (
                            np.real(smoothed_spatial_2)
                            .reshape(-1, d_size, downsize, d_size, downsize)
                            .mean(axis=(2, 4))
                        )
                        order_2_idx += 1

            # Store the current batch in the global arrays
            first_order_features[start:end] = batch_pooled_order_1.reshape(end - start, -1)
            second_order_features[start:end] = batch_pooled_order_2.reshape(end - start, -1)

        # Concatenate all orders into final embeddings
        final_features = np.concatenate(
            (order_0_features, first_order_features, second_order_features), axis=1
        )
        self.final_features = final_features

        return final_features

    def save_embeddings(self) -> None:
        """Save the generated feature embeddings to a compressed .npz file."""
        if self.final_features is None:
            raise ValueError("No embeddings found. Run generate_embeddings() first.")

        m, k, sigma = self.bank.summary(verbose=False)
        os.makedirs("features", exist_ok=True)
        save_path = rf"features/embedding_m{m}_k{k}_sigma{sigma}.npz"

        np.savez_compressed(save_path, embedding=self.final_features)

        print(f"Embedding successfully saved to '{save_path}'")

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

        # ==============================
        # ORDER 0
        # ==============================
        low_pass_coeffs = batch_fft * self.low_pass
        low_pass_spatial = np.real(
            np.fft.ifft2(np.fft.ifftshift(low_pass_coeffs, axes=(-2, -1)), axes=(-2, -1))
        )[0]

        low_pass_down = low_pass_spatial.reshape(d_size, downsize, d_size, downsize).mean(
            axis=(1, 3)
        )

        maps["Order 0 (Low Pass)"] = low_pass_down

        # ==============================
        # ORDER 1
        # ==============================
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

        # ==============================
        # DYNAMIC GRID CALCULATION
        # ==============================
        num_maps = len(maps)

        # 1. Find the best exact integer factors
        best_factor = 1
        for i in range(1, int(np.sqrt(num_maps)) + 1):
            if num_maps % i == 0:
                best_factor = i

        rows = best_factor
        cols = num_maps // best_factor

        # 2. Fallback for prime numbers or extremely stretched grids
        # If the aspect ratio is wider than 3:1, use a square-ish grid instead.
        if cols / rows > 3:
            cols = int(np.ceil(np.sqrt(num_maps)))
            rows = int(np.ceil(num_maps / cols))

        # ==============================
        # PLOTTING
        # ==============================
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

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
