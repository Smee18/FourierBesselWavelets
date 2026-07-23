import os

import numpy as np
from tqdm import tqdm


class FourierBesselScatNet:
    def __init__(self, size, bank):
        self.size = size
        self.bank = bank
        self.num_filters = len(bank)
        self.bank_keys = bank.get_keys()
        self.low_pass = bank[0, 0]

    def generate_embeddings(self, data, downsize, batch_size=32):

        num_samples = data.shape[0]
        d_size = int(self.size / downsize)

        # Allocate memory for feature embedding
        final_features = np.zeros(
            (num_samples, (d_size * d_size * self.num_filters)), dtype=np.float32
        )

        for start in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            end = min(start + batch_size, num_samples)

            batch = data[start:end]  # Select batch (batch, 225, 225)
            # Convert image to fourier on final two dimensions and center DC component
            batch_fft = np.fft.fftshift(np.fft.fft2(batch, axes=(-2, -1)), axes=(-2, -1))

            # Allocate memory for pooled batch
            batch_pooled = np.zeros(
                (end - start, d_size, d_size, self.num_filters), dtype=np.float32
            )

            self.bank_keys = list(self.bank.get_keys())
            for i, key in enumerate(self.bank_keys[1:]):
                wavelet_fft = self.bank[key]  # Extract wavelet (centered in frequency domain)
                filtered_fft = batch_fft * wavelet_fft  # Convolution in frequency domain

                unshifted_freq = np.fft.ifftshift(filtered_fft, axes=(-2, -1))
                spatial_complex = np.fft.ifft2(unshifted_freq, axes=(-2, -1))
                filtered_spatial = np.abs(spatial_complex) * self.low_pass

                # Downsample via spatial block mean pooling
                batch_pooled[..., i] = filtered_spatial.reshape(
                    -1, d_size, downsize, d_size, downsize
                ).mean(axis=(2, 4))

            final_features[start:end] = batch_pooled.reshape(end - start, -1)

        self.final_features = final_features

        return final_features

    def save_embeddings(self):

        m, k, sigma = self.bank.summary(verbose=False)
        os.makedirs("features", exist_ok=True)
        save_path = rf"features/embedding_m{m}_k{k}_sigma{sigma}.npz"

        np.savez_compressed(save_path, embedding=self.final_features)

        print(f"Embedding successfully saved to '{save_path}'")
