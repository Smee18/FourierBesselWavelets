# fbscatnet

[![PyPI version](https://img.shields.io/pypi/v/fbscatnet.svg)](https://pypi.org/project/fbscatnet/)
[![Python Versions](https://img.shields.io/pypi/pyversions/fbscatnet.svg)](https://pypi.org/project/fbscatnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Smee18/FourierBesselWavelets/actions/workflows/ci.yml/badge.svg)](https://github.com/Smee18/FourierBesselWavelets/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Smee18/36c71b13aa4288c24a60643049a4a044/raw/coverage.json)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`fbscatnet` is a high-performance Python library for computing **Fourier-Bessel wavelet scattering transforms**. It generates robust, feature embeddings from 2D images, making it an ideal feature extractor for computer vision, biomedical imaging, and physics-based machine learning. This is a brand new project of mine that I have been working on for some time. I would love any criticism, improvements, corrections to both the code and the maths.

## Features

- **Novel Wavelet:** New wavelet relying on Bessel basis functions.
- **Hardware Accelerated:** Seamlessly switch between multi-core CPU execution (`scipy`/`joblib`) and GPU acceleration (`cupy`).
- **Highly Optimized:** Vectorised filtering in the frequency domain for maximum throughput across large image batches.
- **ML-Ready:** Outputs flattened feature arrays directly compatible with `scikit-learn`, `xgboost`, or `pytorch`.

## Installation

Install the base package (CPU-only) via pip:

```bash
pip install fbscatnet
```

To enable **GPU acceleration**, install with the `gpu` extra (requires a CUDA-compatible GPU):

```bash
pip install fbscatnet[gpu]
```

## Quickstart

Extracting features from a dataset takes just a few lines of code:

```python
import numpy as np
import logging
from fbscatnet import FourierBesselWaveletBank, FourierBesselScatNet, logger_config
logger_config.enable_colored_logs(logging.DEBUG) # IMPORTANT: SET TO SEE LOGS

# 1. Create some dummy image data (e.g., 10 grayscale images of size 64x64)
images = np.random.rand(10, 64, 64)

# 2. Instantiate a Fourier-Bessel Wavelet Bank
# size: spatial dimension (64x64), m: angular order, k: radial roots
bank = FourierBesselWaveletBank(size=64, m=2, k=2, sigma=0.1)

# 3. Initialize the Scattering Network
# Use backend="gpu" if you installed with CuPy
net = FourierBesselScatNet(bank=bank, backend="cpu")

# 4. Generate feature embeddings
# downsize: spatial pooling factor (must evenly divide the image size)
features = net.generate_embeddings(images, downsize=4, use_multiprocessing=True)

print(f"Generated embeddings shape: {features.shape}")
# Output: (10, feature_dimension)
```

## Visualising Wavelet Maps

If you want to inspect how the wavelet filters are interacting with your image at the first scattering order, `fbscatnet` includes a built-in plotting tool:

```python
# Pass a single 2D image to visualize
single_image = images[0]
net.visualise_maps(single_image, downsize=4)
```

## API Overview

Sphinx documentation at: https://smee18.github.io/FourierBesselWavelets/

The `example` folder also contains a full pipeline, classifying MNIST using the library
The pdf of my notes taken along this project contains an in-depth explanation of the mathematics behind these wavelets. Here you can find all derivations, proofs and useful information. Some stuff might seem trivial but my goal is to really expose all aspects so anyone new to wavelet theory can understand the mathematics behind these Fourier Bessel wavelets.

## License
This project is licensed under the MIT License.