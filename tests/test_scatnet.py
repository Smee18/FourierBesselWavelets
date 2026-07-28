import numpy as np
import pytest
from fbscatnet import FourierBesselScatNet, FourierBesselWaveletBank  # type: ignore


def test_embeddings_use_every_filter_channel():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    data = np.random.rand(2, 16, 16)
    features = net.generate_embeddings(data, downsize=2)
    reshaped = features.reshape(2, 8, 8, -1)
    channel_sums = np.abs(reshaped).sum(axis=(0, 1, 2))
    assert np.all(channel_sums > 0), "Some filter channels are dead — check the pooling loop"


def test_feature_dimension_matches_expected_channel_count():
    """Output width should equal (order0 + order1 + order2 channels) * spatial cells."""
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    downsize = 2
    d_size = 16 // downsize

    expected_channels = 1 + net.num_order_1_maps + net.num_order_2_maps
    expected_width = expected_channels * d_size * d_size

    data = np.random.rand(3, 16, 16)
    features = net.generate_embeddings(data, downsize=downsize)

    assert features.shape == (3, expected_width)


def test_sequential_and_multiprocessing_agree():
    """Multi-core execution should produce (numerically) the same features as sequential."""
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    data = np.random.rand(4, 16, 16)

    seq_features = net.generate_embeddings(data, downsize=2, batch_size=2)
    mp_features = net.generate_embeddings(data, downsize=2, batch_size=2, use_multiprocessing=True)

    np.testing.assert_allclose(seq_features, mp_features, rtol=1e-5, atol=1e-6)


def test_batch_size_larger_than_dataset():
    """A single oversized batch should still produce correctly-shaped output."""
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    data = np.random.rand(3, 16, 16)

    features = net.generate_embeddings(data, downsize=2, batch_size=64)

    assert features.shape[0] == 3


def test_save_embeddings_round_trips(tmp_path):
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    data = np.random.rand(2, 16, 16)
    features = net.generate_embeddings(data, downsize=2)

    net.save_embeddings(str(tmp_path))

    saved_files = list(tmp_path.glob("embedding_*.npz"))
    assert len(saved_files) == 1

    loaded = np.load(saved_files[0])["embedding"]
    np.testing.assert_allclose(loaded, features)


def test_save_embeddings_without_generate_raises():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)

    with pytest.raises(ValueError):
        net.save_embeddings("some/path")


def test_gpu_backend_without_cupy_raises(monkeypatch):
    import fbscatnet.scatnet as scatnet_module

    monkeypatch.setattr(scatnet_module, "HAS_CUPY", False)
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)

    with pytest.raises(ImportError):
        FourierBesselScatNet(bank=bank, backend="gpu")


def test_order2_children_exclude_self_and_respect_k_ordering():
    """Order-2 pairing should never pair a filter with itself, and only pairs
    where the second filter's k is strictly less than the first's."""
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)

    for key1, children in net._order2_children.items():
        k1 = net._k_map[key1]
        assert key1 not in children
        assert all(net._k_map[key2] < k1 for key2 in children)


def test_visualise_maps_runs_without_error(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(bank=bank)
    image = np.random.rand(16, 16)

    net.visualise_maps(image, downsize=2)
