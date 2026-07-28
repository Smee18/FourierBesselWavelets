import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fbscatnet import FourierBesselScatNet, FourierBesselWaveletBank  # type: ignore


@pytest.fixture
def bank():
    """Fixture providing a standard FourierBesselWaveletBank for tests."""
    return FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)


@pytest.fixture
def net(bank):
    """Fixture providing a standard FourierBesselScatNet initialized with the bank."""
    return FourierBesselScatNet(bank=bank)


def test_embeddings_use_every_filter_channel(net):
    data = np.random.rand(2, 16, 16)
    features = net.generate_embeddings(data, downsize=2)
    reshaped = features.reshape(2, 8, 8, -1)
    channel_sums = np.abs(reshaped).sum(axis=(0, 1, 2))
    assert np.all(channel_sums > 0), "Some filter channels are dead — check the pooling loop"


def test_feature_dimension_matches_expected_channel_count(net):
    """Output width should equal (order0 + order1 + order2 channels) * spatial cells."""
    downsize = 2
    d_size = 16 // downsize

    expected_channels = 1 + net.num_order_1_maps + net.num_order_2_maps
    expected_width = expected_channels * d_size * d_size

    data = np.random.rand(3, 16, 16)
    features = net.generate_embeddings(data, downsize=downsize)

    assert features.shape == (3, expected_width)


def test_sequential_and_multiprocessing_agree(net):
    """Multi-core execution should produce (numerically) the same features as sequential."""
    data = np.random.rand(4, 16, 16)

    seq_features = net.generate_embeddings(data, downsize=2, batch_size=2)
    mp_features = net.generate_embeddings(data, downsize=2, batch_size=2, use_multiprocessing=True)

    np.testing.assert_allclose(seq_features, mp_features, rtol=1e-5, atol=1e-6)


def test_batch_size_larger_than_dataset(net):
    """A single oversized batch should still produce correctly-shaped output."""
    data = np.random.rand(3, 16, 16)
    features = net.generate_embeddings(data, downsize=2, batch_size=64)

    assert features.shape[0] == 3


def test_save_embeddings_round_trips(net, tmp_path):
    data = np.random.rand(2, 16, 16)
    features = net.generate_embeddings(data, downsize=2)

    net.save_embeddings(str(tmp_path))

    saved_files = list(tmp_path.glob("embedding_*.npz"))
    assert len(saved_files) == 1

    loaded = np.load(saved_files[0])["embedding"]
    np.testing.assert_allclose(loaded, features)


def test_save_embeddings_without_generate_raises(net):
    with pytest.raises(ValueError):
        net.save_embeddings("some/path")


def test_gpu_backend_without_cupy_raises(monkeypatch, bank):
    import fbscatnet.scatnet as scatnet_module

    monkeypatch.setattr(scatnet_module, "HAS_CUPY", False)

    with pytest.raises(ImportError):
        FourierBesselScatNet(bank=bank, backend="gpu")


def test_order2_children_exclude_self_and_respect_k_ordering(net):
    for key1, children in net._order2_children.items():
        k1 = net._k_map[key1]
        assert key1 not in children
        assert all(net._k_map[key2] < k1 for key2 in children)


def test_visualise_maps_runs_without_error(monkeypatch, net):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    image = np.random.rand(16, 16)

    net.visualise_maps(image, downsize=2)


def test_generate_embeddings_invalid_downsize():
    net = FourierBesselScatNet(bank=FourierBesselWaveletBank(16, 2, 2), backend="cpu")

    size = net.size
    dummy_data = np.random.rand(2, size, size)
    invalid_downsize = 3 if size % 3 != 0 else 5
    with pytest.raises(ValueError) as exc_info:
        net.generate_embeddings(dummy_data, downsize=invalid_downsize)

    assert f"exactly divisible by downsize ({invalid_downsize})" in str(exc_info.value)
