"""Cross-family tests for ScatNet and wavelet-bank integration."""

import numpy as np
import pytest

from fbscatnet import FourierBesselWaveletBank, ScatNet, SolidHarmonicsWaveletBank


@pytest.fixture(params=["fb", "sh"])
def bank(request):
    if request.param == "fb":
        return FourierBesselWaveletBank(size=16, m=2, k=3)
    return SolidHarmonicsWaveletBank(size=16, j=3, l=2)


@pytest.fixture
def net(bank):
    return ScatNet(bank=bank, verbose=False)


def test_embedding_shape_and_finiteness(net):
    data = np.random.default_rng(0).random((3, 16, 16), dtype=np.float32)
    features = net.generate_embeddings(data, downsize=2, batch_size=2)
    expected_channels = 1 + net.num_order_1_maps + net.num_order_2_maps
    assert features.shape == (3, expected_channels * 8 * 8)
    assert np.isfinite(features).all()


def test_batch_size_larger_than_dataset(net):
    data = np.random.default_rng(1).random((2, 16, 16), dtype=np.float32)
    features = net.generate_embeddings(data, downsize=2, batch_size=64)
    assert features.shape[0] == 2


def test_save_embeddings_round_trip(net, tmp_path):
    data = np.random.default_rng(2).random((2, 16, 16), dtype=np.float32)
    expected = net.generate_embeddings(data, downsize=2)
    net.save_embeddings(str(tmp_path))
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    np.testing.assert_allclose(np.load(files[0])["embedding"], expected)


@pytest.mark.parametrize(
    "data,downsize,batch_size",
    [
        (np.zeros((2, 16, 16)), 0, 2),
        (np.zeros((2, 16, 16)), -1, 2),
        (np.zeros((2, 16, 16)), 2, 0),
        (np.zeros((0, 16, 16)), 2, 2),
        (np.zeros((2, 16)), 2, 2),
        (np.zeros((2, 8, 8)), 2, 2),
    ],
)
def test_invalid_embedding_inputs_raise(net, data, downsize, batch_size):
    with pytest.raises(ValueError):
        net.generate_embeddings(data, downsize=downsize, batch_size=batch_size)


def test_backend_validation(bank):
    with pytest.raises(ValueError, match="backend"):
        ScatNet(bank, backend="cuda")


def test_second_order_paths_follow_family_ordering(net, bank):
    index_by_key = {key: index for index, key in bank.index_to_key.items()}
    for parent, children in net._order2_children.items():
        parent_index = index_by_key[parent]
        for child in children:
            child_index = index_by_key[child]
            if isinstance(bank, FourierBesselWaveletBank):
                assert child_index[1] < parent_index[1]
            else:
                assert child_index[0] > parent_index[0]


def test_solid_harmonics_embedding_filename_uses_bank_parameters(tmp_path):
    bank = SolidHarmonicsWaveletBank(size=16, j=2, l=1)
    net = ScatNet(bank, verbose=False)
    net.generate_embeddings(np.ones((1, 16, 16)), downsize=2)
    net.save_embeddings(str(tmp_path))
    assert len(list(tmp_path.glob("SH_embedding_j2_l1.npz"))) == 1
