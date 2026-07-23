import numpy as np

from fbscatnet import FourierBesselScatNet, FourierBesselWaveletBank  # type: ignore


def test_embeddings_use_every_filter_channel():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    net = FourierBesselScatNet(size=16, bank=bank)
    data = np.random.rand(2, 16, 16)
    features = net.generate_embeddings(data, downsize=2)
    reshaped = features.reshape(2, 8, 8, -1)
    channel_sums = np.abs(reshaped).sum(axis=(0, 1, 2))
    assert np.all(channel_sums > 0), "some filter channels are dead — check the pooling loop"
