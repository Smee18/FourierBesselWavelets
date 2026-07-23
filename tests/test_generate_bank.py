from fbscatnet import FourierBesselWaveletBank  # type: ignore


def test_bank_is_subscriptable_and_sized():
    bank = FourierBesselWaveletBank(size=32, m=2, k=2, sigma=0.1)
    assert len(bank) == len(list(bank.get_keys()))
    key = next(iter(bank.get_keys()))
    assert bank[key].shape == (32, 32)
