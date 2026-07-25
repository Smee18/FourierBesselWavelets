import numpy as np
import pytest

from fbscatnet import FourierBesselWaveletBank  # type: ignore


def test_bank_is_subscriptable_and_sized():
    bank = FourierBesselWaveletBank(size=32, m=2, k=2, sigma=0.1)
    assert len(bank) == len(list(bank.get_keys()))
    key = next(iter(bank.get_keys()))
    assert bank[key].shape == (32, 32)


def test_bank_rejects_k_greater_than_m():
    with pytest.raises(ValueError):
        FourierBesselWaveletBank(size=16, m=1, k=2, sigma=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 16, "m": -1, "k": 0, "sigma": 0.1},
        {"size": 16, "m": 2, "k": -1, "sigma": 0.1},
        {"size": 16, "m": 2, "k": 2, "sigma": -0.1},
        {"size": 0, "m": 2, "k": 2, "sigma": 0.1},
        {"size": -4, "m": 2, "k": 2, "sigma": 0.1},
    ],
)
def test_bank_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FourierBesselWaveletBank(**kwargs)


def test_key_naming_matches_expected_format():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    for (m_val, k_val), key in bank.mk_to_key.items():
        assert key == f"m_{m_val}_k_{k_val}_s{bank.sigma}"
        assert key in bank.wavelet_bank


def test_getitem_string_and_tuple_access_agree():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    for (m_val, k_val), key in bank.mk_to_key.items():
        np.testing.assert_array_equal(bank[key], bank[m_val, k_val])


def test_getitem_low_pass_via_zero_zero_index():
    """bank[0, 0] is used elsewhere (e.g. FourierBesselScatNet) as the low-pass filter."""
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    low_pass = bank[0, 0]
    assert low_pass.shape == (16, 16)
    assert bank.mk_to_key[(0, 0)] == f"m_0_k_0_s{bank.sigma}"


def test_getitem_missing_string_key_raises_keyerror():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    with pytest.raises(KeyError):
        bank["not_a_real_key"]


def test_getitem_missing_mk_pair_raises_keyerror():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    with pytest.raises(KeyError):
        bank[1, 0]


def test_getitem_invalid_type_raises_typeerror():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    with pytest.raises(TypeError):
        bank[42]  # type: ignore
    with pytest.raises(TypeError):
        bank[(0, 0, 0)]


def test_get_keys_and_get_values_are_consistent():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    keys = list(bank.get_keys())
    values = list(bank.get_values())
    assert len(keys) == len(values)
    for key, value in zip(keys, values):
        np.testing.assert_array_equal(bank[key], value)


def test_summary_returns_m_k_sigma_regardless_of_verbosity():
    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    assert bank.summary(verbose=False) == (2, 2, 0.1)
    assert bank.summary(verbose=True) == (2, 2, 0.1)


def test_plot_bank_runs_without_error(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    bank = FourierBesselWaveletBank(size=16, m=2, k=2, sigma=0.1)
    bank.plot_bank()
