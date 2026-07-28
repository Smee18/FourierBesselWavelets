import logging

from .generate_bank import FourierBesselWaveletBank
from .scatnet import FourierBesselScatNet


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())

    return logger


__all__ = ["FourierBesselWaveletBank", "FourierBesselScatNet"]
