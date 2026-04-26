import numpy as np
import pytest

_SEED: int = 42


@pytest.fixture(autouse=True)
def set_seed() -> None:
    np.random.seed(_SEED)
