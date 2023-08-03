import numpy as np
import pytest

from cobyqa.utils import get_arrays_tol


class TestGetArraysTol:

    @pytest.mark.parametrize('n_max', [0, 1, 2, 10, 100])
    @pytest.mark.parametrize('nb_arrays', [1, 2, 10, 100])
    def test_simple(self, n_max, nb_arrays):
        rng = np.random.default_rng(0)
        arrays = (rng.random(rng.integers(n_max + 1)) for _ in range(nb_arrays))
        tol = get_arrays_tol(*arrays)
        assert tol > 0.0
        assert np.isfinite(tol)

    def test_empty(self):
        with pytest.raises(ValueError):
            get_arrays_tol()
