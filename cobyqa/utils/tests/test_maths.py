import numpy as np
import pytest

from cobyqa.utils import huge, max_abs_arrays


class TestHuge:

    @pytest.mark.parametrize('dtype', [float, np.double, np.float64])
    def test_simple(self, dtype):
        huge_value = huge(dtype)
        assert 0.0 < huge_value < np.finfo(dtype).max


class TestMaxAbsArrays:

    def test_simple(self):
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([-1.5, 1.5, 2.5])
        assert max_abs_arrays(array1, array2) == pytest.approx(3.0)
        assert max_abs_arrays(array1, array2, initial=4.0) == pytest.approx(4.0)

    def test_some_inf(self):
        array1 = np.array([1.0, 2.0, np.inf])
        array2 = np.array([-1.5, 1.5, 2.5])
        assert max_abs_arrays(array1, array2) == pytest.approx(2.5)

    def test_all_inf(self):
        array1 = np.full(3, -np.inf)
        array2 = np.full(3, np.inf)
        assert max_abs_arrays(array1, array2) == pytest.approx(1.0)
