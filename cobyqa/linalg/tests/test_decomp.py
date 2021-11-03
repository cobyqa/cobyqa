import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_equal, \
    assert_raises

from cobyqa.linalg import qr
from cobyqa.tests import assert_array_less_equal, assert_dtype_equal


class TestQR:

    @staticmethod
    def geqrf_test(n, m, overwrite_a, check_finite):
        rng = np.random.default_rng(n + m)
        a = rng.uniform(-1e10, 1e10, (n, m))
        a_copy = a
        if overwrite_a:
            a_copy = np.copy(a)
        q, r = qr(a, overwrite_a, check_finite=check_finite)  # noqa
        assert_dtype_equal(r, a)
        assert_dtype_equal(q, a)
        assert_(q.shape == (n, n))
        assert_(r.shape == (n, m))
        assert_allclose(np.matmul(q.T, q), np.eye(n), atol=1e-11)
        assert_allclose(np.triu(r), r, atol=1e-11)
        assert_allclose(np.matmul(q, r), a_copy, atol=1e-11)
        if overwrite_a:
            assert_array_equal(a, r)

    @staticmethod
    def geqp3_test(n, m, overwrite_a, check_finite):
        rng = np.random.default_rng(n + m)
        a = rng.uniform(-1e10, 1e10, (n, m))
        a_copy = a
        if overwrite_a:
            a_copy = np.copy(a)
        q, r, p = qr(a, overwrite_a, True, check_finite)
        assert_dtype_equal(r, a)
        assert_dtype_equal(q, a)
        assert_dtype_equal(p, int)
        assert_(q.shape == (n, n))
        assert_(r.shape == (n, m))
        assert_(p.shape == (m,))
        assert_allclose(np.matmul(q.T, q), np.eye(n), atol=1e-11)
        assert_allclose(np.triu(r), r, atol=1e-8)
        assert_allclose(np.matmul(q, r), a_copy[:, p], atol=1e-11)
        for i in range(min(n, m) - 1):
            norm_columns = np.linalg.norm(r[i:, i + 1:], axis=0)
            assert_array_less_equal(norm_columns, abs(r[i, i]))
        if overwrite_a:
            assert_array_equal(a, r)
        q2, r2 = qr(a_copy[:, p], check_finite=check_finite)  # noqa
        assert_allclose(q, q2, atol=1e-11)
        assert_allclose(r, r2, atol=1e-11)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n', [1, 10, 100, 500])
    def test_simple(self, n, overwrite_a, check_finite):
        self.geqrf_test(n, n, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n', [1, 10, 100, 500])
    def test_simple_pivoting(self, n, overwrite_a, check_finite):
        self.geqp3_test(n, n, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(1, 4), (10, 23), (100, 250), (500, 750)])
    def test_simple_trap(self, n, m, overwrite_a, check_finite):
        self.geqrf_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(1, 4), (10, 23), (100, 250), (500, 750)])
    def test_simple_trap_pivoting(self, n, m, overwrite_a, check_finite):
        self.geqp3_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(4, 1), (23, 10), (250, 100), (750, 500)])
    def test_simple_tall(self, n, m, overwrite_a, check_finite):
        self.geqrf_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(4, 1), (23, 10), (250, 100), (750, 500)])
    def test_simple_tall_pivoting(self, n, m, overwrite_a, check_finite):
        self.geqp3_test(n, m, overwrite_a, check_finite)

    def test_exceptions(self):
        assert_raises(AssertionError, qr, np.ones(5))
        assert_raises(AssertionError, qr, np.ones((5, 6, 7)))
