import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_raises

from cobyqa.linalg import qr
from cobyqa.tests import assert_array_less_equal, assert_dtype_equal


class TestQR:

    @staticmethod
    def geqrf_test(n, m, check_finite):
        rng = np.random.default_rng(n + m)
        a = rng.uniform(-1e2, 1e2, (n, m))
        q, r = qr(a, check_finite=check_finite)
        assert_dtype_equal(r, a)
        assert_dtype_equal(q, a)
        assert_(q.shape == (n, n))
        assert_(r.shape == (n, m))
        assert_allclose(np.matmul(q.T, q), np.eye(n), atol=1e-11)
        assert_allclose(np.triu(r), r, atol=1e-11)
        assert_allclose(np.matmul(q, r), a, atol=1e-11)

    @staticmethod
    def geqp3_test(n, m, check_finite):
        rng = np.random.default_rng(n + m)
        a = rng.uniform(-1e2, 1e2, (n, m))
        q, r, p = qr(a, pivoting=True, check_finite=check_finite)
        assert_dtype_equal(r, a)
        assert_dtype_equal(q, a)
        assert_dtype_equal(p, np.int32)
        assert_(q.shape == (n, n))
        assert_(r.shape == (n, m))
        assert_(p.shape == (m,))
        assert_allclose(np.matmul(q.T, q), np.eye(n), atol=1e-11)
        assert_allclose(np.triu(r), r, atol=1e-8)
        assert_allclose(np.matmul(q, r), a[:, p], atol=1e-11)
        for i in range(min(n, m) - 1):
            norm_columns = np.linalg.norm(r[i:, i + 1:], axis=0)
            assert_array_less_equal(norm_columns, abs(r[i, i]))
        q2, r2 = qr(a[:, p], check_finite=check_finite)
        assert_allclose(q, q2, atol=1e-11)
        assert_allclose(r, r2, atol=1e-11)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n', [1, 10, 100, 500])
    def test_simple(self, n, check_finite):
        self.geqrf_test(n, n, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n', [1, 10, 100, 500])
    def test_simple_pivoting(self, n, check_finite):
        self.geqp3_test(n, n, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n,m', [(1, 4), (10, 23), (100, 250), (500, 1500)])
    def test_simple_trap(self, n, m, check_finite):
        self.geqrf_test(n, m, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n,m', [(1, 4), (10, 23), (100, 250), (500, 1500)])
    def test_simple_trap_pivoting(self, n, m, check_finite):
        self.geqp3_test(n, m, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n,m', [(4, 1), (23, 10), (250, 100), (1500, 500)])
    def test_simple_tall(self, n, m, check_finite):
        self.geqrf_test(n, m, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('n,m', [(4, 1), (23, 10), (250, 100), (1500, 500)])
    def test_simple_tall_pivoting(self, n, m, check_finite):
        self.geqp3_test(n, m, check_finite)

    def test_exceptions(self):
        assert_raises(AssertionError, qr, np.ones(5))
        assert_raises(AssertionError, qr, np.ones((5, 6, 7)))
