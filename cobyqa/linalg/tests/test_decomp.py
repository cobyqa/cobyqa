import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_equal, \
    assert_raises

from cobyqa.linalg import givens, qr
from cobyqa.tests import assert_array_less_equal, assert_dtype_equal


class TestGivens:

    @staticmethod
    def givens_test(n, axis):
        rng = np.random.default_rng(n)
        tiny = np.finfo(float).tiny
        m = rng.uniform(-1e3, 1e3, (n, n))
        cval, sval = rng.random(), rng.random()
        i, j = rng.choice(n, 2, replace=False)
        hypot = np.hypot(cval, sval)
        gr = np.eye(n, dtype=float)
        if hypot > tiny * max(abs(cval), abs(sval)):
            gr[i, [i, j]] = cval / hypot, -sval / hypot
            gr[j, [i, j]] = sval / hypot, cval / hypot
            if axis == 0:
                mr = np.matmul(gr, m)
            else:
                mr = np.matmul(m, gr.T)
            hval = givens(m, cval, sval, i, j, axis)
            assert_allclose(m, mr, atol=1e-11)
            assert_allclose(hval, hypot)

    @pytest.mark.parametrize('n', [2, 5, 10, 100])
    def test_simple(self, n):
        self.givens_test(n, 0)
        self.givens_test(n, 1)


class TestQR:

    @staticmethod
    def geqrf_test(n, m, overwrite_a, check_finite):
        rng = np.random.default_rng(n + m)
        a = rng.uniform(-1e3, 1e3, (n, m))
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
        a = rng.uniform(-1e3, 1e3, (n, m))
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
            assert_array_less_equal(norm_columns, r[i, i])
        if overwrite_a:
            assert_array_equal(a, r)
        q2, r2 = qr(a_copy[:, p], check_finite=check_finite)  # noqa
        assert_allclose(q, q2, atol=1e-11)
        assert_allclose(r, r2, atol=1e-11)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n, overwrite_a, check_finite):
        self.geqrf_test(n, n, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple_pivoting(self, n, overwrite_a, check_finite):
        self.geqp3_test(n, n, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(1, 4), (5, 12), (10, 23), (100, 250)])
    def test_simple_trap(self, n, m, overwrite_a, check_finite):
        self.geqrf_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(1, 4), (5, 12), (10, 23), (100, 250)])
    def test_simple_trap_pivoting(self, n, m, overwrite_a, check_finite):
        self.geqp3_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(4, 1), (12, 5), (23, 10), (250, 100)])
    def test_simple_tall(self, n, m, overwrite_a, check_finite):
        self.geqrf_test(n, m, overwrite_a, check_finite)

    @pytest.mark.parametrize('check_finite', [True, False])
    @pytest.mark.parametrize('overwrite_a', [False, True])
    @pytest.mark.parametrize('n,m', [(4, 1), (12, 5), (23, 10), (250, 100)])
    def test_simple_tall_pivoting(self, n, m, overwrite_a, check_finite):
        self.geqp3_test(n, m, overwrite_a, check_finite)

    def test_exceptions(self):
        assert_raises(AssertionError, qr, np.ones(5))
        assert_raises(AssertionError, qr, np.ones((5, 6, 7)))
