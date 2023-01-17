import pytest

from cobyqa import OptimizeResult


class TestOptimizeResult:

    def test_simple(self):
        res = OptimizeResult(x=1.0)
        res.y = True
        assert res.x == pytest.approx(1.0)
        assert res.y

        del res.x
        with pytest.raises(AttributeError):
            res.z = res.x

    def test_dir(self):
        res = OptimizeResult(x=1.0, y=2.0)
        assert dir(res) == ["x", "y"]

    def test_repr(self):
        res = OptimizeResult(x=1.0, y=2.0)
        assert repr(res) == "OptimizeResult(x=1.0, y=2.0)"

    def test_str(self):
        res = OptimizeResult()
        assert str(res) == repr(res)

        res.x = 1.0
        res.y = 2.0
        assert str(res) == " x: 1.0\n y: 2.0"
