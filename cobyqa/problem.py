from abc import ABC

import numpy as np

from .utils import get_arrays_tol


class Function(ABC):
    """
    Base class for objective and constraints functions.
    """

    def __init__(self, fun, verbose, store_hist, *args):
        """
        Initialize the function.

        Parameters
        ----------
        fun : callable
            Function to evaluate, or None.

                ``fun(x, *args) -> {float, array_like}``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        verbose : bool
            Whether to print the function evaluations.
        store_hist : bool
            Whether to store the function evaluations.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        self._fun = fun
        self._verbose = verbose
        self._store_hist = store_hist
        self._args = args
        self._n_eval = 0
        self._f_hist = []
        self._x_hist = []
        self._barrier = 2.0 ** min(100, np.finfo(float).maxexp // 2, -np.finfo(float).minexp // 2)

    def __call__(self, x):
        """
        Evaluate the function.

        This method also applies the barrier function to the function value.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the function is evaluated.

        Returns
        -------
        {float, numpy.ndarray}
            Function value at `x`.
        """
        if self._fun is None:
            return self.apply_barrier()
        x = np.array(x, dtype=float)
        val = self._fun(x, *self._args)
        val = self.apply_barrier(val)
        self._n_eval += 1
        if self._verbose:
            print(f'{self.name}({x}) = {val}')
        if self._store_hist:
            self.f_hist.append(val)
            self.x_hist.append(x)
        return val

    @property
    def name(self):
        """
        Name of the function.

        Returns
        -------
        str
            Name of the function.
        """
        return self._fun.__name__

    @property
    def n_eval(self):
        """
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        return self._n_eval

    @property
    def f_hist(self):
        """
        History of function evaluations.

        This property returns an empty list if the history of function
        evaluations is not maintained.

        Returns
        -------
        list
            History of function evaluations.
        """
        return self._f_hist

    @property
    def x_hist(self):
        """
        History of variables.

        This property returns an empty list if the history of function
        evaluations is not maintained.

        Returns
        -------
        list
            History of variables.
        """
        return self._x_hist

    def apply_barrier(self, val=None):
        """
        Apply the barrier function to the function value.

        This method must be implemented in the derived classes. If `val` is
        None, the method must return the default value of the function.

        Parameters
        ----------
        val : {float, array_like}, optional
            Function value to which the barrier function is to be applied.

        Returns
        -------
        {float, numpy.ndarray}
            Function value with the barrier function applied.
        """
        raise NotImplementedError


class Constraints(ABC):
    """
    Base class for constraints.
    """

    @property
    def m(self):
        """
        Number of constraints.

        This method must be implemented in the derived classes.

        Returns
        -------
        int
            Number of constraints.
        """
        raise NotImplementedError

    def resid(self, x, *args):
        """
        Evaluate the constraint residuals.

        This method must be implemented in the derived classes.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint residuals are evaluated.
        *args : tuple
            Additional arguments to be passed to the function.

        Returns
        -------
        float
            Constraint residuals at `x`.
        """
        raise NotImplementedError

    def project(self, x):
        """
        Project a point onto the feasible set.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point to be projected.

        Returns
        -------
        numpy.ndarray
            Projection of `x` onto the feasible set.

        Raises
        ------
        NotImplementedError
            If the constraints do not implement the projection.
        """
        raise NotImplementedError


class ObjectiveFunction(Function):
    """
    Real-valued objective function.
    """

    def __init__(self, fun, verbose, store_hist, *args):
        """
        Initialize the objective function.

        Parameters
        ----------
        fun : callable
            Function to evaluate, or None.

                ``fun(x, *args) -> float``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        verbose : bool
            Whether to print the function evaluations.
        store_hist : bool
            Whether to store the function evaluations.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        super().__init__(fun, verbose, store_hist, *args)

    def apply_barrier(self, val=None):
        """
        Apply the barrier function to the function value.

        If `val` is None, the method returns zero.

        Parameters
        ----------
        val : float, optional
            Function value to which the barrier function is to be applied.

        Returns
        -------
        float
            Function value with the barrier function applied.
        """
        if val is None:
            val = 0.0
        val = float(val)
        if np.isnan(val):
            val = self._barrier
        return min(val, self._barrier)


class BoundConstraints(Constraints):
    """
    Bound constraints ``xl <= x <= xu``.
    """

    def __init__(self, xl, xu):
        """
        Initialize the bound constraints.

        Parameters
        ----------
        xl : array_like, shape (n,)
            Lower bound.
        xu : array_like, shape (n,)
            Upper bound.
        """
        self._xl = _1d_array(xl, 'The lower bound must be a vector.')
        self._xu = _1d_array(xu, 'The upper bound must be a vector.')

        # Check the bounds.
        if self.xl.size != self.xu.size:
            raise ValueError('The bounds must have the same size.')

        # Remove the ill-defined bounds.
        self.xl[np.isnan(self.xl)] = -np.inf
        self.xu[np.isnan(self.xu)] = np.inf

    @property
    def xl(self):
        """
        Lower bound.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Lower bound.
        """
        return self._xl

    @property
    def xu(self):
        """
        Upper bound.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Upper bound.
        """
        return self._xu

    @property
    def m(self):
        """
        Number of bound constraints.

        Returns
        -------
        int
            Number of bound constraints.
        """
        return np.count_nonzero(self.xl > -np.inf) + np.count_nonzero(self.xu < np.inf)

    @property
    def is_feasible(self):
        """
        Whether the bound constraints are feasible.

        Returns
        -------
        bool
            Whether the bound constraints are feasible.
        """
        return np.all(self.xl <= self.xu) and np.all(self.xl < np.inf) and np.all(self.xu > -np.inf)

    def resid(self, x, *args):
        """
        Evaluate the constraint residuals.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint residuals are evaluated.
        *args : tuple
            This argument is ignored.

        Returns
        -------
        float
            Constraint residuals at `x`.
        """
        x = np.asarray(x, dtype=float)
        val = np.max(self.xl - x, initial=0.0)
        return np.max(x - self.xu, initial=val)

    def project(self, x):
        """
        Project a point onto the feasible set.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point to be projected.

        Returns
        -------
        numpy.ndarray
            Projection of `x` onto the feasible set.
        """
        return np.minimum(np.maximum(x, self.xl), self.xu) if self.is_feasible else x


class LinearConstraints(Constraints):
    """
    Linear constraints ``a @ x <= b`` or ``a @ x == b``.
    """

    def __init__(self, a, b, is_equality):
        """
        Initialize the linear constraints.

        Parameters
        ----------
        a : array_like, shape (m, n)
            Left-hand side of the linear constraints.
        b : array_like, shape (m,)
            Right-hand side of the linear constraints.
        is_equality : bool
            Whether the linear constraints are equality constraints. If True,
            the linear constraints are ``a @ x == b``. Otherwise, the linear
            constraints are ``a @ x <= b``.
        """
        c_type = 'equality' if is_equality else 'inequality'
        self._a = _2d_array(a, f'The left-hand side of the linear {c_type} constraints must be a matrix.')
        self._b = _1d_array(b, f'The right-hand side of the linear {c_type} constraints must be a vector.')
        self._is_equality = is_equality

        # Check the constraints.
        if self.a.shape[0] != self.b.size:
            raise ValueError(f'The linear {c_type} constraints are inconsistent.')

        # Remove the ill-defined constraints.
        self.a[np.isnan(self.a)] = 0.0
        undef_c = np.isnan(self.b)
        if not self._is_equality:
            undef_c |= np.isinf(self.b)
        self._a = self.a[~undef_c, :]
        self._b = self.b[~undef_c]

    @property
    def a(self):
        """
        Left-hand side of the linear constraints.

        Returns
        -------
        numpy.ndarray, shape (m, n)
            Left-hand side of the linear constraints.
        """
        return self._a

    @property
    def b(self):
        """
        Right-hand side of the linear constraints.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Right-hand side of the linear constraints.
        """
        return self._b

    @property
    def is_equality(self):
        """
        Whether the linear constraints are equality constraints.

        Returns
        -------
        bool
            Whether the linear constraints are equality constraints.
        """
        return self._is_equality

    @property
    def m(self):
        """
        Number of linear constraints.

        Returns
        -------
        int
            Number of linear constraints.
        """
        return self.b.size

    def resid(self, x, *args):
        """
        Evaluate the constraint residuals.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint residuals are evaluated.
        *args : tuple
            This argument is ignored.

        Returns
        -------
        float
            Constraint residuals at `x`.
        """
        x = np.array(x, dtype=float)
        val = self.a @ x - self.b
        if self.is_equality:
            return np.max(np.abs(val), initial=0.0)
        else:
            return np.max(val, initial=0.0)


class NonlinearConstraints(Function, Constraints):
    """
    Nonlinear constraints ``fun(x) <= 0`` or ``fun(x) == 0``.
    """

    def __init__(self, fun, is_equality, verbose, store_hist, *args):
        """
        Initialize the nonlinear constraints.

        Parameters
        ----------
        fun : callable
            Function to evaluate, or None.

                ``fun(x, *args) -> array_like``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        is_equality : bool
            Whether the nonlinear constraints are equality constraints.
        verbose : bool
            Whether to print the function evaluations.
        store_hist : bool
            Whether to store the function evaluations.
        *args : tuple
            Additional arguments passed to the function.
        """
        super().__init__(fun, verbose, store_hist, *args)
        self._is_equality = is_equality
        self._m = 0 if fun is None else None

    @property
    def is_equality(self):
        """
        Whether the nonlinear constraints are equality constraints.

        Returns
        -------
        bool
            Whether the nonlinear constraints are equality constraints.
        """
        return self._is_equality

    @property
    def m(self):
        """
        Number of nonlinear constraints.

        Returns
        -------
        int
            Number of nonlinear constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear constraints is unknown.
        """
        if self._m is None:
            raise ValueError('The number of nonlinear constraints is unknown.')
        else:
            return self._m

    def apply_barrier(self, val=None):
        """
        Apply the barrier function to the function value.

        If `val` is None, the method returns an empty array.

        Parameters
        ----------
        val : array_like, optional
            Function value to which the barrier function is to be applied.

        Returns
        -------
        numpy.ndarray
            Function value with the barrier function applied.
        """
        if val is None:
            val = np.empty(0)
        val = np.array(val, dtype=float)
        val = np.atleast_1d(val)
        val[np.isnan(val)] = self._barrier
        val = np.minimum(val, self._barrier)
        if self.is_equality:
            val = np.maximum(val, -self._barrier)
        if self._m is None:
            self._m = val.size
        return val

    def resid(self, x, *args):
        """
        Evaluate the constraint residuals.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint residuals are evaluated.
        *args : tuple
            One argument is expected, which is the function value at `x`. If
            this argument is not passed, the constraint function is evaluated.

        Returns
        -------
        float
            Constraint residuals at `x`.
        """
        if len(args) == 1 and args[0] is not None:
            val = np.array(args[0], dtype=float)
        else:
            val = self(x)
        if self.is_equality:
            return np.max(np.abs(val), initial=0.0)
        else:
            return np.max(val, initial=0.0)


class Problem:
    """
    Optimization problem.
    """

    def __init__(self, obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq):
        """
        Initialize the nonlinear problem.

        The problem is preprocessed to remove all the variables that are fixed
        by the bound constraints.

        Parameters
        ----------
        obj : ObjectiveFunction
            Objective function.
        x0 : array_like, shape (n,)
            Initial guess.
        bounds : BoundConstraints
            Bound constraints.
        linear_ub : LinearConstraints
            Linear inequality constraints.
        linear_eq : LinearConstraints
            Linear equality constraints.
        nonlinear_ub : NonlinearConstraints
            Nonlinear inequality constraints.
        nonlinear_eq : NonlinearConstraints
            Nonlinear equality constraints.
        """
        self._obj = obj
        self._nonlinear_ub = nonlinear_ub
        self._nonlinear_eq = nonlinear_eq

        # Check the consistency of the problem.
        x0 = _1d_array(x0, 'The initial guess must be a vector.')
        n = x0.size
        if bounds.xl.size != n:
            raise ValueError(f'The lower bound must have {n} elements.')
        if bounds.xu.size != n:
            raise ValueError(f'The upper bound must have {n} elements.')
        if linear_ub.a.shape[1] != n:
            raise ValueError(f'The left-hand side matrix of the linear inequality constraints must have {n} columns.')
        if linear_eq.a.shape[1] != n:
            raise ValueError(f'The left-hand side matrix of the linear equality constraints must have {n} columns.')

        # Check which variables are fixed.
        tol = get_arrays_tol(bounds.xl, bounds.xu)
        self._fixed_idx = (bounds.xl <= bounds.xu) & (np.abs(bounds.xl - bounds.xu) < tol)
        self._fixed_val = 0.5 * (bounds.xl[self._fixed_idx] + bounds.xu[self._fixed_idx])
        self._fixed_val = np.minimum(np.maximum(self._fixed_val, bounds.xl[self._fixed_idx]), bounds.xu[self._fixed_idx])

        # Set the bound and linear constraints.
        self._bounds = BoundConstraints(bounds.xl[~self._fixed_idx], bounds.xu[~self._fixed_idx])
        self._linear_ub = LinearConstraints(linear_ub.a[:, ~self._fixed_idx], linear_ub.b - linear_ub.a[:, self._fixed_idx] @ self._fixed_val, False)
        self._linear_eq = LinearConstraints(linear_eq.a[:, ~self._fixed_idx], linear_eq.b - linear_eq.a[:, self._fixed_idx] @ self._fixed_val, True)

        # Set the initial guess.
        self._x0 = self._bounds.project(x0[~self._fixed_idx])

    @property
    def n(self):
        """
        Number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self.x0.size

    @property
    def n_orig(self):
        """
        Number of variables in the original problem (with fixed variables).

        Returns
        -------
        int
            Number of variables in the original problem (with fixed variables).
        """
        return self._fixed_idx.size

    @property
    def x0(self):
        """
        Initial guess.

        Returns
        -------
        numpy.ndarray
            Initial guess.
        """
        return self._x0

    @property
    def fun_name(self):
        """
        Name of the objective function.

        Returns
        -------
        str
            Name of the objective function.
        """
        return self._obj.name

    @property
    def n_eval(self):
        """
        Number of objective function evaluations.

        Returns
        -------
        int
            Number of objective function evaluations.
        """
        return self._obj.n_eval

    @property
    def bounds(self):
        """
        Bound constraints.

        Returns
        -------
        BoundConstraints
            Bound constraints.
        """
        return self._bounds

    @property
    def linear_ub(self):
        """
        Linear inequality constraints.

        Returns
        -------
        LinearConstraints
            Linear inequality constraints.
        """
        return self._linear_ub

    @property
    def linear_eq(self):
        """
        Linear equality constraints.

        Returns
        -------
        LinearConstraints
            Linear equality constraints.
        """
        return self._linear_eq

    @property
    def m_bounds(self):
        """
        Number of bound constraints.

        Returns
        -------
        int
            Number of bound constraints.
        """
        return self.bounds.m

    @property
    def m_linear_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        return self.linear_ub.m

    @property
    def m_linear_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self.linear_eq.m

    @property
    def m_nonlinear_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear inequality constraints is not known.
        """
        return self._nonlinear_ub.m

    @property
    def m_nonlinear_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear equality constraints is not known.
        """
        return self._nonlinear_eq.m

    @property
    def type(self):
        """
        Type of the problem.

        The problem can be either 'unconstrained', 'bound-constrained',
        'linearly constrained', or 'nonlinearly constrained'.

        Returns
        -------
        str
            Type of the problem.
        """
        try:
            if self.m_nonlinear_ub > 0 or self.m_nonlinear_eq > 0:
                return 'nonlinearly constrained'
            elif self.m_linear_ub > 0 or self.m_linear_eq > 0:
                return 'linearly constrained'
            elif self.m_bounds > 0:
                return 'bound-constrained'
            else:
                return 'unconstrained'
        except ValueError:
            # The number of nonlinear constraints is not known. It may be zero
            # if the user provided a nonlinear inequality and/or equality
            # constraint function that returns an empty array. However, as this
            # is not known before the first call to the function, we assume that
            # the problem is nonlinearly constrained.
            return 'nonlinearly constrained'

    def build_x(self, x):
        """
        Build the full vector of variables from the reduced vector.

        Parameters
        ----------
        x : array_like, shape (n,)
            Reduced vector of variables.

        Returns
        -------
        numpy.ndarray, shape (n_orig,)
            Full vector of variables.
        """
        x_full = np.empty(self.n_orig)
        x_full[self._fixed_idx] = self._fixed_val
        x_full[~self._fixed_idx] = x
        return x_full

    def fun(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the objective function is evaluated.

        Returns
        -------
        float
            Value of the objective function at `x`.
        """
        return self._obj(self.build_x(self.bounds.project(x)))

    def cub(self, x):
        """
        Evaluate the nonlinear inequality constraints.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the nonlinear inequality constraints are evaluated.

        Returns
        -------
        numpy.ndarray
            Values of the nonlinear inequality constraints at `x`.
        """
        return self._nonlinear_ub(self.build_x(self.bounds.project(x)))

    def ceq(self, x):
        """
        Evaluate the nonlinear equality constraints.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the nonlinear equality constraints are evaluated.

        Returns
        -------
        numpy.ndarray
            Values of the nonlinear equality constraints at `x`.
        """
        return self._nonlinear_eq(self.build_x(self.bounds.project(x)))

    def resid(self, x, cub_val=None, ceq_val=None):
        """
        Evaluate the constraint residuals.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint residuals are evaluated.
        cub_val : array_like, shape (m_nonlinear_ub,), optional
            Values of the nonlinear inequality constraints. If not provided,
            the nonlinear inequality constraints are evaluated at `x`.
        ceq_val : array_like, shape (m_nonlinear_eq,), optional
            Values of the nonlinear equality constraints. If not provided,
            the nonlinear equality constraints are evaluated at `x`.

        Returns
        -------
        float
            Constraint residual at `x`.
        """
        resid_bounds = self.bounds.resid(x)
        resid_linear = max(self.linear_ub.resid(x), self.linear_eq.resid(x))
        resid_nonlinear = max(self._nonlinear_ub.resid(x, cub_val), self._nonlinear_eq.resid(x, ceq_val))
        return max(resid_bounds, resid_linear, resid_nonlinear)


def _1d_array(x, message):
    """
    Preprocess a 1-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 1-dimensional array.

    Returns
    -------
    numpy.ndarray
        Preprocessed array.
    """
    x = np.atleast_1d(np.squeeze(x)).astype(float)
    if x.ndim != 1:
        raise ValueError(message)
    return x


def _2d_array(x, message):
    """
    Preprocess a 2-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 2-dimensional array.

    Returns
    -------
    numpy.ndarray
        Preprocessed array.
    """
    x = np.atleast_2d(x).astype(float)
    if x.ndim != 2:
        raise ValueError(message)
    return x
