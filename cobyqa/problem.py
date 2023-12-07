from abc import ABC

import numpy as np
from scipy.optimize import OptimizeResult

from .settings import PRINT_OPTIONS
from .utils import get_arrays_tol


class Function(ABC):
    """
    Base class for objective and constraints functions.
    """

    def __init__(self, fun, verbose, store_fun_history, store_x_history, history_size, debug, *args):
        """
        Initialize the function.

        Parameters
        ----------
        fun : {callable, None}
            Function to evaluate, or None.

                ``fun(x, *args) -> {float, array_like}``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        verbose : bool
            Whether to print the function evaluations.
        store_fun_history : bool
            Whether to store the function evaluations.
        store_x_history : bool
            Whether to store the visited points.
        history_size : int
            Maximum number of function evaluations to store.
        debug : bool
            Whether to make debugging tests during the execution.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        if debug:
            assert fun is None or callable(fun)
            assert isinstance(verbose, bool)
            assert isinstance(store_fun_history, bool)
            assert isinstance(store_x_history, bool)
            assert isinstance(history_size, int)
            if store_fun_history or store_x_history:
                assert history_size > 0
            assert isinstance(debug, bool)

        self._fun = fun
        self._verbose = verbose
        self._store_fun_history = store_fun_history
        self._store_x_history = store_x_history
        self._history_size = history_size
        self._args = args
        self._n_eval = 0
        self._fun_history = []
        self._x_history = []
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
        {float, `numpy.ndarray`}
            Function value at `x`.
        """
        x = np.array(x, dtype=float)
        if self._fun is None:
            val = self.apply_barrier()
        else:
            val = self._fun(x, *self._args)
            val = self.apply_barrier(val)
            self._n_eval += 1
            if self._verbose:
                with np.printoptions(**PRINT_OPTIONS):
                    print(f'{self.name}({x}) = {val}')
        if self._store_fun_history:
            if len(self._fun_history) >= self._history_size:
                self._fun_history.pop(0)
            self._fun_history.append(val)
        if self._store_x_history:
            if len(self._x_history) >= self._history_size:
                self._x_history.pop(0)
            self._x_history.append(x)
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
        return self._fun.__name__ if self._fun is not None else ''

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
    def fun_history(self):
        """
        History of function evaluations.

        This property returns an empty array if the history of function
        evaluations is not maintained.

        Returns
        -------
        `numpy.ndarray`
            History of function evaluations.
        """
        return np.array(self._fun_history, dtype=float)

    @property
    def x_history(self):
        """
        History of variables.

        This property returns an empty array if the history of function
        evaluations is not maintained.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval, n)
            History of variables.
        """
        return np.array(self._x_history, dtype=float)

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
        {float, `numpy.ndarray`}
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

    def maxcv(self, x, *args):
        """
        Evaluate the maximum constraint violation.

        This method must be implemented in the derived classes.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        *args : tuple
            Additional arguments to be passed to the function.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
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
        `numpy.ndarray`, shape (n,)
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

    def __init__(self, fun, callback, verbose, store_history, history_size, debug, *args):
        """
        Initialize the objective function.

        Parameters
        ----------
        fun : {callable, None}
            Function to evaluate, or None.

                ``fun(x, *args) -> float``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        verbose : bool
            Whether to print the function evaluations.
        store_history : bool
            Whether to store the function evaluations.
        history_size : int
            Maximum number of function evaluations to store.
        debug : bool
            Whether to make debugging tests during the execution.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        super().__init__(fun, verbose, store_history, store_history, history_size, debug, *args)
        self._callback = callback

    def __call__(self, x):
        """
        Evaluate the objective function.

        This method also applies the barrier function to the function value.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the objective function is evaluated.

        Returns
        -------
        float
            Function value at `x`.
        """
        f = super().__call__(x)
        if self._callback is not None:
            if not callable(self._callback):
                raise ValueError('The callback must be a callable function.')
            intermediate_result = OptimizeResult(x=x, fun=f)
            self._callback(intermediate_result)
        return f

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
        return max(min(val, self._barrier), -self._barrier)


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
        `numpy.ndarray`, shape (n,)
            Lower bound.
        """
        return self._xl

    @property
    def xu(self):
        """
        Upper bound.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
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

    def maxcv(self, x, *args):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        *args : tuple
            This argument is ignored.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
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
        `numpy.ndarray`, shape (n,)
            Projection of `x` onto the feasible set.
        """
        return np.clip(x, self.xl, self.xu) if self.is_feasible else x


class LinearConstraints(Constraints):
    """
    Linear constraints ``a @ x <= b`` or ``a @ x == b``.
    """

    def __init__(self, a, b, is_equality, debug):
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
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(is_equality, bool)

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
        `numpy.ndarray`, shape (m, n)
            Left-hand side of the linear constraints.
        """
        return self._a

    @property
    def b(self):
        """
        Right-hand side of the linear constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m,)
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

    def maxcv(self, x, *args):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        *args : tuple
            This argument is ignored.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
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

    def __init__(self, fun, is_equality, verbose, store_history, history_size, debug, *args):
        """
        Initialize the nonlinear constraints.

        Parameters
        ----------
        fun : {callable, None}
            Function to evaluate, or None.

                ``fun(x, *args) -> array_like``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        is_equality : bool
            Whether the nonlinear constraints are equality constraints.
        verbose : bool
            Whether to print the function evaluations.
        store_history : bool
            Whether to store the function evaluations.
        history_size : int
            Maximum number of function evaluations to store.
        debug : bool
            Whether to make debugging tests during the execution.
        *args : tuple
            Additional arguments passed to the function.
        """
        if debug:
            assert isinstance(is_equality, bool)

        super().__init__(fun, verbose, store_history, False, history_size, debug, *args)
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
        `numpy.ndarray`, shape (m,)
            Function value with the barrier function applied.
        """
        if val is None:
            val = np.empty(0)
        val = _1d_array(val, 'The nonlinear constraints must return a vector.')
        val[np.isnan(val)] = self._barrier
        val = np.minimum(val, self._barrier)
        val = np.maximum(val, -self._barrier)
        if self._m is None:
            self._m = val.size
        return val

    def maxcv(self, x, *args):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        *args : tuple
            One argument is expected, which is the function value at `x`. If
            this argument is not passed, the constraint function is evaluated.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
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

    def __init__(self, obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq, feasibility_tol, scale, filter_size, debug):
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
        feasibility_tol : float
            Tolerance on the constraint violation.
        scale : bool
            Whether to scale the problem according to the bounds.
        filter_size : int
            Maximum number of points in the filter.
        debug : bool
            Whether to make debugging tests during the execution.
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
        self._fixed_val = np.clip(self._fixed_val, bounds.xl[self._fixed_idx], bounds.xu[self._fixed_idx])

        # Set the bound and linear constraints.
        self._bounds = BoundConstraints(bounds.xl[~self._fixed_idx], bounds.xu[~self._fixed_idx])
        self._linear_ub = LinearConstraints(linear_ub.a[:, ~self._fixed_idx], linear_ub.b - linear_ub.a[:, self._fixed_idx] @ self._fixed_val, False, debug)
        self._linear_eq = LinearConstraints(linear_eq.a[:, ~self._fixed_idx], linear_eq.b - linear_eq.a[:, self._fixed_idx] @ self._fixed_val, True, debug)

        # Set the initial guess.
        self._x0 = self._bounds.project(x0[~self._fixed_idx])

        # Scale the problem if necessary.
        scale = scale and self._bounds.is_feasible and np.all(np.isfinite(self._bounds.xl)) and np.all(np.isfinite(self._bounds.xu))
        if scale:
            self._scaling_factor = 0.5 * (self._bounds.xu - self._bounds.xl)
            self._scaling_shift = 0.5 * (self._bounds.xu + self._bounds.xl)
            self._bounds = BoundConstraints(-np.ones(self.n), np.ones(self.n))
            self._linear_ub = LinearConstraints(self._linear_ub.a @ np.diag(self._scaling_factor), self._linear_ub.b - self._linear_ub.a @ self._scaling_shift, False, debug)
            self._linear_eq = LinearConstraints(self._linear_eq.a @ np.diag(self._scaling_factor), self._linear_eq.b - self._linear_eq.a @ self._scaling_shift, True, debug)
            self._x0 = (self._x0 - self._scaling_shift) / self._scaling_factor
        else:
            self._scaling_factor = np.ones(self.n)
            self._scaling_shift = np.zeros(self.n)

        # Set the initial filter.
        self._feasibility_tol = feasibility_tol
        self._filter_size = filter_size
        self._fun_filter = []
        self._cub_filter = []
        self._ceq_filter = []
        self._maxcv_filter = []
        self._x_filter = []

    def __call__(self, x):
        """
        Evaluate the objective and nonlinear constraint functions.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the functions are evaluated.

        Returns
        -------
        float
            Objective function value.
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint function values.
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Nonlinear equality constraint function values.
        """
        # Evaluate the objective and nonlinear constraint functions.
        x_eval = self.bounds.project(x)
        fun_val = self._obj(self.build_x(x_eval))
        cub_val = self._nonlinear_ub(self.build_x(x_eval))
        ceq_val = self._nonlinear_eq(self.build_x(x_eval))

        # Add the point to the filter if it is not dominated by any point.
        maxcv_val = self.maxcv(x_eval, cub_val, ceq_val)
        maxcv_shift = max(maxcv_val - self._feasibility_tol, 0.0)
        if all(fun_val < fun_filter or maxcv_shift < max(maxcv_filter - self._feasibility_tol, 0.0) for fun_filter, maxcv_filter in zip(self._fun_filter, self._maxcv_filter)):
            self._fun_filter.append(fun_val)
            self._cub_filter.append(cub_val)
            self._ceq_filter.append(ceq_val)
            self._maxcv_filter.append(maxcv_val)
            self._x_filter.append(x_eval)

        # Remove the points in the filter that are dominated by the new point.
        for k in range(len(self._fun_filter) - 2, -1, -1):
            if fun_val <= self._fun_filter[k] and maxcv_shift <= max(self._maxcv_filter[k] - self._feasibility_tol, 0.0):
                self._fun_filter.pop(k)
                self._cub_filter.pop(k)
                self._ceq_filter.pop(k)
                self._maxcv_filter.pop(k)
                self._x_filter.pop(k)

        # Keep only the most recent points in the filter.
        if len(self._fun_filter) > self._filter_size:
            self._fun_filter.pop(0)
            self._cub_filter.pop(0)
            self._ceq_filter.pop(0)
            self._maxcv_filter.pop(0)
            self._x_filter.pop(0)

        return fun_val, cub_val, ceq_val

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
        `numpy.ndarray`, shape (n,)
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
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        return max(self._obj.n_eval, self._nonlinear_ub.n_eval, self._nonlinear_eq.n_eval)

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
    def fun_history(self):
        """
        History of objective function evaluations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval,)
            History of objective function evaluations.
        """
        return self._obj.fun_history

    @property
    def cub_history(self):
        """
        History of nonlinear inequality constraint function evaluations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval, m_nonlinear_ub)
            History of nonlinear inequality constraint function evaluations.
        """
        return self._nonlinear_ub.fun_history

    @property
    def ceq_history(self):
        """
        History of nonlinear equality constraint function evaluations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval, m_nonlinear_eq)
            History of nonlinear equality constraint function evaluations.
        """
        return self._nonlinear_eq.fun_history

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

    @property
    def is_feasibility(self):
        """
        Whether the problem is a feasibility problem.

        Returns
        -------
        bool
            Whether the problem is a feasibility problem.
        """
        return self.fun_name == ''

    def build_x(self, x):
        """
        Build the full vector of variables from the reduced vector.

        Parameters
        ----------
        x : array_like, shape (n,)
            Reduced vector of variables.

        Returns
        -------
        `numpy.ndarray`, shape (n_orig,)
            Full vector of variables.
        """
        x_full = np.empty(self.n_orig)
        x_full[self._fixed_idx] = self._fixed_val
        x_full[~self._fixed_idx] = x * self._scaling_factor + self._scaling_shift
        return x_full

    def maxcv(self, x, cub_val=None, ceq_val=None):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        cub_val : array_like, shape (m_nonlinear_ub,), optional
            Values of the nonlinear inequality constraints. If not provided,
            the nonlinear inequality constraints are evaluated at `x`.
        ceq_val : array_like, shape (m_nonlinear_eq,), optional
            Values of the nonlinear equality constraints. If not provided,
            the nonlinear equality constraints are evaluated at `x`.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        maxcv_bounds = self.bounds.maxcv(x)
        maxcv_linear = max(self.linear_ub.maxcv(x), self.linear_eq.maxcv(x))
        maxcv_nonlinear = max(self._nonlinear_ub.maxcv(x, cub_val), self._nonlinear_eq.maxcv(x, ceq_val))
        return max(maxcv_bounds, maxcv_linear, maxcv_nonlinear)

    def best_eval(self, penalty):
        """
        Return the best point in the filter and the corresponding objective and
        nonlinear constraint function evaluations.

        Parameters
        ----------
        penalty : float
            Penalty parameter

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Best point.
        float
            Best objective function value.
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Best nonlinear inequality constraint function values.
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Best nonlinear equality constraint function values.
        """
        # If the filter is empty, i.e., if no function evaluation has been
        # performed, we evaluate the objective and nonlinear constraint
        # functions at the initial guess.
        if len(self._fun_filter) == 0:
            self(self.x0)

        # Find the best point in the filter.
        fun_filter = np.array(self._fun_filter)
        cub_filter = np.array(self._cub_filter)
        ceq_filter = np.array(self._ceq_filter)
        maxcv_filter = np.array(self._maxcv_filter)
        maxcv_filter = np.maximum(maxcv_filter - self._feasibility_tol, 0.0)
        x_filter = np.array(self._x_filter)
        feasible_idx = maxcv_filter < max(np.finfo(float).eps, 2.0 * np.min(maxcv_filter))
        if np.any(feasible_idx):
            # At least one point is nearly feasible. We select the one with
            # the least objective function value. If there is a tie, we
            # select the point with the least maximum constraint violation.
            # If there is still a tie, we select the most recent point.
            fun_min_idx = feasible_idx & (fun_filter <= np.min(fun_filter[feasible_idx]))
            if np.count_nonzero(fun_min_idx) == 1:
                i = np.flatnonzero(fun_min_idx)[0]
            else:
                fun_min_idx &= (maxcv_filter <= np.min(maxcv_filter))
                i = np.flatnonzero(fun_min_idx)[-1]
        else:
            # No feasible point is found. We select the one with the least
            # merit function value. If there is a tie, we select the point
            # with the least maximum constraint violation. If there is still
            # a tie, we select the most recent point.
            merit_filter = fun_filter + penalty * maxcv_filter
            merit_min_idx = merit_filter <= np.min(merit_filter)
            if np.count_nonzero(merit_min_idx) == 1:
                i = np.flatnonzero(merit_min_idx)[0]
            else:
                merit_min_idx &= (maxcv_filter <= np.min(maxcv_filter))
                i = np.flatnonzero(merit_min_idx)[-1]
        return self.bounds.project(x_filter[i, :]), fun_filter[i], cub_filter[i, :], ceq_filter[i, :]


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
    `numpy.ndarray`
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
    `numpy.ndarray`
        Preprocessed array.
    """
    x = np.atleast_2d(x).astype(float)
    if x.ndim != 2:
        raise ValueError(message)
    return x
