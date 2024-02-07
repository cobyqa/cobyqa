from contextlib import suppress
from inspect import signature

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
)

from .settings import PRINT_OPTIONS, BARRIER
from .utils import CallbackSuccess, get_arrays_tol
from .utils import exact_1d_array


class ObjectiveFunction:
    """
    Real-valued objective function.
    """

    def __init__(self, fun, verbose, debug, *args):
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
        debug : bool
            Whether to make debugging tests during the execution.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        if debug:
            assert fun is None or callable(fun)
            assert isinstance(verbose, bool)
            assert isinstance(debug, bool)

        self._fun = fun
        self._verbose = verbose
        self._args = args
        self._n_eval = 0

    def __call__(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the objective function is evaluated.

        Returns
        -------
        float
            Function value at `x`.
        """
        x = np.array(x, dtype=float)
        if self._fun is None:
            f = 0.0
        else:
            f = float(np.squeeze(self._fun(x, *self._args)))
            self._n_eval += 1
            if self._verbose:
                with np.printoptions(**PRINT_OPTIONS):
                    print(f"{self.name}({x}) = {f}")
        return f

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
    def name(self):
        """
        Name of the objective function.

        Returns
        -------
        str
            Name of the objective function.
        """
        name = ""
        if self._fun is not None:
            try:
                name = self._fun.__name__
            except AttributeError:
                name = "fun"
        return name


class BoundConstraints:
    """
    Bound constraints ``xl <= x <= xu``.
    """

    def __init__(self, bounds):
        """
        Initialize the bound constraints.

        Parameters
        ----------
        bounds : scipy.optimize.Bounds
            Bound constraints.
        """
        self._xl = np.array(bounds.lb, float)
        self._xu = np.array(bounds.ub, float)

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
        return (
            np.count_nonzero(self.xl > -np.inf)
            + np.count_nonzero(self.xu < np.inf)
        )

    @property
    def is_feasible(self):
        """
        Whether the bound constraints are feasible.

        Returns
        -------
        bool
            Whether the bound constraints are feasible.
        """
        return (
            np.all(self.xl <= self.xu)
            and np.all(self.xl < np.inf)
            and np.all(self.xu > -np.inf)
        )

    def maxcv(self, x):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.

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


class LinearConstraints:
    """
    Linear constraints ``a_ub @ x <= b_ub`` and ``a_eq @ x == b_eq``.
    """

    def __init__(self, constraints, n, debug):
        """
        Initialize the linear constraints.

        Parameters
        ----------
        constraints : list of LinearConstraint
            Linear constraints.
        n : int
            Number of variables.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(constraints, list)
            for constraint in constraints:
                assert isinstance(constraint, LinearConstraint)
            assert isinstance(debug, bool)

        self._a_ub = np.empty((0, n))
        self._b_ub = np.empty(0)
        self._a_eq = np.empty((0, n))
        self._b_eq = np.empty(0)
        for constraint in constraints:
            is_equality = (
                np.abs(constraint.ub - constraint.lb)
                <= get_arrays_tol(constraint.lb, constraint.ub)
            )
            if np.any(is_equality):
                self._a_eq = np.vstack((self.a_eq, constraint.A[is_equality]))
                self._b_eq = np.concatenate((
                    self.b_eq,
                    0.5 * (
                        constraint.lb[is_equality] + constraint.ub[is_equality]
                    ),
                ))
            if not np.all(is_equality):
                self._a_ub = np.vstack((
                    self.a_ub,
                    constraint.A[~is_equality],
                    -constraint.A[~is_equality],
                ))
                self._b_ub = np.concatenate((
                    self.b_ub,
                    constraint.ub[~is_equality],
                    -constraint.lb[~is_equality],
                ))

        # Remove the ill-defined constraints.
        self.a_ub[np.isnan(self.a_ub)] = 0.0
        self.a_eq[np.isnan(self.a_eq)] = 0.0
        undef_ub = np.isnan(self.b_ub) | np.isinf(self.b_ub)
        undef_eq = np.isnan(self.b_eq)
        self._a_ub = self.a_ub[~undef_ub, :]
        self._b_ub = self.b_ub[~undef_ub]
        self._a_eq = self.a_eq[~undef_eq, :]
        self._b_eq = self.b_eq[~undef_eq]

    @property
    def a_ub(self):
        """
        Left-hand side matrix of the linear inequality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Left-hand side matrix of the linear inequality constraints.
        """
        return self._a_ub

    @property
    def b_ub(self):
        """
        Right-hand side vector of the linear inequality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Right-hand side vector of the linear inequality constraints.
        """
        return self._b_ub

    @property
    def a_eq(self):
        """
        Left-hand side matrix of the linear equality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Left-hand side matrix of the linear equality constraints.
        """
        return self._a_eq

    @property
    def b_eq(self):
        """
        Right-hand side vector of the linear equality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Right-hand side vector of the linear equality constraints.
        """
        return self._b_eq

    @property
    def m_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        return self.b_ub.size

    @property
    def m_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self.b_eq.size

    def maxcv(self, x):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        x = np.array(x, dtype=float)
        return np.max([
            np.max(self.a_ub @ x - self.b_ub, initial=0.0),
            np.max(np.abs(self.a_eq @ x - self.b_eq), initial=0.0),
        ])


class NonlinearConstraints:
    """
    Nonlinear constraints ``c_ub(x) <= 0`` and ``c_eq(x) == b_eq``.
    """

    def __init__(self, constraints, verbose, debug):
        """
        Initialize the nonlinear constraints.

        Parameters
        ----------
        constraints : list
            Nonlinear constraints.
        verbose : bool
            Whether to print the function evaluations.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(constraints, list)
            for constraint in constraints:
                assert (
                    isinstance(constraint, NonlinearConstraint)
                    or isinstance(constraint, dict)
                )
            assert isinstance(verbose, bool)
            assert isinstance(debug, bool)

        self._constraints = constraints
        self._verbose = verbose
        self._m_ub = None
        self._m_eq = None
        self._n_eval = 0

    def __call__(self, x):
        """
        Evaluate the constraints.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraints are evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint function values.
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Nonlinear equality constraint function values.
        """
        x = np.array(x, dtype=float)
        c_ub = np.empty(0)
        c_eq = np.empty(0)
        for constraint in self._constraints:
            if isinstance(constraint, NonlinearConstraint):
                fun = constraint.fun
                val = fun(x)
            else:
                fun = constraint["fun"]
                args = constraint["args"]
                if not isinstance(args, tuple):
                    args = (args,)
                val = fun(x, *args)
            val = exact_1d_array(
                val,
                "The nonlinear constraints must return a vector.",
            )
            if self._verbose:
                with np.printoptions(**PRINT_OPTIONS):
                    with suppress(AttributeError):
                        fun_name = fun.__name__
                        print(f"{fun_name}({x}) = {val}")
            if isinstance(constraint, NonlinearConstraint):
                val, lb, ub = np.broadcast_arrays(
                    val,
                    constraint.lb,
                    constraint.ub,
                )
                lb = np.array(lb, float)
                ub = np.array(ub, float)
                lb[np.isnan(lb)] = -np.inf
                ub[np.isnan(ub)] = np.inf
                is_equality = np.abs(ub - lb) <= get_arrays_tol(lb, ub)
                if np.any(is_equality):
                    c_eq = np.concatenate((
                        c_eq,
                        val[is_equality]
                        - 0.5 * (lb[is_equality] + ub[is_equality]),
                    ))
                if not np.all(is_equality):
                    is_inequality_lb = ~is_equality & (lb > -np.inf)
                    is_inequality_ub = ~is_equality & (ub < np.inf)
                    if np.any(is_inequality_lb):
                        c_ub = np.concatenate((
                            c_ub,
                            lb[is_inequality_lb] - val[is_inequality_lb],
                        ))
                    if np.any(is_inequality_ub):
                        c_ub = np.concatenate((
                            c_ub,
                            val[is_inequality_ub] - ub[is_inequality_ub],
                        ))
            elif constraint["type"] == "ineq":
                c_ub = np.concatenate((c_ub, -val))
            else:
                c_eq = np.concatenate((c_eq, val))
        if len(self._constraints) > 0:
            self._n_eval += 1
        if self._m_ub is None:
            self._m_ub = c_ub.size
        if self._m_eq is None:
            self._m_eq = c_eq.size
        return c_ub, c_eq

    @property
    def m_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear inequality constraints is unknown.
        """
        if self._m_ub is None:
            raise ValueError(
                "The number of nonlinear inequality constraints is unknown."
            )
        else:
            return self._m_ub

    @property
    def m_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear equality constraints is unknown.
        """
        if self._m_eq is None:
            raise ValueError(
                "The number of nonlinear equality constraints is unknown."
            )
        else:
            return self._m_eq

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
        if cub_val is None or ceq_val is None:
            cub_val, ceq_val = self(x)
        return np.max([
            np.max(cub_val, initial=0.0),
            np.max(np.abs(ceq_val), initial=0.0),
        ])


class Problem:
    """
    Optimization problem.
    """

    def __init__(
        self,
        obj,
        x0,
        bounds,
        linear,
        nonlinear,
        callback,
        feasibility_tol,
        scale,
        store_history,
        history_size,
        filter_size,
        debug,
    ):
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
        linear : LinearConstraints
            Linear constraints.
        nonlinear : NonlinearConstraints
            Nonlinear constraints.
        callback : {callable, None}
            Callback function.
        feasibility_tol : float
            Tolerance on the constraint violation.
        scale : bool
            Whether to scale the problem according to the bounds.
        store_history : bool
            Whether to store the function evaluations.
        history_size : int
            Maximum number of function evaluations to store.
        filter_size : int
            Maximum number of points in the filter.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(obj, ObjectiveFunction)
            assert isinstance(bounds, BoundConstraints)
            assert isinstance(linear, LinearConstraints)
            assert isinstance(nonlinear, NonlinearConstraints)
            assert isinstance(feasibility_tol, float)
            assert isinstance(scale, bool)
            assert isinstance(store_history, bool)
            assert isinstance(history_size, int)
            if store_history:
                assert history_size > 0
            assert isinstance(filter_size, int)
            assert filter_size > 0
            assert isinstance(debug, bool)

        self._obj = obj
        self._linear = linear
        self._nonlinear = nonlinear
        if callback is not None:
            if not callable(callback):
                raise TypeError("The callback must be a callable function.")
        self._callback = callback

        # Check the consistency of the problem.
        x0 = exact_1d_array(x0, "The initial guess must be a vector.")
        n = x0.size
        if bounds.xl.size != n:
            raise ValueError(f"The bounds must have {n} elements.")
        if linear.a_ub.shape[1] != n:
            raise ValueError(
                f"The left-hand side matrices of the linear constraints must "
                f"have {n} columns."
            )

        # Check which variables are fixed.
        tol = get_arrays_tol(bounds.xl, bounds.xu)
        self._fixed_idx = (
            (bounds.xl <= bounds.xu)
            & (np.abs(bounds.xl - bounds.xu) < tol)
        )
        self._fixed_val = 0.5 * (
            bounds.xl[self._fixed_idx] + bounds.xu[self._fixed_idx]
        )
        self._fixed_val = np.clip(
            self._fixed_val,
            bounds.xl[self._fixed_idx],
            bounds.xu[self._fixed_idx],
        )

        # Set the bound constraints.
        self._orig_bounds = bounds
        self._bounds = BoundConstraints(
            Bounds(bounds.xl[~self._fixed_idx], bounds.xu[~self._fixed_idx])
        )

        # Set the initial guess.
        self._x0 = self._bounds.project(x0[~self._fixed_idx])

        # Set the linear constraints.
        b_eq = linear.b_eq - linear.a_eq[:, self._fixed_idx] @ self._fixed_val
        self._linear = LinearConstraints(
            [
                LinearConstraint(
                    linear.a_ub[:, ~self._fixed_idx],
                    -np.inf,
                    linear.b_ub
                    - linear.a_ub[:, self._fixed_idx] @ self._fixed_val,
                ),
                LinearConstraint(linear.a_eq[:, ~self._fixed_idx], b_eq, b_eq),
            ],
            self.n,
            debug,
        )

        # Scale the problem if necessary.
        scale = (
            scale
            and self._bounds.is_feasible
            and np.all(np.isfinite(self._bounds.xl))
            and np.all(np.isfinite(self._bounds.xu))
        )
        if scale:
            self._scaling_factor = 0.5 * (self._bounds.xu - self._bounds.xl)
            self._scaling_shift = 0.5 * (self._bounds.xu + self._bounds.xl)
            self._bounds = BoundConstraints(
                Bounds(-np.ones(self.n), np.ones(self.n))
            )
            b_eq = self._linear.b_eq - self._linear.a_eq @ self._scaling_shift
            self._linear = LinearConstraints(
                [
                    LinearConstraint(
                        self._linear.a_ub @ np.diag(self._scaling_factor),
                        -np.inf,
                        self._linear.b_ub
                        - self._linear.a_ub @ self._scaling_shift,
                    ),
                    LinearConstraint(
                        self._linear.a_eq @ np.diag(self._scaling_factor),
                        b_eq,
                        b_eq,
                    ),
                ],
                self.n,
                debug,
            )
            self._x0 = (self._x0 - self._scaling_shift) / self._scaling_factor
        else:
            self._scaling_factor = np.ones(self.n)
            self._scaling_shift = np.zeros(self.n)

        # Set the initial filter.
        self._feasibility_tol = feasibility_tol
        self._filter_size = filter_size
        self._fun_filter = []
        self._maxcv_filter = []
        self._x_filter = []

        # Set the initial history.
        self._store_history = store_history
        self._history_size = history_size
        self._fun_history = []
        self._maxcv_history = []
        self._x_history = []

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

        Raises
        ------
        `cobyqa.utils.CallbackSuccess`
            If the callback function raises a ``StopIteration``.
        """
        # Evaluate the objective and nonlinear constraint functions.
        x = np.asarray(x, dtype=float)
        fun_val = self._obj(self.build_x(x))
        if np.isnan(fun_val):
            fun_val = BARRIER
        fun_val = max(min(fun_val, BARRIER), -BARRIER)
        cub_val, ceq_val = self._nonlinear(self.build_x(x))
        cub_val[np.isnan(cub_val)] = BARRIER
        ceq_val[np.isnan(ceq_val)] = BARRIER
        cub_val = np.minimum(cub_val, BARRIER)
        ceq_val = np.minimum(ceq_val, BARRIER)
        cub_val = np.maximum(cub_val, -BARRIER)
        ceq_val = np.maximum(ceq_val, -BARRIER)
        maxcv_val = self.maxcv(x, cub_val, ceq_val)
        if self._store_history:
            if len(self._fun_history) >= self._history_size:
                self._fun_history.pop(0)
            self._fun_history.append(fun_val)
            if len(self._maxcv_history) >= self._history_size:
                self._maxcv_history.pop(0)
            self._maxcv_history.append(maxcv_val)
            if len(self._x_history) >= self._history_size:
                self._x_history.pop(0)
            self._x_history.append(x)

        # Add the point to the filter if it is not dominated by any point.
        maxcv_shift = np.max([maxcv_val - self._feasibility_tol, 0.0])
        if all(
            fun_val < fun_filter
            or maxcv_shift < np.max([
                maxcv_filter - self._feasibility_tol,
                0.0,
            ]) for fun_filter, maxcv_filter in zip(
                self._fun_filter,
                self._maxcv_filter,
            )
        ):
            self._fun_filter.append(fun_val)
            self._maxcv_filter.append(maxcv_val)
            self._x_filter.append(np.copy(x))

        # Remove the points in the filter that are dominated by the new point.
        for k in range(len(self._fun_filter) - 2, -1, -1):
            if fun_val <= self._fun_filter[k] and maxcv_shift <= np.max([
                self._maxcv_filter[k] - self._feasibility_tol,
                0.0,
            ]):
                self._fun_filter.pop(k)
                self._maxcv_filter.pop(k)
                self._x_filter.pop(k)

        # Keep only the most recent points in the filter.
        if len(self._fun_filter) > self._filter_size:
            self._fun_filter.pop(0)
            self._maxcv_filter.pop(0)
            self._x_filter.pop(0)

        # Evaluate the callback function after updating the filter to ensure
        # that the current point can be returned by the method.
        if self._callback is not None:
            sig = signature(self._callback)
            try:
                if set(sig.parameters) == {"intermediate_result"}:
                    intermediate_result = OptimizeResult(x=x, fun=fun_val)
                    self._callback(intermediate_result)
                else:
                    self._callback(x)
            except StopIteration as exc:
                raise CallbackSuccess from exc

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
    def n_eval(self):
        """
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        return max(self._obj.n_eval, self._nonlinear.n_eval)

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
    def linear(self):
        """
        Linear constraints.

        Returns
        -------
        LinearConstraints
            Linear constraints.
        """
        return self._linear

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
        return self.linear.m_ub

    @property
    def m_linear_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self.linear.m_eq

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
        return self._nonlinear.m_ub

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
        return self._nonlinear.m_eq

    @property
    def fun_history(self):
        """
        History of objective function evaluations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval,)
            History of objective function evaluations.
        """
        return np.array(self._fun_history, dtype=float)

    @property
    def maxcv_history(self):
        """
        History of maximum constraint violations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval,)
            History of maximum constraint violations.
        """
        return np.array(self._maxcv_history, dtype=float)

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
                return "nonlinearly constrained"
            elif self.m_linear_ub > 0 or self.m_linear_eq > 0:
                return "linearly constrained"
            elif self.m_bounds > 0:
                return "bound-constrained"
            else:
                return "unconstrained"
        except ValueError:
            # The number of nonlinear constraints is not known. It may be zero
            # if the user provided a nonlinear inequality and/or equality
            # constraint function that returns an empty array. However, as this
            # is not known before the first call to the function, we assume
            # that the problem is nonlinearly constrained.
            return "nonlinearly constrained"

    @property
    def is_feasibility(self):
        """
        Whether the problem is a feasibility problem.

        Returns
        -------
        bool
            Whether the problem is a feasibility problem.
        """
        return self.fun_name == ""

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
        x_full[~self._fixed_idx] = (
            x * self._scaling_factor + self._scaling_shift
        )
        return self._orig_bounds.project(x_full)

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
        return np.max([
            self.bounds.maxcv(x),
            self.linear.maxcv(x),
            self._nonlinear.maxcv(x, cub_val, ceq_val),
        ])

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
            Corresponding objective function value.
        float
            Corresponding maximum constraint violation.
        """
        # If the filter is empty, i.e., if no function evaluation has been
        # performed, we evaluate the objective and nonlinear constraint
        # functions at the initial guess.
        if len(self._fun_filter) == 0:
            self(self.x0)

        # Find the best point in the filter.
        fun_filter = np.array(self._fun_filter)
        maxcv_filter = np.array(self._maxcv_filter)
        maxcv_filter_shifted = np.maximum(
            maxcv_filter - self._feasibility_tol,
            0.0,
        )
        x_filter = np.array(self._x_filter)
        feasible_idx = maxcv_filter_shifted <= min(
            np.finfo(float).eps,
            np.min(maxcv_filter_shifted),
        )
        if np.any(feasible_idx):
            # At least one point is nearly feasible. We select the one with
            # the least objective function value. If there is a tie, we
            # select the point with the least maximum constraint violation.
            # If there is still a tie, we select the most recent point.
            fun_min_idx = (
                feasible_idx
                & (fun_filter <= np.min(fun_filter[feasible_idx]))
            )
            if np.count_nonzero(fun_min_idx) == 1:
                i = np.flatnonzero(fun_min_idx)[0]
            else:
                fun_min_idx &= (
                    maxcv_filter_shifted
                    <= np.min(maxcv_filter_shifted[fun_min_idx])
                )
                i = np.flatnonzero(fun_min_idx)[-1]
        else:
            # No feasible point is found. We select the one with the least
            # merit function value. If there is a tie, we select the point
            # with the least maximum constraint violation. If there is still
            # a tie, we select the most recent point.
            if np.all(~np.isfinite(maxcv_filter_shifted)):
                if np.all(np.isnan(fun_filter)):
                    i = 0
                else:
                    i = np.nanargmin(fun_filter)
            else:
                merit_filter = fun_filter + penalty * maxcv_filter_shifted
                merit_min_idx = merit_filter <= np.min(merit_filter)
                if np.count_nonzero(merit_min_idx) == 1:
                    i = np.flatnonzero(merit_min_idx)[0]
                else:
                    merit_min_idx &= (
                        maxcv_filter_shifted
                        <= np.min(maxcv_filter_shifted[merit_min_idx])
                    )
                    i = np.flatnonzero(merit_min_idx)[-1]
        return (
            self.bounds.project(x_filter[i, :]),
            fun_filter[i],
            maxcv_filter[i],
        )
