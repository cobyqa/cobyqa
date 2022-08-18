import copy
import warnings
from contextlib import suppress

import numpy as np
from numpy.testing import assert_, assert_array_less
from scipy.linalg import get_blas_funcs
from scipy.optimize import Bounds, NonlinearConstraint,\
    minimize as scipy_minimize

from .linalg import bvcs, bvlag, bvtcg, cpqp, lctcg, nnls
from .utils import RestartRequiredException, huge, implicit_hessian, \
    normalize, absmax_arrays


class OptimizeResult(dict):
    """
    Structure for the result of the optimization algorithm.

    Attributes
    ----------
    x : numpy.ndarray, shape (n,)
        Solution point provided by the optimization solver.
    success : bool
        Flag indicating whether the optimization solver terminated successfully.
    status : int
        Termination status of the optimization solver.
    message : str
        Description of the termination status of the optimization solver.
    fun : float
        Value of the objective function at the solution point provided by the
        optimization solver.
    jac : numpy.ndarray, shape (n,)
        Approximation of the gradient of the objective function at the solution
        point provided by the optimization solver, based on undetermined
        interpolation. If the value of a component (or more) of the gradient is
        unknown, it is replaced by ``numpy.nan``.
    nfev : int
        Number of objective and constraint function evaluations.
    nit : int
        Number of iterations performed by the optimization solver.
    maxcv : float
        Maximum constraint violation at the solution point provided by the
        optimization solver. It is set only if the problem is not declared
        unconstrained by the optimization solver.
    """

    def __dir__(self):
        """
        Get the names of the attributes in the current scope.

        Returns
        -------
        list:
            Names of the attributes in the current scope.
        """
        return list(self.keys())

    def __getattr__(self, name):
        """
        Get the value of an attribute that is not explicitly defined.

        Parameters
        ----------
        name : str
            Name of the attribute to be assessed.

        Returns
        -------
        object
            Value of the attribute.

        Raises
        ------
        AttributeError
            The required attribute does not exist.
        """
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, key, value):
        """
        Assign an existing or a new attribute.

        Parameters
        ----------
        key : str
            Name of the attribute to be assigned.
        value : object
            Value of the attribute to be assigned.
        """
        super().__setitem__(key, value)

    def __delattr__(self, key):
        """
        Delete an attribute.

        Parameters
        ----------
        key : str
            Name of the attribute to be deleted.

        Raises
        ------
        KeyError
            The required attribute does not exist.
        """
        super().__delitem__(key)

    def __repr__(self):
        """
        Get a string representation that looks like valid Python expression,
        which can be used to recreate an object with the same value, given an
        appropriate environment.

        Returns
        -------
        str
            String representation of instances of this class.
        """
        attrs = ', '.join(f'{k}={repr(v)}' for k, v in sorted(self.items()))
        return f'{self.__class__.__name__}({attrs})'

    def __str__(self):
        """
        Get a string representation, designed to be nicely printable.

        Returns
        -------
        str
            String representation of instances of this class.
        """
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return '\n'.join(f'{k:>{m}}: {v}' for k, v in sorted(self.items()))
        return f'{self.__class__.__name__}()'


class TrustRegion:
    """
    Framework atomization of the derivative-free trust-region SQP method.
    """

    def __init__(self, fun, x0, xl=None, xu=None, Aub=None, bub=None, Aeq=None,
                 beq=None, cub=None, ceq=None, options=None, *args, **kwargs):
        """
        Initialize the derivative-free trust-region SQP method.

        Parameters
        ----------
        fun : callable
            Objective function to be minimized.

                ``fun(x, *args) -> float``

            where ``x`` is an array with shape (n,) and `args` is a tuple of
            parameters to specify the objective function.
        x0 : array_like, shape (n,)
            Initial guess.
        xl : array_like, shape (n,), optional
            Lower-bound constraints on the decision variables ``x >= xl``.
        xu : array_like, shape (n,), optional
            Upper-bound constraints on the decision variables ``x <= xu``.
        Aub : array_like, shape (mlub, n), optional
            Jacobian matrix of the linear inequality constraints. Each row of
            `Aub` stores the gradient of a linear inequality constraint.
        bub : array_like, shape (mlub,), optional
            Right-hand side vector of the linear inequality constraints
            ``Aub @ x <= bub``.
        Aeq : array_like, shape (mleq, n), optional
            Jacobian matrix of the linear equality constraints. Each row of
            `Aeq` stores the gradient of a linear equality constraint.
        beq : array_like, shape (mleq,), optional
            Right-hand side vector of the linear equality constraints
            ``Aeq @ x = beq``.
        cub : callable, optional
            Nonlinear inequality constraint function ``cub(x) <= 0``.

                ``cub(x, *args) -> numpy.ndarray, shape (mnlub,)``

            where ``x`` is an array with shape (n,) and `args` is a tuple of
            parameters to specify the constraint function.
        ceq : callable, optional
            Nonlinear equality constraint function ``ceq(x) = 0``.

                ``ceq(x, *args) -> numpy.ndarray, shape (mnleq,)``

            where ``x`` is an array with shape (n,) and `args` is a tuple of
            parameters to specify the constraint function.
        options : dict, optional
            Options to forward to the solver. Accepted options are:

                rhobeg : float, optional
                    Initial trust-region radius (the default is 1).
                rhoend : float, optional
                    Final trust-region radius (the default is 1e-6).
                npt : int, optional
                    Number of interpolation points for the objective and
                    constraint models (the default is ``2 * n + 1``).
                maxfev : int, optional
                    Upper bound on the number of objective and constraint
                    function evaluations (the default is ``500 * n``).
                maxiter: int, optional
                    Upper bound on the number of main loop iterations (the
                    default is ``1000 * n``).
                target : float, optional
                    Target value on the objective function (the default is
                    ``-numpy.inf``). If the solver encounters a feasible point
                    at which the objective function evaluations is below the
                    target value, then the computations are stopped.
                ftol_abs : float, optional
                    Absolute tolerance on the objective function.
                ftol_rel : float, optional
                    Relative tolerance on the objective function.
                xtol_abs : float, optional
                    Absolute tolerance on the decision variables.
                xtol_rel : float, optional
                    Relative tolerance on the decision variables.
                disp : bool, optional
                    Whether to print pieces of information on the execution of
                    the solver (the default is False).
                respect_bounds : bool, optional
                    Whether to respect the bounds through the iterations (the
                    default is True).
                debug : bool, optional
                    Whether to make debugging tests during the execution, which
                    is not recommended in production (the default is False).
        *args : tuple, optional
            Parameters to forward to the objective function, the nonlinear
            inequality constraint function, and the nonlinear equality
            constraint function.

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).
        """
        self._fun = fun
        self._cub = cub
        self._ceq = ceq
        x0 = np.atleast_1d(x0).astype(float)
        n = x0.size
        self._args = args
        if xl is None:
            xl = np.full_like(x0, -np.inf)
        xl = np.atleast_1d(xl).astype(float)
        if xu is None:
            xu = np.full_like(x0, np.inf)
        xu = np.atleast_1d(xu).astype(float)
        if Aub is None:
            Aub = np.empty((0, n))
        Aub = np.atleast_2d(Aub).astype(float)
        if bub is None:
            bub = np.empty(0)
        bub = np.atleast_1d(bub).astype(float)
        if Aeq is None:
            Aeq = np.empty((0, n))
        Aeq = np.atleast_2d(Aeq).astype(float)
        if beq is None:
            beq = np.empty(0)
        beq = np.atleast_1d(beq).astype(float)
        if options is None:
            options = {}
        self._options = dict(options)

        # Remove NaN and infinite values from the constraints.
        xl[np.isnan(xl)] = -np.inf
        xu[np.isnan(xu)] = np.inf
        np.nan_to_num(Aub, False)
        np.nan_to_num(Aeq, False)
        iub = np.isfinite(bub)
        Aub = Aub[iub, :]
        bub = bub[iub]
        ieq = np.isfinite(beq)
        Aeq = Aeq[ieq, :]
        beq = beq[ieq]

        # Remove the variables that are fixed by the bounds.
        bdtol = 10.0 * np.finfo(float).eps * n
        bdtol *= absmax_arrays(xl, xu, initial=1.0)
        self._ifix = np.abs(xl - xu) <= bdtol
        ifree = np.logical_not(self.ifix)
        self._xfix = 0.5 * (xl[self.ifix] + xu[self.ifix])
        x0 = x0[ifree]
        xl = xl[ifree]
        xu = xu[ifree]
        bub -= np.dot(Aub[:, self.ifix], self.xfix)
        Aub = Aub[:, ifree]
        beq -= np.dot(Aeq[:, self.ifix], self.xfix)
        Aeq = Aeq[:, ifree]

        # Set the default options based on the reduced problem. If all variables
        # are fixed by the bounds, the reduced value of n will be zero, in which
        # case npt will be fixed to one, and only one function evaluation will
        # be performed during the initialization procedure.
        n -= np.count_nonzero(self.ifix)
        self.set_default_options(n)
        self.check_options(n)

        # Consider the bounds as linear constraints if it is not necessary to
        # respect them.
        if not self.respect_bounds:
            ixl = xl > -np.inf
            ixu = xu < np.inf
            identity = np.eye(n)
            Aub = np.r_[Aub, -identity[ixl, :], identity[ixu, :]]
            bub = np.r_[bub, -xl[ixl], xu[ixu]]
            xl = np.full_like(xl, -np.inf)
            xu = np.full_like(xu, np.inf)

        # Project the initial guess onto the bound constraints.
        x0 = np.minimum(xu, np.maximum(xl, x0))

        # Modify the initial guess in order to avoid conflicts between the
        # bounds and the initial interpolation points. The coordinates of the
        # initial guess should either equal the bound components or allow the
        # projection of the initial trust region onto the components to lie
        # entirely inside the bounds.
        if not np.all(self.ifix):
            rhobeg = self.rhobeg
            rhoend = self.rhoend
            rhobeg = min(0.5 * np.min(xu - xl), rhobeg)
            rhoend = min(rhobeg, rhoend)
            self.options.update({'rhobeg': rhobeg, 'rhoend': rhoend})
            adj = (x0 - xl <= rhobeg) & (xl < x0)
            if np.any(adj):
                x0[adj] = xl[adj] + rhobeg
            adj = (xu - x0 <= rhobeg) & (x0 < xu)
            if np.any(adj):
                x0[adj] = xu[adj] - rhobeg

        # Set the initial shift of the origin, designed to manage the effects
        # of computer rounding errors in the calculations.
        self._xbase = x0
        self._x_hist = []
        self._fun_hist = []
        self._cub_hist = []
        self._ceq_hist = []

        # Set the initial models of the problem.
        self._models = Models(self.fun, self.xbase, xl, xu, Aub, bub, Aeq, beq,
                              self.cub, self.ceq, self.options, **kwargs)
        self._target_reached = self._models.target_reached
        if not self.target_reached:
            if self.debug:
                self.check_models()

            # Determine the optimal point so far.
            self._penalty = 0.0
            self._lmlub = np.zeros_like(bub)
            self._lmleq = np.zeros_like(beq)
            self._lmnlub = np.zeros(self.mnlub, dtype=float)
            self._lmnleq = np.zeros(self.mnleq, dtype=float)
            self.kopt = self.get_best_point()
            if self.debug:
                self.check_models()

            # Determine the initial least-squares multipliers.
            self.update_multipliers()

            # The attribute knew contains the index of the interpolation point
            # to be removed from the interpolation set. It is set only during
            # model step. Therefore, if set to None, the current step is a
            # trust-region step.
            self._knew = None

    def __call__(self, x, fx, cubx, ceqx, model=False):
        """
        Evaluate the merit function.

        The merit function is an l2 merit function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the merit function is to be evaluated.
        fx : float
            Value of the objective function at `x`.
        cubx : numpy.ndarray, shape (mnlub,)
            Value of the nonlinear inequality constraint function at `x`.
        ceqx : numpy.ndarray, shape (mnleq,)
            Value of the nonlinear equality constraint function at `x`.
        model : bool, optional
            Whether to also evaluate the merit function on the different models
            (the default is False).

        Returns
        -------
        {float, (float, float)}
            Value of the merit function at `x`, evaluated on the nonlinear
            optimization problem. If ``model = True``, the merit function
            evaluated on the different models is also returned.
        """
        ax = fx
        if self.penalty > 0.0:
            cx = np.r_[
                np.maximum(0.0, np.dot(self.aub, x) - self.bub),
                np.dot(self.aeq, x) - self.beq,
                np.maximum(0.0, cubx),
                ceqx,
            ]
            ax += self.penalty * np.linalg.norm(cx)
        if model:
            # The model of the l2-merit function includes the Hessian of the
            # Lagrangian, not directly of the objective function. See Eq.
            # (15.3.4) of "Trust Region Methods" by Conn, Gould, and Toint for
            # details. In this equation, H must be the Hessian of the
            # Lagrangian, because "Choosing ||.|| = ||.||_1 in (15.3.3) and
            # (15.3.4) gives rise to what is commonly known as the Sl1QP
            # method".
            step = x - self.xopt
            gopt = self.model_obj_grad(self.xopt)
            hstep = self.model_lag_hessp(step)
            mx = self.fopt + np.inner(gopt, step) + 0.5 * np.inner(step, hstep)
            if self.penalty > 0.0:
                aub, bub = self.get_linear_ub()
                aeq, beq = self.get_linear_eq()
                cx = np.r_[
                    np.maximum(0.0, np.dot(aub, x) - bub),
                    np.dot(aeq, x) - beq,
                ]
                mx += self.penalty * np.linalg.norm(cx)
            return ax, mx
        return ax

    def __getattr__(self, item):
        """
        Get options as attributes of the class.

        Parameters
        ----------
        item : str
            Name of the option to retrieve.

        Returns
        -------
        object
            Value of the option.

        Raises
        ------
        AttributeError
            The required option does not exist.
        """
        try:
            return self.options[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @property
    def xbase(self):
        """
        Shift of the origin in the calculations.

        The shift of the origin is designed to tackle numerical difficulties
        caused by ill-conditioned problems.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Shift of the origin in the calculations.
        """
        return self._xbase

    @property
    def ifix(self):
        """
        Indices of the fixed variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Boolean array indicating whether a variable is fixed.
        """
        return self._ifix

    @property
    def xfix(self):
        """
        Values of the fixed variables.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Values of the fixed variables (``m = numpy.count_nonzero(ifix)``).
        """
        return self._xfix

    @property
    def options(self):
        """
        Options forwarded to the solver.

        Returns
        -------
        dict
            Options forwarded to the solver.
        """
        return self._options

    @property
    def penalty(self):
        """
        Penalty coefficient associated with the constraints.

        Returns
        -------
        float
            Penalty coefficient associated with the constraints.
        """
        return self._penalty

    @property
    def lmlub(self):
        """
        Lagrange multipliers associated with the linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub,)
            Lagrange multipliers associated with the linear inequality
            constraints.
        """
        return self._lmlub

    @property
    def lmleq(self):
        """
        Lagrange multipliers associated with the linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq,)
            Lagrange multipliers associated with the linear equality
            constraints.
        """
        return self._lmleq

    @property
    def lmnlub(self):
        """
        Lagrange multipliers associated with the quadratic models of the
        nonlinear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        """
        return self._lmnlub

    @property
    def lmnleq(self):
        """
        Lagrange multipliers associated with the quadratic models of the
        nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.
        """
        return self._lmnleq

    @property
    def knew(self):
        """
        Index of the interpolation point to be removed from the interpolation
        set.

        It is set only during model steps. Therefore, if set to None, the
        current step is a trust-region step.

        Returns
        -------
        int
            Index of the interpolation point to be removed from the
            interpolation set.
        """
        return self._knew

    @property
    def target_reached(self):
        """
        Indicate whether the computations have been stopped because the target
        value has been reached.

        Returns
        -------
        bool
            Flag indicating whether the computations have been stopped because
            the target value has been reached.
        """
        return self._target_reached

    @property
    def xl(self):
        """
        Lower-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Lower-bound constraints on the decision variables.
        """
        return self._models.xl

    @property
    def xu(self):
        """
        Upper-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Upper-bound constraints on the decision variables.
        """
        return self._models.xu

    @property
    def aub(self):
        """
        Jacobian matrix of the normalized linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub, n)
            Jacobian matrix of the normalized linear inequality constraints.
            Each row stores the gradient of a linear inequality constraint.
        """
        return self._models.aub

    @property
    def bub(self):
        """
        Right-hand side vector of the normalized linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub,)
            Right-hand side vector of the normalized linear inequality
            constraints.
        """
        return self._models.bub

    @property
    def mlub(self):
        """
        Number of the linear inequality constraints.

        Returns
        -------
        int
            Number of the linear inequality constraints.
        """
        return self._models.mlub

    @property
    def aeq(self):
        """
        Jacobian matrix of the normalized linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq, n)
            Jacobian matrix of the normalized linear equality constraints. Each
            row stores the gradient of a linear equality constraint.
        """
        return self._models.aeq

    @property
    def beq(self):
        """
        Right-hand side vector of the normalized linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq,)
            Right-hand side vector of the normalized linear equality
            constraints.
        """
        return self._models.beq

    @property
    def mleq(self):
        """
        Number of the linear equality constraints.

        Returns
        -------
        int
            Number of the linear equality constraints.
        """
        return self._models.mleq

    @property
    def xpt(self):
        """
        Displacements of the interpolation points from the origin.

        Returns
        -------
        numpy.ndarray, shape (npt, n)
            Displacements of the interpolation points from the origin. Each row
            stores the displacements of an interpolation point from the origin
            of the calculations.
        """
        return self._models.xpt

    @property
    def fval(self):
        """
        Evaluations of the objective function of the nonlinear optimization
        problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            Evaluations of the objective function of the nonlinear optimization
            problem at the interpolation points.
        """
        return self._models.fval

    @property
    def rval(self):
        """
        Residuals associated with the constraints of the nonlinear optimization
        problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            Residuals associated with the constraints of the nonlinear
            optimization problem at the interpolation points.
        """
        return self._models.rval

    @property
    def cvalub(self):
        """
        Evaluations of the nonlinear inequality constraint function of the
        nonlinear optimization problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt, mnlub)
            Evaluations of the nonlinear inequality constraint function of the
            nonlinear optimization problem at the interpolation points. Each row
            stores the evaluation of the nonlinear inequality constraint
            functions at an interpolation point.
        """
        return self._models.cvalub

    @property
    def mnlub(self):
        """
        Number of the nonlinear inequality constraints.

        Returns
        -------
        int
            Number of the nonlinear inequality constraints.
        """
        return self._models.mnlub

    @property
    def cvaleq(self):
        """
        Evaluations of the nonlinear equality constraint function of the
        nonlinear optimization problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt, mnleq)
            Evaluations of the nonlinear equality constraint function of the
            nonlinear optimization problem at the interpolation points. Each row
            stores the evaluation of the nonlinear equality constraint functions
            at an interpolation point.
        """
        return self._models.cvaleq

    @property
    def mnleq(self):
        """
        Number of the nonlinear equality constraints.

        Returns
        -------
        int
            Number of the nonlinear equality constraints.
        """
        return self._models.mnleq

    @property
    def kopt(self):
        """
        Index of the best interpolation point so far, corresponding to the point
        around which the Taylor expansions of the quadratic models are defined.

        Returns
        -------
        int
            Index of the best interpolation point so far, corresponding to the
            point around which the Taylor expansions of the quadratic models are
            defined.
        """
        return self._models.kopt

    @kopt.setter
    def kopt(self, knew):
        """
        Index of the best interpolation point so far, corresponding to the point
        around which the Taylor expansions of the quadratic models are defined.

        Parameters
        ----------
        knew : int
            New index of the best interpolation point so far, which hereinafter
            corresponds to the point around which the Taylor expansions of the
            quadratic models are defined.
        """
        self._models.kopt = knew

    @property
    def xopt(self):
        """
        Best interpolation point so far, corresponding to the point around which
        the Taylor expansion of the quadratic models are defined.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Best interpolation point so far, corresponding to the point around
            which the Taylor expansion of the quadratic models are defined.
        """
        return self._models.xopt

    @property
    def fopt(self):
        """
        Evaluation of the objective function of the nonlinear optimization
        problem at `xopt`.

        Returns
        -------
        float
            Evaluation of the objective function of the nonlinear optimization
            problem at `xopt`.
        """
        return self._models.fopt

    @property
    def maxcv(self):
        """
        Constraint violation evaluated on the nonlinear optimization problem at
        t`xopt`.

        Returns
        -------
        float
            Constraint violation evaluated on the nonlinear optimization
            problem at `xopt`.
        """
        return self._models.ropt

    @property
    def coptub(self):
        """
        Evaluation of the nonlinear inequality constraint function of the
        nonlinear optimization problem at `xopt`.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Evaluation of the nonlinear inequality constraint function of the
            nonlinear optimization problem at `xopt`.
        """
        return self._models.coptub

    @property
    def copteq(self):
        """
        Evaluation of the nonlinear equality constraint function of the
        nonlinear optimization problem at `xopt`.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Evaluation of the nonlinear equality constraint function of the
            nonlinear optimization problem at `xopt`.
        """
        return self._models.copteq

    @property
    def x_hist(self):
        """
        History of the decision variables considered.

        Returns
        -------
        numpy.ndarray, shape (-1, n)
            History of the decision variables considered.
        """
        x_hist = np.array(self._x_hist, dtype=float)
        return np.reshape(x_hist, (-1, self.xopt.size))

    @property
    def fun_hist(self):
        """
        History of the objective function values.

        Returns
        -------
        numpy.ndarray, shape (-1, n)
            History of the objective function values.
        """
        return np.array(self._fun_hist, dtype=float)

    @property
    def cub_hist(self):
        """
        History of the nonlinear inequality constraint function values.

        Returns
        -------
        numpy.ndarray, shape (-1, n)
            History of the nonlinear inequality constraint function values.
        """
        cub_hist = np.array(self._cub_hist, dtype=float)
        return np.reshape(cub_hist, (-1, max(self.mnlub, 1)))

    @property
    def ceq_hist(self):
        """
        History of the nonlinear equality constraint function values.

        Returns
        -------
        numpy.ndarray, shape (-1, n)
            History of the nonlinear equality constraint function values.
        """
        ceq_hist = np.array(self._ceq_hist, dtype=float)
        return np.reshape(ceq_hist, (-1, max(self.mnleq, 1)))

    @property
    def type(self):
        """
        Type of the nonlinear optimization problem.

        It follows the CUTEst classification scheme for the constraint types
        (see https://www.cuter.rl.ac.uk/Problems/classification.shtml).

        Returns
        -------
        str
            Type of the nonlinear optimization problem:
                - U : the problem is unconstrained.
                - B : the problem's only constraints are bounds constraints.
                - L : the problem's constraints are linear.
                - O : the problem's constraints general.
        """
        return self._models.type

    @property
    def is_model_step(self):
        """
        Flag indicating whether the current step is a model step.

        Returns
        -------
        bool
            Flag indicating whether the current step is a model step.
        """
        return self.knew is not None

    def fun(self, x, **kwargs):
        """
        Evaluate the objective function of the nonlinear optimization problem.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the objective function is to be evaluated.

        Returns
        -------
        float
            Value of the objective function of the nonlinear optimization
            problem at `x`.

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).
        """
        x_full = self.get_x(x)
        fx = float(self._fun(x_full, *self._args))
        threshold = huge(x_full.dtype)
        if np.isnan(fx) or fx > threshold:
            fx = threshold
        fx = np.nan_to_num(fx)
        if kwargs.get('store_history'):
            self._x_hist.append(np.copy(x))
            self._fun_hist.append(fx)
        if self.disp:
            print(f'{self._fun.__name__}({x_full}) = {fx}.')
        return fx

    def cub(self, x, **kwargs):
        """
        Evaluate the nonlinear inequality constraint function of the nonlinear
        optimization problem.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint function is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Value of the nonlinear inequality constraint function of the
            nonlinear optimization problem at `x`.

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).
        """
        cx = self._eval_con(self._cub, x)
        if kwargs.get('store_history') and cx.size > 0:
            self._cub_hist.append(np.copy(cx))
        return cx

    def ceq(self, x, **kwargs):
        """
        Evaluate the nonlinear equality constraint function of the nonlinear
        optimization problem.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraint function is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnleq,)
            Value of the nonlinear equality constraint function of the
            nonlinear optimization problem at `x`.

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).
        """
        cx = self._eval_con(self._ceq, x)
        if kwargs.get('store_history') and cx.size > 0:
            self._ceq_hist.append(np.copy(cx))
        return cx

    def active_set(self, x):
        """
        Determine the set of active constraints of the models.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            The point at which the constraints of the models are to be
            evaluated.

        Returns
        -------
        numpy.ndarray
            Indices of the active constraints of the models.
        """
        bdtol = 10.0 * np.finfo(float).eps * self.xopt.size
        bdtol *= absmax_arrays(self.xl, self.xu, initial=1.0)
        aub, bub = self.get_linear_ub()
        resid = np.r_[np.dot(aub, x) - bub, self.xl - x, x - self.xu]
        iact = np.flatnonzero(np.abs(resid) <= bdtol)
        return iact

    def get_x(self, x):
        """
        Build the full decision variables.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            The reduced vector of decision variables.

        Returns
        -------
        numpy.ndarray, shape (n + m,)
            All decision variables, included the fixed ones.
        """
        x_full = np.zeros(self.ifix.size, dtype=float)
        x_full[self.ifix] = self.xfix
        x_full[np.logical_not(self.ifix)] = x
        return x_full

    def get_linear_ub(self, x=None):
        """
        Get the linear inequality constraints of the models.

        The linear inequality constraints of the models start with the linear
        inequality constraints of the original problem, followed by the linear
        Taylor-like expansion of the nonlinear inequality constraints.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point around which to expand the Taylor models (the default is the
            best point so far).

        Returns
        -------
        numpy.ndarray, shape (m, n)
            Jacobian matrix of the linear inequality constraints.
        numpy.ndarray, shape (m,)
            Right-hand side of the linear inequality constraints.
        """
        aub = np.copy(self.aub)
        bub = np.copy(self.bub)
        if x is None:
            x = self.xopt
        for i in range(self.mnlub):
            lhs = self.model_cub_grad(x, i)
            rhs = np.inner(x, lhs) - self.coptub[i]
            aub = np.vstack([aub, lhs])
            bub = np.r_[bub, rhs]
        return aub, bub

    def get_linear_eq(self, x=None):
        """
        Get the linear equality constraints of the models.

        The linear equality constraints of the models start with the linear
        equality constraints of the original problem, followed by the linear
        Taylor-like expansion of the nonlinear equality constraints.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point around which to expand the Taylor models (the default is the
            best point so far).

        Returns
        -------
        numpy.ndarray, shape (m, n)
            Jacobian matrix of the linear equality constraints.
        numpy.ndarray, shape (m,)
            Right-hand side of the linear equality constraints.
        """
        aeq = np.copy(self.aeq)
        beq = np.copy(self.beq)
        if x is None:
            x = self.xopt
        for i in range(self.mnleq):
            lhs = self.model_ceq_grad(x, i)
            rhs = np.inner(x, lhs) - self.copteq[i]
            aeq = np.vstack([aeq, lhs])
            beq = np.r_[beq, rhs]
        return aeq, beq

    def model_obj(self, x):
        """
        Evaluate the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the objective function of the model at `x`.
        """
        return self._models.obj(x)

    def model_obj_grad(self, x):
        """
        Evaluate the gradient of the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the objective function of the model at `x`.
        """
        return self._models.obj_grad(x)

    def model_obj_hess(self):
        """
        Evaluate the Hessian matrix of the objective function of the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the objective function of the model.
        """
        return self._models.obj_hess()

    def model_obj_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the objective function of
        the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the objective function
            of the model with the vector `x`.
        """
        return self._models.obj_hessp(x)

    def model_obj_curv(self, x):
        """
        Evaluate the curvature of the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the objective function of the model at `x`.
        """
        return self._models.obj_curv(x)

    def model_obj_alt(self, x):
        """
        Evaluate the alternative objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the alternative objective function of the model at `x`.
        """
        return self._models.obj_alt(x)

    def model_obj_alt_grad(self, x):
        """
        Evaluate the gradient of the alternative objective function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the alternative objective function of the model at `x`.
        """
        return self._models.obj_alt_grad(x)

    def model_obj_alt_hess(self):
        """
        Evaluate the Hessian matrix of the alternative objective function of the
        model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the alternative objective function of the model.
        """
        return self._models.obj_alt_hess()

    def model_obj_alt_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the alternative objective
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the alternative
            objective function of the model with the vector `x`.
        """
        return self._models.obj_alt_hessp(x)

    def model_obj_alt_curv(self, x):
        """
        Evaluate the curvature of the alternative objective function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the alternative objective function of the model at `x`.
        """
        return self._models.obj_alt_curv(x)

    def model_cub(self, x, i):
        """
        Evaluate an inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th inequality constraint function of the model at
            `x`.
        """
        return self._models.cub(x, i)

    def model_cub_grad(self, x, i):
        """
        Evaluate the gradient of an inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th inequality constraint function of the model
            at `x`.
        """
        return self._models.cub_grad(x, i)

    def model_cub_hess(self, i):
        """
        Evaluate the Hessian matrix of an inequality constraint function of the
        model.

        Parameters
        ----------
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th inequality constraint function of the
            model.
        """
        return self._models.cub_hess(i)

    def model_cub_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an inequality constraint
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th inequality
            constraint function of the model with the vector `x`.
        """
        return self._models.cub_hessp(x, i)

    def model_cub_curv(self, x, i):
        """
        Evaluate the curvature of an inequality constraint function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th inequality constraint function of the model
            at `x`.
        """
        return self._models.cub_curv(x, i)

    def model_cub_alt(self, x, i):
        """
        Evaluate an alternative inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th alternative inequality constraint function of
            the model at `x`.
        """
        return self._models.cub_alt(x, i)

    def model_cub_alt_grad(self, x, i):
        """
        Evaluate the gradient of an alternative inequality constraint function
        of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th alternative inequality constraint function of
            the model at `x`.
        """
        return self._models.cub_alt_grad(x, i)

    def model_cub_alt_hess(self, i):
        """
        Evaluate the Hessian matrix of an alternative inequality constraint
        function of the model.

        Parameters
        ----------
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th alternative inequality constraint
            function of the model.
        """
        return self._models.cub_alt_hess(i)

    def model_cub_alt_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an alternative inequality
        constraint function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th alternative
            inequality constraint function of the model with the vector `x`.
        """
        return self._models.cub_alt_hessp(x, i)

    def model_cub_alt_curv(self, x, i):
        """
        Evaluate the curvature of an alternative inequality constraint function
        of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th alternative inequality constraint function
            of the model at `x`.
        """
        return self._models.cub_alt_curv(x, i)

    def model_ceq(self, x, i):
        """
        Evaluate an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self._models.ceq(x, i)

    def model_ceq_grad(self, x, i):
        """
        Evaluate the gradient of an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self._models.ceq_grad(x, i)

    def model_ceq_hess(self, i):
        """
        Evaluate the Hessian matrix of an equality constraint function of the
        model.

        Parameters
        ----------
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th equality constraint function of the
            model.
        """
        return self._models.ceq_hess(i)

    def model_ceq_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an equality constraint
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th equality
            constraint function of the model with the vector `x`.
        """
        return self._models.ceq_hessp(x, i)

    def model_ceq_curv(self, x, i):
        """
        Evaluate the curvature of an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self._models.ceq_curv(x, i)

    def model_ceq_alt(self, x, i):
        """
        Evaluate an alternative equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th alternative equality constraint function of the
            model at `x`.
        """
        return self._models.ceq_alt(x, i)

    def model_ceq_alt_grad(self, x, i):
        """
        Evaluate the gradient of an alternative equality constraint function of
        the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th alternative equality constraint function of
            the model at `x`.
        """
        return self._models.ceq_alt_grad(x, i)

    def model_ceq_alt_hess(self, i):
        """
        Evaluate the Hessian matrix of an alternative equality constraint
        function of the model.

        Parameters
        ----------
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th alternative equality constraint
            function of the model.
        """
        return self._models.ceq_alt_hess(i)

    def model_ceq_alt_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an alternative equality
        constraint function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th alternative
            equality constraint function of the model with the vector `x`.
        """
        return self._models.ceq_alt_hessp(x, i)

    def model_ceq_alt_curv(self, x, i):
        """
        Evaluate the curvature of an alternative equality constraint function of
        the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th alternative equality constraint function of
            the model at `x`.
        """
        return self._models.ceq_alt_curv(x, i)

    def model_lag(self, x):
        """
        Evaluate the Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the Lagrangian function of the model at `x`.
        """
        return self._models.lag(x, self.lmlub, self.lmleq, self.lmnlub,
                                self.lmnleq)

    def model_lag_grad(self, x):
        """
        Evaluate the gradient of Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the Lagrangian function of the model at `x`.
        """
        return self._models.lag_grad(x, self.lmlub, self.lmleq, self.lmnlub,
                                     self.lmnleq)

    def model_lag_hess(self):
        """
        Evaluate the Hessian matrix of the Lagrangian function of the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the Lagrangian function of the model.
        """
        return self._models.lag_hess(self.lmnlub, self.lmnleq)

    def model_lag_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the Lagrangian function of
        the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the Lagrangian
            function of the model with the vector `x`.
        """
        return self._models.lag_hessp(x, self.lmnlub, self.lmnleq)

    def model_lag_curv(self, x):
        """
        Evaluate the curvature of the Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the Lagrangian function of the model at `x`.
        """
        return self._models.lag_curv(x, self.lmnlub, self.lmnleq)

    def model_lag_alt(self, x):
        """
        Evaluate the alternative Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the alternative Lagrangian function of the model at `x`.
        """
        return self._models.lag_alt(x, self.lmlub, self.lmleq, self.lmnlub,
                                    self.lmnleq)

    def model_lag_alt_grad(self, x):
        """
        Evaluate the gradient of the alternative Lagrangian function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the alternative Lagrangian function of the model at `x`.
        """
        return self._models.lag_alt_grad(x, self.lmlub, self.lmleq, self.lmnlub,
                                         self.lmnleq)

    def model_lag_alt_hess(self):
        """
        Evaluate the Hessian matrix of the alternative Lagrangian function of
        the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the alternative Lagrangian function of the model.
        """
        return self._models.lag_alt_hess(self.lmnlub, self.lmnleq)

    def model_lag_alt_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the alternative Lagrangian
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the alternative
            Lagrangian function of the model with the vector `x`.
        """
        return self._models.lag_alt_hessp(x, self.lmnlub, self.lmnleq)

    def model_lag_alt_curv(self, x):
        """
        Evaluate the curvature of the alternative Lagrangian function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the alternative Lagrangian function of the model at
            `x`.
        """
        return self._models.lag_alt_curv(x, self.lmnlub, self.lmnleq)

    def set_default_options(self, n):
        """
        Set the default options for the solvers.

        Parameters
        ----------
        n : int
            Number of decision variables.
        """
        rhoend = getattr(self, 'rhoend', 1e-6)
        self.options.setdefault('rhobeg', max(1.0, rhoend))
        self.options.setdefault('rhoend', min(rhoend, self.rhobeg))
        self.options.setdefault('npt', 2 * n + 1)
        self.options.setdefault('maxfev', max(500 * n, self.npt + 1))
        self.options.setdefault('maxiter', 1000 * n)
        self.options.setdefault('target', -np.inf)
        self.options.setdefault('ftol_abs', -1.0)
        self.options.setdefault('ftol_rel', -1.0)
        self.options.setdefault('xtol_abs', -1.0)
        self.options.setdefault('xtol_rel', -1.0)
        self.options.setdefault('disp', False)
        self.options.setdefault('respect_bounds', True)
        self.options.setdefault('debug', False)

    def check_options(self, n, stack_level=2):
        """
        Ensure that the options are consistent, and modify them if necessary.

        Parameters
        ----------
        n : int
            Number of decision variables.
        stack_level : int, optional
            Stack level of the warning (the default is 2).

        Warns
        -----
        RuntimeWarning
            The options are inconsistent and modified.
        """
        # Ensure that the option 'npt' is in the required interval.
        npt_max = (n + 1) * (n + 2) // 2
        npt_min = min(n + 2, npt_max)
        npt = self.npt
        if not (npt_min <= npt <= npt_max):
            self.options['npt'] = min(npt_max, max(npt_min, npt))
            message = "option 'npt' is not in the required interval and is "
            message += 'increased.' if npt_min > npt else 'decreased.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the option 'maxfev' is large enough.
        maxfev = self.maxfev
        if maxfev <= self.npt:
            self.options['maxfev'] = self.npt + 1
            if maxfev <= npt:
                message = "option 'maxfev' is too low and is increased."
            else:
                message = "option 'maxfev' is correspondingly increased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the options 'rhobeg' and 'rhoend' are consistent.
        if self.rhoend > self.rhobeg:
            self.options['rhoend'] = self.rhobeg
            message = "option 'rhoend' is too large and is decreased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

    def get_best_point(self):
        """
        Get the index of the optimal interpolation point.

        Returns
        -------
        int
            Index of the optimal interpolation point.
        """
        kopt = self.kopt
        mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        for k in range(self.npt):
            if k != self.kopt:
                mval = self(self.xpt[k, :], self.fval[k], self.cvalub[k, :],
                            self.cvaleq[k, :])
                if self.less_merit(mval, self.rval[k], mopt, self.rval[kopt]):
                    kopt = k
                    mopt = mval
        return kopt

    def prepare_trust_region_step(self):
        """
        Set the next iteration to a trust-region step.
        """
        self._knew = None

    def prepare_model_step(self, delta):
        """
        Set the next iteration to a model-step if necessary.

        The method checks whether the furthest interpolation point from
        `xopt` is more than the provided trust-region radius to set a
        model-step. If such a point does not exist, the next iteration is a
        trust-region step.

        Parameters
        ----------
        delta : float
            Trust-region radius.
        """
        dsq = np.sum((self.xpt - self.xopt[np.newaxis, :]) ** 2.0, axis=1)
        dsq[dsq <= delta ** 2.0] = -np.inf
        if np.any(np.isfinite(dsq)):
            self._knew = np.argmax(dsq)
        else:
            self._knew = None

    def less_merit(self, mval1, rval1, mval2, rval2):
        """
        Indicates whether a point is better than another.

        Parameters
        ----------
        mval1 : float
            Merit value associated with the first point.
        rval1 : float
            Residual value associated with the first point.
        mval2 : float
            Merit value associated with the second point.
        rval2 : float
            Residual value associated with the second point.

        Returns
        -------
        bool
            A flag indicating whether the first point is better than the other.
        """
        eps = np.finfo(float).eps
        tol = 10.0 * eps * self.npt * max(1.0, abs(mval2))
        if mval1 < mval2:
            return True
        elif self.penalty < tol:
            # If the penalty coefficient is zero and if the merit values are
            # equal, the optimality of the points is decided based on their
            # residual values.
            if abs(mval1 - mval2) <= tol and rval1 < rval2:
                return True
        return False

    def shift_origin(self, delta):
        """
        Shift the origin of the calculations if necessary.

        Although the shift of the origin in the calculations does not change
        anything from a theoretical point of view, it is designed to tackle
        numerical difficulties caused by ill-conditioned problems. If the method
        is triggered, the origin is shifted to the best point so far.

        Parameters
        ----------
        delta : float
            Trust-region radius.
        """
        xoptsq = np.inner(self.xopt, self.xopt)

        # Update the shift from the origin only if the displacement from the
        # shift of the best point is substantial in the trust region.
        if xoptsq >= 10.0 * delta ** 2.0:
            # Update the models of the problem to include the new shift.
            self._xbase += self.xopt
            self._models.shift_origin()
            if self.debug:
                self.check_models()

    def update(self, nstep, tstep, delta, **kwargs):
        """
        Include a new point in the interpolation set.

        When the new point is included in the interpolation set, the models of
        the nonlinear optimization problems are updated.

        Parameters
        ----------
        nstep : numpy.ndarray, shape (n,)
            Normal step from `xopt` of the new point to include in the
            interpolation set.
        tstep: numpy.ndarray, shape (n,)
            Tangential step from `xopt + nstep` of the new point to include in
            the interpolation set.
        delta : float
            Trust-region radius.

        Returns
        -------
        float:
            Objective function evaluation of the trial point.
        float
            Merit value of the new interpolation point.
        float
            Trust-region ratio associated with the new interpolation point.

        Other Parameters
        ----------------
        penalty_detection_factor : float, optional
            Factor on the penalty coefficient used to decide whether it should
            be increased (the default is 1.5).
        penalty_growth_factor : float, optional
            Increasing factor on the penalty coefficient (the default is 2).
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).

        Raises
        ------
        RestartRequiredException
            The iteration must be restarted because the index of the optimal
            point among the interpolation set has changed.
        """
        # Evaluate the objective function, the nonlinear inequality constraint
        # function, and the nonlinear equality constraint function at the trial
        # point. The functions are defined in the space centered at the origin.
        bdtol = 10.0 * np.finfo(float).eps * self.xopt.size
        bdtol *= absmax_arrays(self.xl, self.xu, initial=1.0)
        step = nstep + tstep
        xnew = self.xopt + step
        fx = self.fun(self.xbase + xnew, **kwargs)
        cubx = self.cub(self.xbase + xnew, **kwargs)
        ceqx = self.ceq(self.xbase + xnew, **kwargs)
        rx = self._models.resid(xnew, cubx, ceqx)
        self._target_reached = fx <= self.target and rx <= bdtol

        # Update the Lagrange multipliers and the penalty parameters for the
        # trust-region ratio to be well-defined.
        tiny = np.finfo(float).tiny
        is_trust_region_step = not self.is_model_step
        if not self.target_reached:
            ksav = self.kopt
            mx, mmx, mopt = self.increase_penalty(
                nstep, tstep, fx, cubx, ceqx, **kwargs)
            if ksav != self.kopt:
                # When increasing the penalty parameters is required to make the
                # trust-region ratio meaningful, the index of the optimal point
                # changed. A new trust-region iteration has to be entertained.
                self.prepare_trust_region_step()
                raise RestartRequiredException

            # Determine the trust-region ratio.
            if is_trust_region_step and abs(mopt - mmx) > tiny * abs(mopt - mx):
                ratio = (mopt - mx) / (mopt - mmx)
            else:
                ratio = -1.0

            # Update the least-squares Lagrange multipliers.
            # If we were solving the SQP subproblem as: (1) we find d, and (2)
            # we find lambda, then the following code should be used. However,
            # in practice, it worsen a lot the performance.
            # TODO: Find why (where?).
            # TODO: Which gradients should be used? Updated models?
            # TODO: How to formulate the complementary slackness condition?
            # if ratio >= 0.0:
            #     self.update_multipliers(step)
            # else:
            #     self.update_multipliers(np.zeros_like(step))
            self.update_multipliers()
        else:
            mx = self(xnew, fx, cubx, ceqx)
            mopt = mx
            ratio = -1.0

        # Update the models of the problem and perform a second-order correction
        # step if possible whenever the trust-region ratio is too low.
        self._knew = self._models.update(step, fx, cubx, ceqx, self.knew)
        if self.target_reached or self.less_merit(mx, rx, mopt, self.maxcv):
            self.kopt = self.knew
            mopt = mx
        elif is_trust_region_step and self.type == 'O':
            nssq = np.inner(nstep, nstep)
            xi = kwargs.get('normal_step_shrinkage_factor')
            if nssq <= xi ** 4.0 * delta ** 2.0:
                ssoc = self.soc_step(step, **kwargs)
                if np.inner(ssoc, ssoc) > 0.0:
                    ssoc += step
                    xsoc = self.xopt + ssoc
                    fxs = self.fun(self.xbase + xsoc, **kwargs)
                    cubx = self.cub(self.xbase + xsoc, **kwargs)
                    ceqx = self.ceq(self.xbase + xsoc, **kwargs)
                    mx, mmx = self(xsoc, fxs, cubx, ceqx, True)
                    rx = self._models.resid(xsoc, cubx, ceqx)
                    if self.less_merit(mx, rx, mopt, self.maxcv):
                        if abs(mopt - mmx) > tiny * abs(mopt - mx):
                            fx = fxs
                            ratio = (mopt - mx) / (mopt - mmx)
                            mopt = mx
                            self._knew = self._models.update(
                                ssoc, fx, cubx, ceqx, self.knew)
                            self.kopt = self.knew
        if not self.target_reached and self.debug:
            self.check_models()
        return fx, mopt, ratio

    def update_multipliers(self):
        """
        Set the least-squares Lagrange multipliers.
        """
        n = self.xopt.size
        if self.mlub + self.mleq + self.mnlub + self.mnleq > 0:
            # Determine the matrix of the least-squares problem. The Lagrange
            # multipliers corresponding to nonzero inequality constraint values
            # are zeroed to satisfy the complementary slackness conditions.
            ilub = np.dot(self.aub, self.xopt) >= self.bub
            mlub = np.count_nonzero(ilub)
            inlub = self.coptub >= 0.0
            mnlub = np.count_nonzero(inlub)
            cub_jac = np.empty((mnlub, n), dtype=float)
            for i, j in enumerate(np.flatnonzero(inlub)):
                cub_jac[i, :] = self.model_cub_grad(self.xopt, j)
            ceq_jac = np.empty((self.mnleq, n), dtype=float)
            for i in range(self.mnleq):
                ceq_jac[i, :] = self.model_ceq_grad(self.xopt, i)
            ixl = self.xl >= self.xopt
            ixu = self.xu <= self.xopt
            nxl = np.count_nonzero(ixl)
            nxu = np.count_nonzero(ixu)
            identity = np.eye(n)
            A = np.r_[
                -identity[ixl, :],
                identity[ixu, :],
                self.aub[ilub, :],
                cub_jac,
                self.aeq,
                ceq_jac,
            ].T

            # Determine the least-squares Lagrange multipliers that have not
            # been fixed by the complementary slackness conditions.
            gopt = self.model_obj_grad(self.xopt)
            shift = nxl + nxu
            lm = nnls(A, -gopt, shift + mlub + mnlub)
            self.lmlub.fill(0.0)
            self.lmnlub.fill(0.0)
            self.lmlub[ilub] = lm[shift:shift + mlub]
            self.lmnlub[inlub] = lm[shift + mlub:shift + mlub + mnlub]
            self.lmleq[:] = lm[shift + mlub + mnlub:shift + mlub + mnlub + self.mleq]
            self.lmnleq[:] = lm[shift + mlub + mnlub + self.mleq:]

    def increase_penalty(self, nstep, tstep, fx, cubx, ceqx, **kwargs):
        """
        Increase the penalty coefficients.

        The penalty coefficients are increased to make the trust-region ratio
        meaningful. The increasing process of the penalty coefficients may be
        prematurely stop if the index of the best point so far changes.

        Parameters
        ----------
        nstep : numpy.ndarray, shape (n,)
            Normal step from `xopt` of the new point to include in the
            interpolation set.
        tstep: numpy.ndarray, shape (n,)
            Tangential step from `xopt + nstep` of the new point to include in
            the interpolation set.

        fx : float
            Value of the objective function at the trial point.
        cubx : numpy.ndarray, shape (mnlub,)
            Value of the nonlinear inequality constraint function at the trial
            point.
        ceqx : numpy.ndarray, shape (mnleq,)
            Value of the nonlinear equality constraint function at the trial
            point.

        Returns
        -------
        float
            Value of the merit function at the trial point, evaluated on the
            nonlinear optimization problem.
        float
            Value of the merit function at the trial point, evaluated on the
            different models.
        float
            Value of the merit function at `xopt`, evaluated on the nonlinear
            optimization problem.

        Other Parameters
        ----------------
        penalty_detection_factor : float, optional
            Factor on the penalty coefficient used to decide whether it should
            be increased (the default is 1.5).
        penalty_growth_factor : float, optional
            Increasing factor on the penalty coefficient (the default is 2).
        """
        tiny = np.finfo(float).tiny
        step = nstep + tstep
        xnew = self.xopt + step
        mx, mmx = self(xnew, fx, cubx, ceqx, True)
        mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        if self.type not in 'UB' and not self.is_model_step:
            gopt = self.model_obj_grad(self.xopt)
            hstep = self.model_lag_hessp(step)
            reduct = np.inner(gopt, step) + 0.5 * np.inner(step, hstep)
            aub, bub = self.get_linear_ub()
            aeq, beq = self.get_linear_eq()
            bub -= np.dot(aub, self.xopt)
            beq -= np.dot(aeq, self.xopt)
            resid = np.r_[np.maximum(0.0, -bub), beq]
            violation = np.linalg.norm(resid)
            resid = np.r_[
                np.maximum(0.0, np.dot(aub, nstep) - bub),
                np.dot(aeq, nstep) - beq,
            ]
            violation -= np.linalg.norm(resid)
            lm = np.r_[self.lmlub, self.lmleq, self.lmnlub, self.lmnleq]
            thold = np.linalg.norm(lm)
            if violation > tiny * abs(reduct):
                thold = max(thold, reduct / violation)
            if self.penalty < kwargs.get('penalty_detection_factor') * thold:
                self._penalty = kwargs.get('penalty_growth_factor') * thold
                mx, mmx = self(xnew, fx, cubx, ceqx, True)
                self.kopt = self.get_best_point()
                mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        return mx, mmx, mopt

    def reduce_penalty(self):
        """
        Reduce the penalty coefficients if possible, to prevent overflows.

        Notes
        -----
        The thresholds at which the penalty coefficients are set are empirical
        and based on Equation (13) of [1]_.

        References
        ----------
        .. [1] M. J. D. Powell. "A direct search optimization method that models
           the objective and constraint functions by linear interpolation." In:
           Advances in Optimization and Numerical Analysis. Ed. by S. Gomez and
           J. P. Hennart. Dordrecht, NL: Springer, 1994, pp. 51--67.
        """
        tiny = np.finfo(float).tiny
        fmin = np.min(self.fval)
        fmax = np.max(self.fval)
        if self.penalty > 0.0:
            rlub = np.matmul(self.xpt, self.aub.T) - self.bub[np.newaxis, :]
            rleq = np.matmul(self.xpt, self.aeq.T) - self.beq[np.newaxis, :]
            rub = np.c_[rlub, self.cvalub]
            req = np.c_[rleq, -rleq, self.cvaleq, -self.cvaleq]
            resid = np.c_[rub, req]
            cmin = np.min(resid, axis=0)
            cmax = np.max(resid, axis=0)
            indices = cmin < 2.0 * cmax
            if np.any(indices):
                cmin_neg = np.minimum(0.0, cmin[indices])
                denom = np.min(cmax[indices] - cmin_neg)
                if denom > tiny * (fmax - fmin):
                    self._penalty = min(self.penalty, (fmax - fmin) / denom)
            else:
                self._penalty = 0.0

    def trust_region_step(self, delta, **kwargs):
        """
        Evaluate a Byrd-Omojokun-like trust-region step from `xopt`.

        Parameters
        ----------
        delta : float
            Trust-region radius.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Trust-region step from `xopt`.

        Other Parameters
        ----------------
        improve_tcg : bool, optional
            Whether to improve the truncated conjugate gradient step round the
            trust-region boundary (the default is True).

        Notes
        -----
        The trust-region constraint of the tangential subproblem is not centered
        if the normal step is nonzero. To cope with this difficulty, we use the
        result presented in Equation (15.4.3) of [1]_.

        References
        ----------
        .. [1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint. Trust-Region
           Methods. MPS-SIAM Ser. Optim. Philadelphia, PA, US: SIAM, 2009.
        """
        kwargs = dict(kwargs)
        kwargs['debug'] = self.debug

        # Evaluate the linear approximations of the inequality and equality
        # constraints around the best point so far.
        delsav = delta
        if self.type in 'LO':
            delta *= np.sqrt(0.5)
        aub, bub = self.get_linear_ub()
        aeq, beq = self.get_linear_eq()

        # Evaluate the normal step of the Byrd-Omojokun approach. The normal
        # step attempts to reduce the violations of the linear constraints
        # subject to the bound constraints and a trust-region constraint. The
        # trust-region radius is shrunk to leave some elbow room to the
        # tangential subproblem for the computations whenever the trust-region
        # subproblem is infeasible.
        if self.type in 'UB':
            nstep = np.zeros_like(self.xopt)
            ssq = 0.0
        else:
            xi = kwargs.get('normal_step_shrinkage_factor')

            def normal_obj(x):
                rub = np.maximum(0.0, np.dot(aub, x) - bub)
                req = np.dot(aeq, x) - beq
                fx = 0.5 * (np.inner(rub, rub) + np.inner(req, req))
                gx = np.dot(aub.T, rub) + np.dot(aeq.T, req)
                return fx, gx

            def ball(x):
                return np.linalg.norm(x - self.xopt) - xi * delta

            if kwargs.get('exact_normal_step'):
                bounds = Bounds(self.xl, self.xu)
                constraints = NonlinearConstraint(ball, -np.inf, 0.0)
                options = {'ftol': max(np.finfo(float).eps, 1e-8 * delta)}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = scipy_minimize(normal_obj, self.xopt, method='slsqp',
                                         jac=True, bounds=bounds,
                                         constraints=constraints,  # noqa
                                         options=options)
                nstep = res.x - self.xopt
            else:
                nstep = cpqp(self.xopt, aub, bub, aeq, beq, self.xl, self.xu,
                             xi * delta, **kwargs)
            ssq = np.inner(nstep, nstep)
            if self.debug:
                tol = 10.0 * np.finfo(float).eps * self.xopt.size
                assert_array_less(self.xl - self.xopt - nstep,
                                  tol * absmax_arrays(self.xl, initial=1.0))
                assert_array_less(self.xopt + nstep - self.xu,
                                  tol * absmax_arrays(self.xl, initial=1.0))
                assert_(ssq <= delta ** 2.0)

        # Evaluate the tangential step of the trust-region subproblem, and set
        # the global trust-region step. The tangential step attempts to reduce
        # the objective function of the model without worsening the constraint
        # violation provided by the normal step.
        delta = np.sqrt(delta ** 2.0 - ssq)
        xopt = self.xopt + nstep
        bub = np.maximum(bub, np.dot(aub, xopt))
        beq = np.dot(aeq, xopt)
        gopt = self.model_obj_grad(self.xopt) + self.model_lag_hessp(nstep)
        if self.type in 'UB':
            tstep = bvtcg(xopt, gopt, self.model_obj_hessp, self.xl, self.xu,
                          delta, **kwargs)
        else:
            tstep = lctcg(xopt, gopt, self.model_lag_hessp, aub, bub, aeq, beq,
                          self.xl, self.xu, delta, **kwargs)
        if self.debug:
            tol = 10.0 * np.finfo(float).eps * self.xopt.size
            assert_array_less(self.xl - xopt - tstep,
                              tol * absmax_arrays(self.xl, initial=1.0))
            assert_array_less(xopt + tstep - self.xu,
                              tol * absmax_arrays(self.xu, initial=1.0))
            assert_(np.linalg.norm(nstep + tstep) <= 1.1 * delsav)
            reduct = np.inner(gopt, tstep)
            reduct += 0.5 * np.inner(tstep, self.model_lag_hessp(tstep))
            assert_(reduct <= 0.0)
        return nstep, tstep

    def model_step(self, delta, **kwargs):
        """
        Estimate a model-improvement step from `xopt`.

        Parameters
        ----------
        delta : float
            Trust-region radius.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Model-improvement step from `xopt`.

        Other Parameters
        ----------------
        debug : bool, optional
            Whether to make debugging tests during the execution, which is
            not recommended in production (the default is False).
        improve_tcg : bool, optional
            Whether to improve the truncated conjugate gradient step round the
            trust-region boundary (the default is True).

        Notes
        -----
        Two alternative steps are computed.

            1. The first alternative step is selected on the lines that join
               `xopt` to the other interpolation points that maximize a lower
               bound on the denominator of the updating formula.
            2. The second alternative is a constrained Cauchy step.

        Among the two alternative steps, the method selects the one that leads
        to the greatest denominator in Equation (2.12) of [1]_.

        References
        ----------
        .. [1] M. J. D. Powell. "On updating the inverse of a KKT matrix." In:
           Numerical Linear Algebra and Optimization. Ed. by Y. Yuan. Beijing,
           CN: Science Press, 2004, pp. 56--78.
        """
        kwargs = dict(kwargs)
        kwargs['debug'] = self.debug
        aub, bub = self.get_linear_ub()
        aeq, beq = self.get_linear_eq()
        step = self._models.improve_geometry(self.knew, delta, aub, bub ,aeq,
                                             beq, **kwargs)
        if self.debug:
            tol = 10.0 * np.finfo(float).eps * self.xopt.size
            assert_array_less(self.xl - self.xopt - step,
                              tol * absmax_arrays(self.xl, initial=1.0))
            assert_array_less(self.xopt + step - self.xu,
                              tol * absmax_arrays(self.xl, initial=1.0))
            assert_(np.linalg.norm(step) <= 1.1 * delta)
        return step

    def soc_step(self, step, **kwargs):
        """
        Estimate a second-order correction step from ``xopt + step``.

        Parameters
        ----------
        step : numpy.ndarray, shape (n,)
            The current trust-region step.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Second order correction step from `xopt + step`.
        """
        delta = np.linalg.norm(step)
        xsav = self.xopt + step
        aub, bub = self.get_linear_ub(xsav)
        aeq, beq = self.get_linear_eq(xsav)
        ssoc = cpqp(xsav, aub, bub, aeq, beq, self.xl, self.xu, delta, **kwargs)
        ssq = np.inner(ssoc, ssoc)
        if self.debug:
            tol = 10.0 * np.finfo(float).eps * self.xopt.size
            assert_array_less(self.xl - xsav - ssoc,
                              tol * absmax_arrays(self.xl, initial=1.0))
            assert_array_less(xsav + ssoc - self.xu,
                              tol * absmax_arrays(self.xl, initial=1.0))
            assert_(ssq <= 1.1 * delta ** 2.0)
        return ssoc

    def reset_models(self):
        """
        Reset the models.

        The standard models of the objective function, the nonlinear inequality
        constraint function, and the nonlinear equality constraint function are
        set to the ones whose Hessian matrices are least in Frobenius norm.
        """
        self._models.reset_models()

    def build_result(self, penalty=0.0, **kwargs):
        """
        Build the result of the optimization solver.

        Parameters
        ----------
        penalty : float, optional
            Penalty parameter of the merit function used to decide which point
            in the history is the best (only if the history is recorded).

        Returns
        -------
        OptimizeResult
            Result of the optimization solver. See `OptimizeResult` for a
            description of the attributes.

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations has been stored
            (the default is False).
        """
        if kwargs.get('store_history'):
            # Evaluate the constraint violation at each point in the history.
            violmx = np.empty_like(self.fun_hist)
            for i in range(violmx.size):
                x = self.x_hist[i, :] - self.xbase
                cubx = self.cub_hist[i, :] if self.cub_hist.size > 0 else []
                ceqx = self.ceq_hist[i, :] if self.ceq_hist.size > 0 else []
                violmx[i] = self._models.resid(x, cubx, ceqx)

            # The points considers are those for which the constraint violation
            # is at most twice as large as the least one.
            iref = violmx <= 2.0 * np.min(violmx)

            # Select the point at minimize the merit values. If there are
            # multiple minimizers, we select the one with the least constraint
            # violation. If there are again multiple minimizers, we select the
            # with the least objective function value.
            phi = np.full_like(violmx, np.inf)
            if penalty <= 0:
                phi[iref] = self.fun_hist[iref]
            elif np.isinf(penalty):
                phi[iref] = violmx[iref]
            else:
                phi[iref] = self.fun_hist[iref] + penalty * violmx[iref]
            imin = np.argmin(phi)
            iref = iref & (phi <= phi[imin])
            if np.count_nonzero(iref) > 1:
                violmx[np.logical_not(iref)] = np.inf
                imin = np.argmin(violmx)
                iref = iref & (violmx <= violmx[imin])
                if np.count_nonzero(iref) > 1:
                    fun_hist = np.copy(self.fun_hist)
                    fun_hist[np.logical_not(iref)] = np.inf
                    imin = np.argmin(fun_hist)
            xopt = self.x_hist[imin, :] - self.xbase
            fopt = self.fun_hist[imin]
            maxcv = violmx[imin]
        else:
            # The optimal point is the best point in the interpolation set.
            xopt = self.xopt
            fopt = self.fopt
            maxcv = self.maxcv

        # Build the optimization result.
        result = OptimizeResult()
        result.x = self.get_x(self.xbase + xopt)
        result.fun = fopt
        result.jac = np.full_like(result.x, np.nan)
        with suppress(AttributeError):
            free_indices = np.logical_not(self.ifix)
            result.jac[free_indices] = self.model_obj_grad(xopt)
        if self.type != 'U':
            result.maxcv = maxcv
        return result

    def check_models(self, stack_level=2):
        """
        Check the interpolation conditions.

        The method checks whether the evaluations of the quadratic models at the
        interpolation points match their expected values.

        Parameters
        ----------
        stack_level : int, optional
            Stack level of the warning (the default is 2).

        Warns
        -----
        RuntimeWarning
            The evaluations of a quadratic function do not satisfy the
            interpolation conditions up to a certain tolerance.
        """
        self._models.check_models(stack_level)

    def _eval_con(self, con, x):
        """
        Evaluate a constraint function.

        Parameters
        ----------
        con : callable
            Constraint function.

                ``con(x, *args) -> numpy.ndarray, shape (mnl,)``

            where ``x`` is an array with shape (n,) and `args` is a tuple of
            parameters to specify the constraint function.
        x : array_like, shape (n,)
            Point at which the constraint function is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnl,)
            Value of the nonlinear constraint function at `x`.
        """
        if con is not None:
            x_full = self.get_x(x)
            cx = np.atleast_1d(con(x_full, *self._args))
            if cx.dtype.kind in np.typecodes['AllInteger']:
                cx = np.asarray(cx, dtype=float)
            threshold = huge(x_full.dtype)
            cx[np.isnan(cx) | (cx > threshold)] = threshold
            cx[cx < -threshold] = -threshold
            if self.disp and cx.size > 0:
                print(f'{con.__name__}({x_full}) = {cx}.')
        else:
            cx = np.asarray([], dtype=float)
        return cx


class Models:
    """
    Model a nonlinear optimization problem.

    The nonlinear optimization problem is modeled using quadratic functions
    obtained by underdetermined interpolation. The interpolation points may be
    infeasible with respect to the linear and nonlinear constraints, but they
    always satisfy the bound constraints.

    Notes
    -----
    Given the interpolation set, the freedom bequeathed by the interpolation
    conditions is taken up by minimizing the updates of the Hessian matrices of
    the objective and nonlinear constraint functions in Frobenius norm [1]_.

    References
    ----------
    .. [1] M. J. D. Powell. "Least Frobenius norm updating of quadratic models
       that satisfy interpolation conditions." In: Math. Program. 100 (2004),
       pp. 183--215.
    """

    def __init__(self, fun, x0, xl, xu, Aub, bub, Aeq, beq, cub, ceq, options,
                 **kwargs):
        """
        Construct the initial models of an optimization problem.

        Parameters
        ----------
        fun : callable
            Objective function of the nonlinear optimization problem.

                ``fun(x) -> float``

            where ``x`` is an array with shape (n,).
        x0 : numpy.ndarray, shape (n,)
            Initial guess of the nonlinear optimization problem. It is assumed
            that there is no conflict between the bound constraints and `x0`.
            Hence, the components of the initial guess should either equal the
            bound components or allow the projection of the ball centered at
            `x0` of radius ``options.get('rhobeg')`` onto the coordinates to lie
            entirely inside the bounds.
        xl : numpy.ndarray, shape (n,)
            Lower-bound constraints on the decision variables of the nonlinear
            optimization problem ``x >= xl``.
        xu : numpy.ndarray, shape (n,)
            Upper-bound constraints on the decision variables of the nonlinear
            optimization problem ``x <= xu``.
        Aub : numpy.ndarray, shape (mlub, n)
            Jacobian matrix of the linear inequality constraints of the
            nonlinear optimization problem. Each row of `Aub` stores the
            gradient of a linear inequality constraint.
        bub : numpy.ndarray, shape (mlub,)
            Right-hand side vector of the linear inequality constraints of the
            nonlinear optimization problem ``Aub @ x <= bub``.
        Aeq : numpy.ndarray, shape (mleq, n)
            Jacobian matrix of the linear equality constraints of the nonlinear
            optimization problem. Each row of `Aeq` stores the gradient of a
            linear equality constraint.
        beq : numpy.ndarray, shape (mleq,)
            Right-hand side vector of the linear equality constraints of the
            nonlinear optimization problem ``Aeq @ x = beq``.
        cub : callable
            Nonlinear inequality constraint function of the nonlinear
            optimization problem ``cub(x) <= 0``.

                ``cub(x) -> numpy.ndarray, shape (mnlub,)``

            where ``x`` is an array with shape (n,).
        ceq : callable
            Nonlinear equality constraint function of the nonlinear
            optimization problem ``ceq(x) = 0``.

                ``ceq(x) -> numpy.ndarray, shape (mnleq,)``

            where ``x`` is an array with shape (n,).
        options : dict
            Options to forward to the solver. Accepted options are:

                rhobeg : float, optional
                    Initial trust-region radius (the default is 1).
                rhoend : float, optional
                    Final trust-region radius (the default is 1e-6).
                npt : int, optional
                    Number of interpolation points for the objective and
                    constraint models (the default is ``2 * n + 1``).
                maxfev : int, optional
                    Upper bound on the number of objective and constraint
                    function evaluations (the default is ``500 * n``).
                maxiter: int, optional
                    Upper bound on the number of main loop iterations (the
                    default is ``1000 * n``).
                target : float, optional
                    Target value on the objective function (the default is
                    ``-numpy.inf``). If the solver encounters a feasible point
                    at which the objective function evaluations is below the
                    target value, then the computations are stopped.
                ftol_abs : float, optional
                    Absolute tolerance on the objective function.
                ftol_rel : float, optional
                    Relative tolerance on the objective function.
                xtol_abs : float, optional
                    Absolute tolerance on the decision variables.
                xtol_rel : float, optional
                    Relative tolerance on the decision variables.
                disp : bool, optional
                    Whether to print pieces of information on the execution of
                    the solver (the default is False).
                debug : bool, optional
                    Whether to make debugging tests during the execution, which
                    is not recommended in production (the default is False).

        Other Parameters
        ----------------
        store_history : bool, optional
            Whether the history of the different evaluations should be stored
            (the default is False).
        """
        self._xl = xl
        self._xu = xu
        self._Aub = Aub
        self._bub = bub
        self._Aeq = Aeq
        self._beq = beq
        normalize(self.aub, self.bub)
        normalize(self.aeq, self.beq)
        self.shift_constraints(x0)
        n = x0.size
        npt = options.get('npt')
        rhobeg = options.get('rhobeg')
        target = options.get('target')
        cub_x0 = cub(x0, **kwargs)
        mnlub = cub_x0.size
        ceq_x0 = ceq(x0, **kwargs)
        mnleq = ceq_x0.size
        self._xpt = np.zeros((npt, n), dtype=float)
        self._fval = np.empty(npt, dtype=float)
        self._rval = np.empty(npt, dtype=float)
        self._cvalub = np.empty((npt, mnlub), dtype=float)
        self._cvaleq = np.empty((npt, mnleq), dtype=float)
        self._bmat = np.zeros((npt + n, n), dtype=float)
        self._zmat = np.zeros((npt, npt - n - 1), dtype=float)
        self._idz = 0
        self._kopt = 0
        stepa = 0.0
        stepb = 0.0
        bdtol = 10.0 * np.finfo(float).eps * n
        bdtol *= absmax_arrays(xl, xu, initial=1.0)
        for k in range(npt):
            km = k - 1
            kx = km - n

            # Set the displacements from the origin x0 of the calculations of
            # the initial interpolation points in the rows of xpt. It is assumed
            # that there is no conflict between the bounds and x0. Hence, the
            # components of the initial guess should either equal the bound
            # components or allow the projection of the initial trust region
            # onto the components to lie entirely inside the bounds.
            if 1 <= k <= n:
                if abs(self.xu[km]) <= 0.5 * rhobeg:
                    stepa = -rhobeg
                else:
                    stepa = rhobeg
                self.xpt[k, km] = stepa
            elif n < k <= 2 * n:
                stepa = self.xpt[kx + 1, kx]
                if abs(self.xl[kx]) <= 0.5 * rhobeg:
                    stepb = min(2.0 * rhobeg, self.xu[kx])
                elif abs(self.xu[kx]) <= 0.5 * rhobeg:
                    stepb = max(-2.0 * rhobeg, self.xl[kx])
                else:
                    stepb = -rhobeg
                self.xpt[k, kx] = stepb
            elif k > 2 * n:
                shift = kx // n
                ipt = kx - shift * n
                jpt = (ipt + shift) % n
                self.xpt[k, ipt] = self.xpt[ipt + 1, ipt]
                self.xpt[k, jpt] = self.xpt[jpt + 1, jpt]

            # Evaluate the objective and the nonlinear constraint functions at
            # the interpolations points and set the residual of each
            # interpolation point in rval. Stop the computations if a feasible
            # point has reached the target value.
            self.fval[k] = fun(x0 + self.xpt[k, :], **kwargs)
            if k == 0:
                # The constraints functions have already been evaluated at x0
                # to initialize the shapes of cvalub and cvaleq.
                self.cvalub[0, :] = cub_x0
                self.cvaleq[0, :] = ceq_x0
            else:
                self.cvalub[k, :] = cub(x0 + self.xpt[k, :], **kwargs)
                self.cvaleq[k, :] = ceq(x0 + self.xpt[k, :], **kwargs)
            self.rval[k] = self.resid(k)
            if self.fval[k] <= target and self.rval[k] <= bdtol:
                self.kopt = k
                self._target_reached = True
                break

            # Set the initial inverse KKT matrix of interpolation. The matrix
            # bmat holds its last n columns, while zmat stored the rank
            # factorization matrix of its leading not submatrix.
            if k <= 2 * n:
                if 1 <= k <= n and npt <= k + n:
                    self.bmat[0, km] = -1 / stepa
                    self.bmat[k, km] = 1 / stepa
                    self.bmat[npt + km, km] = -0.5 * rhobeg ** 2.0
                elif k > n:
                    self.bmat[0, kx] = -(stepa + stepb) / (stepa * stepb)
                    self.bmat[k, kx] = -0.5 / self.xpt[kx + 1, kx]
                    self.bmat[kx + 1, kx] = -self.bmat[0, kx]
                    self.bmat[kx + 1, kx] -= self.bmat[k, kx]
                    self.zmat[0, kx] = np.sqrt(2.0) / (stepa * stepb)
                    self.zmat[k, kx] = np.sqrt(0.5) / rhobeg ** 2.0
                    self.zmat[kx + 1, kx] = -self.zmat[0, kx]
                    self.zmat[kx + 1, kx] -= self.zmat[k, kx]
            else:
                shift = kx // n
                ipt = kx - shift * n
                jpt = (ipt + shift) % n
                self.zmat[0, kx] = 1.0 / rhobeg ** 2.0
                self.zmat[k, kx] = 1.0 / rhobeg ** 2.0
                self.zmat[ipt + 1, kx] = -1.0 / rhobeg ** 2.0
                self.zmat[jpt + 1, kx] = -1.0 / rhobeg ** 2.0
        else:
            # Set the initial models of the objective and nonlinear constraint
            # functions. The standard models minimize the updates of their
            # Hessian matrices in Frobenius norm when a point of xpt is
            # modified, while the alternative models minimizes their Hessian
            # matrices in Frobenius norm.
            self._target_reached = False
            self._obj = self.new_model(self.fval)
            self._obj_alt = copy.deepcopy(self._obj)
            self._cub = np.empty(mnlub, dtype=Quadratic)
            self._cub_alt = np.empty(mnlub, dtype=Quadratic)
            for i in range(mnlub):
                self._cub[i] = self.new_model(self.cvalub[:, i])
                self._cub_alt[i] = copy.deepcopy(self._cub[i])
            self._ceq = np.empty(mnleq, dtype=Quadratic)
            self._ceq_alt = np.empty(mnleq, dtype=Quadratic)
            for i in range(mnleq):
                self._ceq[i] = self.new_model(self.cvaleq[:, i])
                self._ceq_alt[i] = copy.deepcopy(self._ceq[i])

        # Determine the type of the problem.
        if self.mlub + self.mleq + self.mnlub + self.mnleq == 0:
            if np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
                self._type = 'U'
            else:
                self._type = 'B'
        else:
            if self.mnlub + self.mnleq > 0:
                self._type = 'O'
            else:
                self._type = 'L'

    @property
    def xl(self):
        """
        Lower-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Lower-bound constraints on the decision variables.
        """
        return self._xl

    @property
    def xu(self):
        """
        Upper-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Upper-bound constraints on the decision variables.
        """
        return self._xu

    @property
    def aub(self):
        """
        Jacobian matrix of the normalized linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub, n)
            Jacobian matrix of the normalized linear inequality constraints.
            Each row stores the gradient of a linear inequality constraint.
        """
        return self._Aub

    @property
    def bub(self):
        """
        Right-hand side vector of the normalized linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub,)
            Right-hand side vector of the normalized linear inequality
            constraints.
        """
        return self._bub

    @property
    def mlub(self):
        """
        Number of the linear inequality constraints.

        Returns
        -------
        int
            Number of the linear inequality constraints.
        """
        return self.bub.size

    @property
    def aeq(self):
        """
        Jacobian matrix of the normalized linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq, n)
            Jacobian matrix of the normalized linear equality constraints. Each
            row stores the gradient of a linear equality constraint.
        """
        return self._Aeq

    @property
    def beq(self):
        """
        Right-hand side vector of the normalized linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq,)
            Right-hand side vector of the normalized linear equality
            constraints.
        """
        return self._beq

    @property
    def mleq(self):
        """
        Number of the linear equality constraints.

        Returns
        -------
        int
            Number of the linear equality constraints.
        """
        return self.beq.size

    @property
    def xpt(self):
        """
        Displacements of the interpolation points from the origin.

        Returns
        -------
        numpy.ndarray, shape (npt, n)
            Displacements of the interpolation points from the origin. Each row
            stores the displacements of an interpolation point from the origin
            of the calculations.
        """
        return self._xpt

    @property
    def fval(self):
        """
        Evaluations of the objective function of the nonlinear optimization
        problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            Evaluations of the objective function of the nonlinear optimization
            problem at the interpolation points.
        """
        return self._fval

    @property
    def rval(self):
        """
        Residuals associated with the constraints of the nonlinear optimization
        problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            Residuals associated with the constraints of the nonlinear
            optimization problem at the interpolation points.
        """
        return self._rval

    @property
    def cvalub(self):
        """
        Evaluations of the nonlinear inequality constraint function of the
        nonlinear optimization problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt, mnlub)
            Evaluations of the nonlinear inequality constraint function of the
            nonlinear optimization problem at the interpolation points. Each row
            stores the evaluation of the nonlinear inequality constraint
            functions at an interpolation point.
        """
        return self._cvalub

    @property
    def mnlub(self):
        """
        Number of the nonlinear inequality constraints.

        Returns
        -------
        int
            Number of the nonlinear inequality constraints.
        """
        return self.cvalub.shape[1]

    @property
    def cvaleq(self):
        """
        Evaluations of the nonlinear equality constraint function of the
        nonlinear optimization problem at the interpolation points.

        Returns
        -------
        numpy.ndarray, shape (npt, mnleq)
            Evaluations of the nonlinear equality constraint function of the
            nonlinear optimization problem at the interpolation points. Each row
            stores the evaluation of the nonlinear equality constraint functions
            at an interpolation point.
        """
        return self._cvaleq

    @property
    def mnleq(self):
        """
        Number of the nonlinear equality constraints.

        Returns
        -------
        int
            Number of the nonlinear equality constraints.
        """
        return self.cvaleq.shape[1]

    @property
    def bmat(self):
        """
        Last ``n`` columns of the inverse KKT matrix of interpolation.

        Returns
        -------
        numpy.ndarray, shape (npt + n, n)
            Last ``n`` columns of the inverse KKT matrix of interpolation.
        """
        return self._bmat

    @property
    def zmat(self):
        """
        Rank factorization matrix of the leading ``npt`` submatrix of the
        inverse KKT matrix of interpolation.

        Returns
        -------
        numpy.ndarray, shape (npt, npt - n - 1)
            Rank factorization matrix of the leading ``npt`` submatrix of the
            inverse KKT matrix of interpolation.
        """
        return self._zmat

    @property
    def idz(self):
        """
        Number of nonpositive eigenvalues of the leading ``npt`` submatrix of
        the inverse KKT matrix of interpolation.

        Returns
        -------
        int
            Number of nonpositive eigenvalues of the leading ``npt`` submatrix
            of the inverse KKT matrix of interpolation.

        Notes
        -----
        Although the theoretical number of nonpositive eigenvalues of this
        matrix is always 0, it is designed to tackle numerical difficulties
        caused by ill-conditioned problems.
        """
        return self._idz

    @property
    def kopt(self):
        """
        Index of the interpolation point around which the Taylor expansions of
        the quadratic models are defined.

        Returns
        -------
        int
            Index of the interpolation point around which the Taylor expansion
            of the quadratic models are defined.
        """
        return self._kopt

    @kopt.setter
    def kopt(self, knew):
        """
        Index of the interpolation point around which the Taylor expansions of
        the quadratic models are defined.

        Parameters
        ----------
        knew : int
            New index of the interpolation point around which the Taylor
            expansions of the quadratic models is to be defined.
        """
        if self._kopt != knew:
            # Update the Taylor expansion point of the quadratic models. The
            # models may be undefined when this setter is invoked.
            with suppress(AttributeError):
                step = self.xpt[knew, :] - self.xopt
                self._obj.shift_expansion_point(step, self.xpt)
                self._obj_alt.shift_expansion_point(step, self.xpt)
                for i in range(self.mnlub):
                    self._cub[i].shift_expansion_point(step, self.xpt)
                    self._cub_alt[i].shift_expansion_point(step, self.xpt)
                for i in range(self.mnleq):
                    self._ceq[i].shift_expansion_point(step, self.xpt)
                    self._ceq_alt[i].shift_expansion_point(step, self.xpt)
            self._kopt = knew

    @property
    def xopt(self):
        """
        Interpolation point around which the Taylor expansion of the quadratic
        models are defined.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Interpolation point around which the Taylor expansion of the
            quadratic models are defined.
        """
        return self.xpt[self.kopt, :]

    @property
    def fopt(self):
        """
        Evaluation of the objective function of the nonlinear optimization
        problem at `xopt`.

        Returns
        -------
        float
            Evaluation of the objective function of the nonlinear optimization
            problem at `xopt`.
        """
        return self.fval[self.kopt]

    @property
    def ropt(self):
        """
        Residual associated with the constraints of the nonlinear optimization
        problem at `xopt`.

        Returns
        -------
        float
            Residual associated with the constraints of the nonlinear
            optimization problem at `xopt`.
        """
        return self.rval[self.kopt]

    @property
    def coptub(self):
        """
        Evaluation of the nonlinear inequality constraint function of the
        nonlinear optimization problem at `xopt`.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Evaluation of the nonlinear inequality constraint function of the
            nonlinear optimization problem at `xopt`.
        """
        return self.cvalub[self.kopt, :]

    @property
    def copteq(self):
        """
        Evaluation of the nonlinear equality constraint function of the
        nonlinear optimization problem at `xopt`.

        Returns
        -------
        numpy.ndarray, shape (mnleq,)
            Evaluation of the nonlinear equality constraint function of the
            nonlinear optimization problem at `xopt`.
        """
        return self.cvaleq[self.kopt, :]

    @property
    def type(self):
        """
        Type of the nonlinear optimization problem.

        It follows the CUTEst classification scheme for the constraint types
        (see https://www.cuter.rl.ac.uk/Problems/classification.shtml).

        Returns
        -------
        str
            Type of the nonlinear optimization problem:
                - U : the problem is unconstrained.
                - B : the problem's only constraints are bounds constraints.
                - L : the problem's constraints are linear.
                - O : the problem's constraints general.
        """
        return self._type

    @property
    def target_reached(self):
        """
        Indicate whether the computations have been stopped because the target
        value has been reached.

        Returns
        -------
        bool
            Flag indicating whether the computations have been stopped because
            the target value has been reached.
        """
        return self._target_reached

    def obj(self, x):
        """
        Evaluate the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the objective function of the model at `x`.
        """
        return self.fopt + self._obj(x, self.xpt, self.kopt)

    def obj_grad(self, x):
        """
        Evaluate the gradient of the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the objective function of the model at `x`.
        """
        return self._obj.grad(x, self.xpt, self.kopt)

    def obj_hess(self):
        """
        Evaluate the Hessian matrix of the objective function of the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the objective function of the model.
        """
        return self._obj.hess(self.xpt)

    def obj_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the objective function of
        the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the objective function
            of the model with the vector `x`.
        """
        return self._obj.hessp(x, self.xpt)

    def obj_curv(self, x):
        """
        Evaluate the curvature of the objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the objective function of the model at `x`.
        """
        return self._obj.curv(x, self.xpt)

    def obj_alt(self, x):
        """
        Evaluate the alternative objective function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.

        Returns
        -------
        float
            Value of the alternative objective function of the model at `x`.
        """
        return self.fopt + self._obj_alt(x, self.xpt, self.kopt)

    def obj_alt_grad(self, x):
        """
        Evaluate the gradient of the alternative objective function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the alternative objective function of the model at `x`.
        """
        return self._obj_alt.grad(x, self.xpt, self.kopt)

    def obj_alt_hess(self):
        """
        Evaluate the Hessian matrix of the alternative objective function of the
        model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the alternative objective function of the model.
        """
        return self._obj_alt.hess(self.xpt)

    def obj_alt_hessp(self, x):
        """
        Evaluate the product of the Hessian matrix of the alternative objective
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the alternative
            objective function of the model with the vector `x`.
        """
        return self._obj_alt.hessp(x, self.xpt)

    def obj_alt_curv(self, x):
        """
        Evaluate the curvature of the alternative objective function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.

        Returns
        -------
        float
            Curvature of the alternative objective function of the model at `x`.
        """
        return self._obj_alt.curv(x, self.xpt)

    def cub(self, x, i):
        """
        Evaluate an inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th inequality constraint function of the model at
            `x`.
        """
        return self.coptub[i] + self._cub[i](x, self.xpt, self.kopt)

    def cub_grad(self, x, i):
        """
        Evaluate the gradient of an inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th inequality constraint function of the model
            at `x`.
        """
        return self._cub[i].grad(x, self.xpt, self.kopt)

    def cub_hess(self, i):
        """
        Evaluate the Hessian matrix of an inequality constraint function of the
        model.

        Parameters
        ----------
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th inequality constraint function of the
            model.
        """
        return self._cub[i].hess(self.xpt)

    def cub_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an inequality constraint
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th inequality
            constraint function of the model with the vector `x`.
        """
        return self._cub[i].hessp(x, self.xpt)

    def cub_curv(self, x, i):
        """
        Evaluate the curvature of an inequality constraint function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th inequality constraint function of the model
            at `x`.
        """
        return self._cub[i].curv(x, self.xpt)

    def cub_alt(self, x, i):
        """
        Evaluate an alternative inequality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th alternative inequality constraint function of
            the model at `x`.
        """
        return self.coptub[i] + self._cub_alt[i](x, self.xpt, self.kopt)

    def cub_alt_grad(self, x, i):
        """
        Evaluate the gradient of an alternative inequality constraint function
        of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th alternative inequality constraint function of
            the model at `x`.
        """
        return self._cub_alt[i].grad(x, self.xpt, self.kopt)

    def cub_alt_hess(self, i):
        """
        Evaluate the Hessian matrix of an alternative inequality constraint
        function of the model.

        Parameters
        ----------
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th alternative inequality constraint
            function of the model.
        """
        return self._cub_alt[i].hess(self.xpt)

    def cub_alt_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an alternative inequality
        constraint function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th alternative
            inequality constraint function of the model with the vector `x`.
        """
        return self._cub_alt[i].hessp(x, self.xpt)

    def cub_alt_curv(self, x, i):
        """
        Evaluate the curvature of an alternative inequality constraint function
        of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the inequality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th alternative inequality constraint function
            of the model at `x`.
        """
        return self._cub_alt[i].curv(x, self.xpt)

    def ceq(self, x, i):
        """
        Evaluate an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self.copteq[i] + self._ceq[i](x, self.xpt, self.kopt)

    def ceq_grad(self, x, i):
        """
        Evaluate the gradient of an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self._ceq[i].grad(x, self.xpt, self.kopt)

    def ceq_hess(self, i):
        """
        Evaluate the Hessian matrix of an equality constraint function of the
        model.

        Parameters
        ----------
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th equality constraint function of the
            model.
        """
        return self._ceq[i].hess(self.xpt)

    def ceq_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an equality constraint
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th equality
            constraint function of the model with the vector `x`.
        """
        return self._ceq[i].hessp(x, self.xpt)

    def ceq_curv(self, x, i):
        """
        Evaluate the curvature of an equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th equality constraint function of the model at
            `x`.
        """
        return self._ceq[i].curv(x, self.xpt)

    def ceq_alt(self, x, i):
        """
        Evaluate an alternative equality constraint function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Value of the `i`-th alternative equality constraint function of the
            model at `x`.
        """
        return self.copteq[i] + self._ceq_alt[i](x, self.xpt, self.kopt)

    def ceq_alt_grad(self, x, i):
        """
        Evaluate the gradient of an alternative equality constraint function of
        the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the `i`-th alternative equality constraint function of
            the model at `x`.
        """
        return self._ceq_alt[i].grad(x, self.xpt, self.kopt)

    def ceq_alt_hess(self, i):
        """
        Evaluate the Hessian matrix of an alternative equality constraint
        function of the model.

        Parameters
        ----------
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the `i`-th alternative equality constraint
            function of the model.
        """
        return self._ceq_alt[i].hess(self.xpt)

    def ceq_alt_hessp(self, x, i):
        """
        Evaluate the product of the Hessian matrix of an alternative equality
        constraint function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the `i`-th alternative
            equality constraint function of the model with the vector `x`.
        """
        return self._ceq_alt[i].hessp(x, self.xpt)

    def ceq_alt_curv(self, x, i):
        """
        Evaluate the curvature of an alternative equality constraint function of
        the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        i : int
            Index of the equality constraint to be considered.

        Returns
        -------
        float
            Curvature of the `i`-th alternative equality constraint function of
            the model at `x`.
        """
        return self._ceq_alt[i].curv(x, self.xpt)

    def lag(self, x, lmlub, lmleq, lmnlub, lmnleq):
        """
        Evaluate the Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        lmlub : numpy.ndarray, shape (mlub,)
            Lagrange multipliers associated with the linear inequality
            constraints.
        lmleq : numpy.ndarray, shape (mleq,)
            Lagrange multipliers associated with the linear equality
            constraints.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        float
            Value of the Lagrangian function of the model at `x`.
        """
        lx = self.obj(x)
        lx += np.inner(lmlub, np.dot(self.aub, x) - self.bub)
        lx += np.inner(lmleq, np.dot(self.aeq, x) - self.beq)
        for i in range(self.mnlub):
            lx += lmnlub[i] * self.cub(x, i)
        for i in range(self.mnleq):
            lx += lmnleq[i] * self.ceq(x, i)
        return lx

    def lag_grad(self, x, lmlub, lmleq, lmnlub, lmnleq):
        """
        Evaluate the gradient of Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        lmlub : numpy.ndarray, shape (mlub,)
            Lagrange multipliers associated with the linear inequality
            constraints.
        lmleq : numpy.ndarray, shape (mleq,)
            Lagrange multipliers associated with the linear equality
            constraints.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the Lagrangian function of the model at `x`.
        """
        gx = self.obj_grad(x)
        gx += np.dot(self.aub.T, lmlub)
        gx += np.dot(self.aeq.T, lmleq)
        for i in range(self.mnlub):
            gx += lmnlub[i] * self.cub_grad(x, i)
        for i in range(self.mnleq):
            gx += lmnleq[i] * self.ceq_grad(x, i)
        return gx

    def lag_hess(self, lmnlub, lmnleq):
        """
        Evaluate the Hessian matrix of the Lagrangian function of the model.

        Parameters
        ----------
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the Lagrangian function of the model.
        """
        hx = self.obj_hess()
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_hess(i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_hess(i)
        return hx

    def lag_hessp(self, x, lmnlub, lmnleq):
        """
        Evaluate the product of the Hessian matrix of the Lagrangian function of
        the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the Lagrangian
            function of the model with the vector `x`.
        """
        hx = self.obj_hessp(x)
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_hessp(x, i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_hessp(x, i)
        return hx

    def lag_curv(self, x, lmnlub, lmnleq):
        """
        Evaluate the curvature of the Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        float
            Curvature of the Lagrangian function of the model at `x`.
        """
        cx = self.obj_curv(x)
        for i in range(self.mnlub):
            cx += lmnlub[i] * self.cub_curv(x, i)
        for i in range(self.mnleq):
            cx += lmnleq[i] * self.ceq_curv(x, i)
        return cx

    def lag_alt(self, x, lmlub, lmleq, lmnlub, lmnleq):
        """
        Evaluate the alternative Lagrangian function of the model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        lmlub : numpy.ndarray, shape (mlub,)
            Lagrange multipliers associated with the linear inequality
            constraints.
        lmleq : numpy.ndarray, shape (mleq,)
            Lagrange multipliers associated with the linear equality
            constraints.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        float
            Value of the alternative Lagrangian function of the model at `x`.
        """
        lx = self.obj_alt(x)
        lx += np.inner(lmlub, np.dot(self.aub, x) - self.bub)
        lx += np.inner(lmleq, np.dot(self.aeq, x) - self.beq)
        for i in range(self.mnlub):
            lx += lmnlub[i] * self.cub_alt(x, i)
        for i in range(self.mnleq):
            lx += lmnleq[i] * self.ceq_alt(x, i)
        return lx

    def lag_alt_grad(self, x, lmlub, lmleq, lmnlub, lmnleq):
        """
        Evaluate the gradient of the alternative Lagrangian function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        lmlub : numpy.ndarray, shape (mlub,)
            Lagrange multipliers associated with the linear inequality
            constraints.
        lmleq : numpy.ndarray, shape (mleq,)
            Lagrange multipliers associated with the linear equality
            constraints.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Gradient of the alternative Lagrangian function of the model at `x`.
        """
        gx = self.obj_alt_grad(x)
        gx += np.dot(self.aub.T, lmlub)
        gx += np.dot(self.aeq.T, lmleq)
        for i in range(self.mnlub):
            gx += lmnlub[i] * self.cub_alt_grad(x, i)
        for i in range(self.mnleq):
            gx += lmnleq[i] * self.ceq_alt_grad(x, i)
        return gx

    def lag_alt_hess(self, lmnlub, lmnleq):
        """
        Evaluate the Hessian matrix of the alternative Lagrangian function of
        the model.

        Parameters
        ----------
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the alternative Lagrangian function of the model.
        """
        hx = self.obj_alt_hess()
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_alt_hess(i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_alt_hess(i)
        return hx

    def lag_alt_hessp(self, x, lmnlub, lmnleq):
        """
        Evaluate the product of the Hessian matrix of the alternative Lagrangian
        function of the model with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the alternative
            Lagrangian function of the model with the vector `x`.
        """
        hx = self.obj_alt_hessp(x)
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_alt_hessp(x, i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_alt_hessp(x, i)
        return hx

    def lag_alt_curv(self, x, lmnlub, lmnleq):
        """
        Evaluate the curvature of the alternative Lagrangian function of the
        model.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        lmnlub : numpy.ndarray, shape (mnlub,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear inequality constraints.
        lmnleq : numpy.ndarray, shape (mnleq,)
            Lagrange multipliers associated with the quadratic models of the
            nonlinear equality constraints.

        Returns
        -------
        float
            Curvature of the alternative Lagrangian function of the model at
            `x`.
        """
        cx = self.obj_alt_curv(x)
        for i in range(self.mnlub):
            cx += lmnlub[i] * self.cub_alt_curv(x, i)
        for i in range(self.mnleq):
            cx += lmnleq[i] * self.ceq_alt_curv(x, i)
        return cx

    def shift_constraints(self, x):
        """
        Shift the bound and linear constraints.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Coordinates of the shift to be performed.
        """
        self._xl -= x
        self._xu -= x
        self._bub -= np.dot(self.aub, x)
        self._beq -= np.dot(self.aeq, x)

    def shift_origin(self):
        """
        Update the models when the origin of the calculations is modified.

        Notes
        -----
        Given ``xbase`` the previous origin of the calculations, it is assumed
        that the origin is shifted by `xopt`.
        """
        xopt = np.copy(self.xopt)
        npt, n = self.xpt.shape
        xoptsq = np.inner(xopt, xopt)

        # Make the changes to bmat that do not depend on zmat.
        qoptsq = 0.25 * xoptsq
        updt = np.dot(self.xpt, xopt) - 0.5 * xoptsq
        hxpt = self.xpt - 0.5 * xopt[np.newaxis, :]
        for k in range(npt):
            step = updt[k] * hxpt[k, :] + qoptsq * xopt
            temp = np.outer(self.bmat[k, :], step)
            self.bmat[npt:, :] += temp + temp.T

        # Revise bmat to incorporate the changes that depend on zmat.
        temp = qoptsq * np.outer(xopt, np.sum(self.zmat, axis=0))
        temp += np.matmul(hxpt.T, self.zmat * updt[:, np.newaxis])
        for k in range(self.idz):
            self.bmat[:npt, :] -= np.outer(self.zmat[:, k], temp[:, k])
            self.bmat[npt:, :] -= np.outer(temp[:, k], temp[:, k])
        for k in range(self.idz, npt - n - 1):
            self.bmat[:npt, :] += np.outer(self.zmat[:, k], temp[:, k])
            self.bmat[npt:, :] += np.outer(temp[:, k], temp[:, k])

        # Complete the shift by updating the quadratic models, the bound
        # constraints, the right-hand side of the linear inequality and equality
        # constraints, and the interpolation points.
        self._obj.shift_interpolation_points(self.xpt, self.kopt)
        self._obj_alt.shift_interpolation_points(self.xpt, self.kopt)
        for i in range(self.mnlub):
            self._cub[i].shift_interpolation_points(self.xpt, self.kopt)
            self._cub_alt[i].shift_interpolation_points(self.xpt, self.kopt)
        for i in range(self.mnleq):
            self._ceq[i].shift_interpolation_points(self.xpt, self.kopt)
            self._ceq_alt[i].shift_interpolation_points(self.xpt, self.kopt)
        self.shift_constraints(xopt)
        self._xpt -= xopt[np.newaxis, :]

    def update(self, step, fx, cubx, ceqx, knew=None):
        """
        Update the models of the nonlinear optimization problem when a point of
        the interpolation set is modified.

        Parameters
        ----------
        step : numpy.ndarray, shape (n,)
            Displacement from `xopt` of the point to replace an interpolation
            point.
        fx : float
            Value of the objective function at the trial point.
        cubx : numpy.ndarray, shape (mnlub,)
            Value of the nonlinear inequality constraint function at the trial
            point.
        ceqx : numpy.ndarray, shape (mnleq,)
            Value of the nonlinear equality constraint function at the trial
            point.
        knew : int, optional
            Index of the interpolation point to be removed. It is automatically
            chosen if it is not provided.

        Returns
        -------
        int
            Index of the interpolation point that has been replaced.

        Raises
        ------
        ZeroDivisionError
            The denominator of the updating formula is zero.

        Notes
        -----
        When the index `knew` of the interpolation point to be removed is not
        provided, it is chosen by the method to maximize the product absolute
        value of the denominator in Equation (2.12) of [1]_ with the quartic
        power of the distance between the point and `xopt`.

        References
        ----------
        .. [1] M. J. D. Powell. "On updating the inverse of a KKT matrix." In:
           Numerical Linear Algebra and Optimization. Ed. by Y. Yuan. Beijing,
           CN: Science Press, 2004, pp. 56--78.
        """
        npt, n = self.xpt.shape
        tiny = np.finfo(float).tiny

        # Evaluate the Lagrange polynomials related to the interpolation points
        # and the real parameter beta given in Equation (2.13) of Powell (2004).
        beta, vlag = self._beta(step)

        # Select the index of the interpolation point to be deleted.
        if knew is None:
            knew = self._get_point_to_remove(beta, vlag)

        # Put zeros in the knew-th row of zmat by applying a sequence of Givens
        # rotations. The remaining updates are performed below.
        drotg, = get_blas_funcs(('rotg',), (self.zmat,))
        drot, = get_blas_funcs(('rot',), (self.zmat,))
        jdz = 0
        for j in range(1, npt - n - 1):
            if j == self.idz:
                jdz = self.idz
            elif abs(self.zmat[knew, j]) > 0.0:
                cval = self.zmat[knew, jdz]
                sval = self.zmat[knew, j]
                cosv, sinv = drotg(cval, sval)
                self.zmat[:, jdz], self.zmat[:, j] = \
                    drot(self.zmat[:, jdz], self.zmat[:, j], cosv, sinv)
                self.zmat[knew, j] = 0.0

        # Evaluate the denominator in Equation (2.12) of Powell (2004).
        scala = self.zmat[knew, 0] if self.idz == 0 else -self.zmat[knew, 0]
        scalb = 0.0 if jdz == 0 else self.zmat[knew, jdz]
        omega = scala * self.zmat[:, 0] + scalb * self.zmat[:, jdz]
        alpha = omega[knew]
        tau = vlag[knew]
        sigma = alpha * beta + tau ** 2.0
        vlag[knew] -= 1.0
        bmax = np.max(np.abs(self.bmat), initial=1.0)
        zmax = np.max(np.abs(self.zmat), initial=1.0)
        if abs(sigma) < tiny * max(bmax, zmax):
            # The denominator of the updating formula is too small to safely
            # divide the coefficients of the KKT matrix of interpolation.
            # Theoretically, the value of abs(sigma) is always positive, and
            # becomes small only for ill-conditioned problems.
            raise ZeroDivisionError

        # Complete the update of the matrix zmat. The boolean variable reduce
        # indicates whether the number of nonpositive eigenvalues of the leading
        # npt submatrix of the inverse KKT matrix of interpolation in self.idz
        # must be decreased by one.
        reduce = False
        hval = np.sqrt(abs(sigma))
        if jdz == 0:
            scala = tau / hval
            scalb = self.zmat[knew, 0] / hval
            self.zmat[:, 0] = scala * self.zmat[:, 0] - scalb * vlag[:npt]
            if sigma < 0.0:
                if self.idz == 0:
                    self._idz = 1
                else:
                    reduce = True
        else:
            kdz = jdz if beta >= 0.0 else 0
            jdz -= kdz
            tempa = self.zmat[knew, jdz] * beta / sigma
            tempb = self.zmat[knew, jdz] * tau / sigma
            temp = self.zmat[knew, kdz]
            scala = 1. / np.sqrt(abs(beta) * temp ** 2.0 + tau ** 2.0)
            scalb = scala * hval
            self.zmat[:, kdz] = tau * self.zmat[:, kdz] - temp * vlag[:npt]
            self.zmat[:, kdz] *= scala
            self.zmat[:, jdz] -= tempa * omega + tempb * vlag[:npt]
            self.zmat[:, jdz] *= scalb
            if sigma <= 0.0:
                if beta < 0.0:
                    self._idz += 1
                else:
                    reduce = True
        if reduce:
            self._idz -= 1
            self.zmat[:, [0, self.idz]] = self.zmat[:, [self.idz, 0]]

        # Update accordingly bmat. The copy below is crucial, as the slicing
        # would otherwise return a view of the knew-th row of bmat only.
        bsav = np.copy(self.bmat[knew, :])
        for j in range(n):
            cosv = (alpha * vlag[npt + j] - tau * bsav[j]) / sigma
            sinv = (tau * vlag[npt + j] + beta * bsav[j]) / sigma
            self.bmat[:npt, j] += cosv * vlag[:npt] - sinv * omega
            self.bmat[npt:npt + j + 1, j] += cosv * vlag[npt:npt + j + 1]
            self.bmat[npt:npt + j + 1, j] -= sinv * bsav[:j + 1]
            self.bmat[npt + j, :j + 1] = self.bmat[npt:npt + j + 1, j]

        # Update finally the evaluations of the objective function, the
        # nonlinear inequality constraint function, and the nonlinear equality
        # constraint function, the interpolation points, the residuals of the
        # interpolation points, and the models of the problem.
        xnew = self.xopt + step
        xold = np.copy(self.xpt[knew, :])
        dfx = fx - self.obj(xnew)
        self.fval[knew] = fx
        dcubx = np.empty(self.mnlub, dtype=float)
        for i in range(self.mnlub):
            dcubx[i] = cubx[i] - self.cub(xnew, i)
        self.cvalub[knew, :] = cubx
        dceqx = np.empty(self.mnleq, dtype=float)
        for i in range(self.mnleq):
            dceqx[i] = ceqx[i] - self.ceq(xnew, i)
        self.cvaleq[knew, :] = ceqx
        self.xpt[knew, :] = xnew
        self.rval[knew] = self.resid(knew)
        self._obj.update(self.xpt, self.kopt, xold, self.bmat, self.zmat,
                         self.idz, knew, dfx)
        self._obj_alt = self.new_model(self.fval)
        for i in range(self.mnlub):
            self._cub[i].update(self.xpt, self.kopt, xold, self.bmat, self.zmat,
                                self.idz, knew, dcubx[i])
            self._cub_alt[i] = self.new_model(self.cvalub[:, i])
        for i in range(self.mnleq):
            self._ceq[i].update(self.xpt, self.kopt, xold, self.bmat, self.zmat,
                                self.idz, knew, dceqx[i])
            self._ceq_alt[i] = self.new_model(self.cvaleq[:, i])
        return knew

    def new_model(self, val):
        """
        Generate a model obtained by underdetermined interpolation.

        The freedom bequeathed by the interpolation conditions defined by `val`
        is taken up by minimizing the Hessian matrix of the quadratic function
        in Frobenius norm.

        Parameters
        ----------
        val : {int, numpy.ndarray, shape (npt,)}
            Evaluations associated with the interpolation points. An integer
            value represents the ``npt``-dimensional vector whose components are
            all zero, except the `val`-th one whose value is one. Hence,
            passing an integer value construct the `val`-th Lagrange polynomial
            associated with the interpolation points.

        Returns
        -------
        Quadratic
            The quadratic model that satisfy the interpolation conditions
            defined by `val`, whose Hessian matrix is least in Frobenius norm.
        """
        model = Quadratic(self.bmat, self.zmat, self.idz, val)
        model.shift_expansion_point(self.xopt, self.xpt)
        return model

    def reset_models(self):
        """
        Reset the models.

        The standard models of the objective function, the nonlinear inequality
        constraint function, and the nonlinear equality constraint function are
        set to the ones whose Hessian matrices are least in Frobenius norm.
        """
        self._obj = copy.deepcopy(self._obj_alt)
        self._cub = copy.deepcopy(self._cub_alt)
        self._ceq = copy.deepcopy(self._ceq_alt)

    def improve_geometry(self, klag, delta, aub, bub, aeq, beq, **kwargs):
        """
        Estimate a step from `xopt` that aims at improving the geometry of the
        interpolation set.

        Two alternative steps are computed.

            1. The first alternative step is selected on the lines that join
               `xopt` to the other interpolation points that maximize a lower
               bound on the denominator of the updating formula.
            2. The second alternative is a constrained Cauchy step.
            3. The third alternative uses a truncated conjugate gradient method
               to estimate the maximum of a lower bound on the denominator of
               the updating formula.

        Among the alternative steps, the method selects the one that leads to
        the greatest denominator of the updating formula.

        Parameters
        ----------
        klag : int
            Index of the interpolation point that is to be replaced.
        delta : float
            Upper bound on the length of the step.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Step from `xopt` that aims at improving the geometry of the
            interpolation set.

        Other Parameters
        ----------------
        debug : bool, optional
            Whether to make debugging tests during the execution, which is
            not recommended in production (the default is False).
        improve_tcg : bool, optional
            Whether to improve the truncated conjugate gradient step round the
            trust-region boundary (the default is True).
        """
        # Define the tolerances to compare floating-point numbers with zero.
        npt = self.xpt.shape[0]
        eps = np.finfo(float).eps
        tol = 10.0 * eps * npt

        # Determine the klag-th Lagrange polynomial. It is the quadratic
        # function whose value is zero at each interpolation point, except at
        # the klag-th one, whose value is one. The freedom bequeathed by these
        # interpolation conditions is taken up by minimizing the Hessian matrix
        # of the quadratic function is Frobenius norm.
        lag = self.new_model(klag)
        glag = lag.grad(self.xopt, self.xpt, self.kopt)
        omega = implicit_hessian(self.zmat, self.idz, klag)
        alpha = omega[klag]

        # Determine a point on a line between xopt and another interpolation
        # points, chosen to maximize the absolute value of the klag-th
        # Lagrange polynomial, which is a lower bound on the denominator of
        # the updating formula.
        step = bvlag(self.xpt, self.kopt, klag, glag, self.xl, self.xu,
                     delta, alpha, **kwargs)
        beta, vlag = self._beta(step)
        sigma = vlag[klag] ** 2.0 + alpha * beta

        # Evaluate the constrained Cauchy step from the optimal point of the
        # absolute value of the klag-th Lagrange polynomial.
        salt, cauchy = bvcs(self.xpt, self.kopt, glag, lag.curv, self.xl,
                            self.xu, delta, self.xpt, **kwargs)
        if sigma < cauchy and cauchy > tol * max(1.0, abs(sigma)):
            step = salt
            # sigma = cauchy

        # Estimate the maximum of the absolute value of the klag-th Lagrange
        # polynomial using a truncated conjugate gradient method.

        return step

    def resid(self, x, cubx=None, ceqx=None):
        """
        Evaluate the residual associated with the constraints of the nonlinear
        optimization problem.

        Parameters
        ----------
        x : {int, numpy.ndarray, shape (n,)}
            Point at which the residual is to be evaluated. An integer value
            represents the `x`-th interpolation point.
        cubx : numpy.ndarray, shape (mnlub,), optional
            Value of the nonlinear inequality constraint function at `x`. It
            is required only if `x` is not an integer, and is not considered if
            `x` represents an interpolation point.
        ceqx : numpy.ndarray, shape (mnleq,), optional
            Value of the nonlinear equality constraint function at `x`. It is
            required only if `x` is not an integer, and is not considered if `x`
            represents an interpolation point.

        Returns
        -------
        float
            Residual associated with the constraints of the nonlinear
            optimization problem at `x`.
        """
        if isinstance(x, (int, np.integer)):
            cubx = self.cvalub[x, :]
            ceqx = self.cvaleq[x, :]
            x = self.xpt[x, :]
        cub = np.r_[np.dot(self.aub, x) - self.bub, cubx]
        ceq = np.r_[np.dot(self.aeq, x) - self.beq, ceqx]
        cbd = np.r_[x - self.xu, self.xl - x]
        return np.max(np.r_[cub, np.abs(ceq), cbd], initial=0.0)

    def check_models(self, stack_level=2):
        """
        Check the interpolation conditions.

        The method checks whether the evaluations of the quadratic models at the
        interpolation points match their expected values.

        Parameters
        ----------
        stack_level : int, optional
            Stack level of the warning (the default is 2).

        Warns
        -----
        RuntimeWarning
            The evaluations of a quadratic function do not satisfy the
            interpolation conditions up to a certain tolerance.
        """
        stack_level += 1
        self._obj.check_model(self.xpt, self.fval, self.kopt, stack_level)
        for i in range(self.mnlub):
            self._cub[i].check_model(
                self.xpt, self.cvalub[:, i], self.kopt, stack_level)
        for i in range(self.mnleq):
            self._ceq[i].check_model(
                self.xpt, self.cvaleq[:, i], self.kopt, stack_level)

    def _get_point_to_remove(self, beta, vlag):
        """
        Select a point to remove from the interpolation set.

        Parameters
        ----------
        beta : float
            Parameter beta involved in the denominator of the updating formula.
        vlag : numpy.ndarray, shape (2 * npt,)
            Vector whose first ``npt`` components are evaluations of the
            Lagrange polynomials associated with the interpolation points.

        Returns
        -------
        int
            Index of the point to remove from the interpolation.

        Notes
        -----
        The point to remove is chosen to maximize the product absolute value of
        the denominator in Equation (2.12) of [1]_ with the quartic power of the
        distance between the point and `xopt`.

        References
        ----------
        .. [1] M. J. D. Powell. "On updating the inverse of a KKT matrix." In:
           Numerical Linear Algebra and Optimization. Ed. by Y. Yuan. Beijing,
           CN: Science Press, 2004, pp. 56--78.
        """
        npt = self.xpt.shape[0]
        zsq = self.zmat ** 2.0
        zsq = np.c_[-zsq[:, :self.idz], zsq[:, self.idz:]]
        alpha = np.sum(zsq, axis=1)
        sigma = vlag[:npt] ** 2.0 + beta * alpha
        dsq = np.sum((self.xpt - self.xopt[np.newaxis, :]) ** 2.0, axis=1)
        return np.argmax(np.abs(sigma) * np.square(dsq))

    def _beta(self, step):
        """
        Evaluate the parameter beta involved in the denominator of the updating
        formula for the trial point.

        Parameters
        ----------
        step : numpy.ndarray, shape (n,)
            Displacement from `xopt` of the trial step included in the parameter
            beta involved in the denominator of the updating formula.

        Returns
        -------
        beta : float
            Parameter beta involved in the denominator of the updating formula.
        vlag : numpy.ndarray, shape (2 * npt,)
            Vector whose first ``npt`` components are the evaluations of the
            Lagrange polynomials associated with the interpolation points at the
            trial point. The remaining components of `vlag` are not meaningful,
            but are involved in several updating formulae.
        """
        npt, n = self.xpt.shape
        vlag = np.empty(npt + n, dtype=float)
        stepsq = np.inner(step, step)
        xoptsq = np.inner(self.xopt, self.xopt)
        stx = np.inner(step, self.xopt)
        xstep = np.dot(self.xpt, step)
        xxopt = np.dot(self.xpt, self.xopt)
        check = xstep * (0.5 * xstep + xxopt)
        zalt = np.c_[-self.zmat[:, :self.idz], self.zmat[:, self.idz:]]
        temp = np.dot(zalt.T, check)
        beta = np.inner(temp[:self.idz], temp[:self.idz])
        beta -= np.inner(temp[self.idz:], temp[self.idz:])
        vlag[:npt] = np.dot(self.bmat[:npt, :], step)
        vlag[:npt] += np.dot(self.zmat, temp)
        vlag[self.kopt] += 1.0
        vlag[npt:] = np.dot(self.bmat[:npt, :].T, check)
        bsp = np.inner(vlag[npt:], step)
        vlag[npt:] += np.dot(self.bmat[npt:, :], step)
        bsp += np.inner(vlag[npt:], step)
        beta += stx ** 2.0 + stepsq * (xoptsq + 2.0 * stx + 0.5 * stepsq) - bsp
        return beta, vlag


class Quadratic:
    """
    Representation of a quadratic multivariate function.

    Notes
    -----
    To improve the computational efficiency of the updates of the quadratic
    functions, the Hessian matrices of a quadratic functions are stored as
    explicit and implicit parts, which define the model relatively to the
    coordinates of the interpolation points [1]_. Initially, the explicit part
    of an Hessian matrix is zero and so, is not explicitly stored.

    References
    ----------
    .. [1] M. J. D. Powell. "The NEWUOA software for unconstrained optimization
       without derivatives." In: Large-Scale Nonlinear Optimization. Ed. by G.
       Di Pillo and M. Roma. New York, NY, US: Springer, 2006, pp. 255--297.
    """

    def __init__(self, bmat, zmat, idz, fval):
        """
        Construct a quadratic function by underdetermined interpolation.

        Parameters
        ----------
        bmat : numpy.ndarray, shape (npt + n, n)
            Last ``n`` columns of the inverse KKT matrix of interpolation.
        zmat : numpy.ndarray, shape (npt, npt - n - 1)
            Rank factorization matrix of the leading ``npt`` submatrix of the
            inverse KKT matrix of interpolation.
        idz : int
            Number of nonpositive eigenvalues of the leading ``npt`` submatrix
            of the inverse KKT matrix of interpolation. Although its theoretical
            value is always 0, it is designed to tackle numerical difficulties
            caused by ill-conditioned problems.
        fval : {int, numpy.ndarray, shape (npt,)}
            Evaluations associated with the interpolation points. An integer
            value represents the ``npt``-dimensional vector whose components are
            all zero, except the `fval`-th one whose value is one. Hence,
            passing an integer value construct the `fval`-th Lagrange polynomial
            associated with the interpolation points.
        """
        npt = zmat.shape[0]
        if isinstance(fval, (int, np.integer)):
            # The gradient of the fval-th Lagrange quadratic model is the
            # product of the first npt columns of the transpose of bmat with the
            # npt-dimensional vector whose components are zero, except the
            # fval-th one whose value is one.
            self._gq = np.copy(bmat[fval, :])
        else:
            self._gq = np.dot(bmat[:npt, :].T, fval)
        self._pq = implicit_hessian(zmat, idz, fval)

        # Initially, the explicit part of the Hessian matrix of the model is the
        # zero matrix. To improve the computational efficiency of the code, it
        # is not explicitly initialized, and is stored only when updating the
        # model, if it might become a nonzero matrix.
        self._hq = None

    def __call__(self, x, xpt, kopt):
        """
        Evaluate the quadratic function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. The constant term of the quadratic function is not
            maintained, and zero is returned at ``xpt[kopt, :]``.

        Returns
        -------
        float
            Value of the quadratic function at `x`.
        """
        x = x - xpt[kopt, :]
        qx = np.inner(self.gq, x)
        qx += 0.5 * np.inner(self.pq, np.dot(xpt, x) ** 2.0)
        if self._hq is not None:
            # If the explicit part of the Hessian matrix is not defined, it is
            # understood as the zero matrix. Therefore, if self.hq is None, the
            # second-order term is entirely defined by the implicit part of the
            # Hessian matrix of the quadratic function.
            qx += 0.5 * np.inner(x, np.dot(self.hq, x))
        return qx

    @property
    def gq(self):
        """
        Stored gradient of the model.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Stored gradient of the model.
        """
        return self._gq

    @property
    def pq(self):
        """
        Stored implicit part of the Hessian matrix of the model.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            Stored implicit part of the Hessian matrix of the model.
        """
        return self._pq

    @property
    def hq(self):
        """
        Stored explicit part of the Hessian matrix of the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Stored explicit part of the Hessian matrix of the model.
        """
        if self._hq is None:
            return np.zeros((self.gq.size, self.gq.size), dtype=float)
        return self._hq

    def grad(self, x, xpt, kopt):
        """
        Evaluate the gradient of the quadratic function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the gradient of the quadratic function is to be
            evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. The constant term of the quadratic function is not
            maintained, and zero is returned at ``xpt[kopt, :]``.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the gradient of the quadratic function at `x`.
        """
        return self.gq + self.hessp(x - xpt[kopt, :], xpt)

    def hess(self, xpt):
        """
        Evaluate the Hessian matrix of the quadratic function.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the quadratic function.

        Notes
        -----
        The Hessian matrix of the model is not explicitly stored and its
        computation requires a matrix multiplication. If only products of the
        Hessian matrix of the model with any vector are required, consider using
        instead `hessp`.
        """
        return self.hq + np.matmul(xpt.T, self.pq[:, np.newaxis] * xpt)

    def hessp(self, x, xpt):
        """
        Evaluate the product of the Hessian matrix of the quadratic function
        with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Vector to be left-multiplied by the Hessian matrix of the quadratic
            function.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Value of the product of the Hessian matrix of the quadratic function
            with the vector `x`.
        """
        hx = np.dot(xpt.T, self.pq * np.dot(xpt, x))
        if self._hq is not None:
            # If the explicit part of the Hessian matrix is not defined, it is
            # understood as the zero matrix. Therefore, if self.hq is None, the
            # Hessian matrix is entirely defined by its implicit part.
            hx += np.dot(self.hq, x)
        return hx

    def curv(self, x, xpt):
        """
        Evaluate the curvature of the quadratic function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature of the quadratic function is to be
            evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.

        Returns
        -------
        float
            Curvature of the quadratic function at `x`.

        Notes
        -----
        Although the value can be recovered using `hessp`, the evaluation of
        this method improves the computational efficiency.
        """
        cx = np.inner(self.pq, np.dot(xpt, x) ** 2.0)
        if self._hq is not None:
            cx += np.inner(x, np.dot(self.hq, x))
        return cx

    def shift_expansion_point(self, step, xpt):
        """
        Shift the point around which the quadratic function is defined.

        This method must be called when the index around which the quadratic
        function is defined is modified, or when the point in `xpt` around
        which the quadratic function is defined is modified.

        Parameters
        ----------
        step : numpy.ndarray, shape (n,)
            Displacement from the current point ``xopt`` around which the
            quadratic function is defined. After calling this method, the value
            of the quadratic function at ``xopt + step`` is zero, since the
            constant term of the function is not maintained.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        """
        self._gq += self.hessp(step, xpt)

    def shift_interpolation_points(self, xpt, kopt):
        """
        Update the components of the quadratic function when the origin from
        which the interpolation points are defined is to be displaced.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. The constant term of the quadratic function is not
            maintained, and zero is returned at ``xpt[kopt, :]``.

        Notes
        -----
        Given ``xbase`` the previous origin of the calculations, it is assumed
        that the origin is shifted to ``xbase + xpt[kopt, :]``.
        """
        hxpt = xpt - 0.5 * xpt[np.newaxis, kopt, :]
        temp = np.outer(np.dot(hxpt.T, self.pq), xpt[kopt, :])
        self._hq = self.hq + temp + temp.T

    def update(self, xpt, kopt, xold, bmat, zmat, idz, knew, diff):
        """
        Update the model when a point of the interpolation set is modified.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. The constant term of the quadratic function is not
            maintained, and zero is returned at ``xpt[kopt, :]``.
        xold : numpy.ndarray, shape (n,)
            Previous point around which the quadratic function was defined.
        bmat : numpy.ndarray, shape (npt + n, n)
            Last ``n`` columns of the inverse KKT matrix of interpolation.
        zmat : numpy.ndarray, shape (npt, npt - n - 1)
            Rank factorization matrix of the leading ``npt`` submatrix of the
            inverse KKT matrix of interpolation.
        idz : int
            Number of nonpositive eigenvalues of the leading ``npt`` submatrix
            of the inverse KKT matrix of interpolation. Although its theoretical
            value is always 0, it is designed to tackle numerical difficulties
            caused by ill-conditioned problems.
        knew : int
            Index of the interpolation point that is modified.
        diff : float
            Difference between the evaluation of the previous model and the
            expected value at ``xpt[kopt, :]``.
        """
        # Update the explicit and implicit parts of the Hessian matrix of the
        # quadratic function. The knew-th component of the implicit part of the
        # Hessian matrix is added to the explicit Hessian matrix. Then, the
        # implicit part of the Hessian matrix is modified.
        omega = implicit_hessian(zmat, idz, knew)
        self._hq = self.hq + self.pq[knew] * np.outer(xold, xold)
        self.pq[knew] = 0.0
        self._pq += diff * omega

        # Update the gradient of the model.
        temp = omega * np.dot(xpt, xpt[kopt, :])
        self._gq += diff * (bmat[knew, :] + np.dot(xpt.T, temp))

    def check_model(self, xpt, fval, kopt, stack_level=2):
        """
        Check the interpolation conditions.


        The method checks whether the evaluations of the quadratic function at
        the interpolation points match their expected values.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function. Each row of
            `xpt` stores the coordinates of an interpolation point.
        fval : numpy.ndarray, shape (npt,)
            Evaluations associated with the interpolation points.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. The constant term of the quadratic function is not
            maintained, and zero is returned at ``xpt[kopt, :]``.
        stack_level : int, optional
            Stack level of the warning (the default is 2).

        Warns
        -----
        RuntimeWarning
            The evaluations of the quadratic function do not satisfy the
            interpolation conditions up to a certain tolerance.
        """
        npt = fval.size
        eps = np.finfo(float).eps
        tol = 10.0 * np.sqrt(eps) * npt * np.max(np.abs(fval), initial=1.0)
        diff = 0.0
        for k in range(npt):
            qx = self(xpt[k, :], xpt, kopt)
            diff = max(diff, abs(qx + fval[kopt] - fval[k]))
        if diff > tol:
            stack_level += 1
            message = f'error in interpolation conditions is {diff:e}.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)
