import copy
import logging
import warnings

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.optimize import lsq_linear

from .subproblems import bound_constrained_cauchy_step, bound_constrained_xpt_step, bound_constrained_normal_step, bound_constrained_tangential_step, get_qr_tangential, linearly_constrained_tangential_step
from .utils import huge, max_abs_arrays

_log = logging.getLogger(__name__)


class NonlinearProblem:
    """
    Nonlinear optimization problem.

    Attributes
    ----------
    xl : numpy.ndarray, shape (n,)
        Lower-bound constraints.
    xu : numpy.ndarray, shape (n,)
        Upper-bound constraints.
    aub : numpy.ndarray, shape (m_linear_ub, n)
        Jacobian matrix of the linear inequality constraints.
    bub : numpy.ndarray, shape (m_linear_ub,)
        Right-hand side of the linear inequality constraints ``aub @ x <= bub``.
    aeq : numpy.ndarray, shape (m_linear_eq, n)
        Jacobian matrix of the linear equality constraints.
    beq : numpy.ndarray, shape (m_linear_eq,)
        Right-hand side of the linear equality constraints ``aeq @ x = bub``.
    n_fev : int
        Number of call to `fun` made so far.
    ibd_fixed : numpy.ndarray, shape (n_init,)
        Array indicating the positions of the fixed variables.
    ibd_free : numpy.ndarray, shape (n_init,)
        Array indicating the positions of the free variables.
    x_fixed : numpy.ndarray, shape (n_fixed,)
        Values of the fixed variables, with ``n_fixed = n_init - n``.
    """

    def __init__(self, fun, args, xl, xu, aub, bub, aeq, beq, cub, ceq, options, store_hist):
        self._fun = fun
        self._args = args
        self.xl = xl
        self.xu = xu
        self.aub = aub
        self.bub = bub
        self.aeq = aeq
        self.beq = beq
        self._cub = cub
        self._ceq = ceq
        self.n_fev = 0

        # Remove the nan values from the bound constraints.
        ixl_nan = np.isnan(self.xl)
        if np.any(ixl_nan):
            warnings.warn("xl contains NaN values; they are replaced with -inf")
            self.xl[ixl_nan] = -np.inf
        ixu_nan = np.isnan(self.xu)
        if np.any(ixu_nan):
            warnings.warn("xu contains NaN values; they are replaced with inf")
            self.xu[ixu_nan] = np.inf

        # Remove the nan/inf values from the linear inequality constraints.
        if not np.all(np.isfinite(self.aub)):
            warnings.warn("aub contains NaN and/or infinite values, they are replaced with zero and large finite numbers")
            np.nan_to_num(self.aub, False)
        iub_finite = np.isfinite(self.bub)
        if not np.all(iub_finite):
            warnings.warn("bub contains NaN and/or infinite values, the corresponding constraints are removed")
            self.aub = self.aub[iub_finite, :]
            self.bub = self.bub[iub_finite]

        # Remove the nan/inf values from the linear equality constraints.
        if not np.all(np.isfinite(self.aeq)):
            warnings.warn("aeq contains NaN and/or infinite values, they are replaced with zero and large finite numbers")
            np.nan_to_num(self.aeq, False)
        ieq_finite = np.isfinite(self.beq)
        if not np.all(ieq_finite):
            warnings.warn("beq contains NaN and/or infinite values, the corresponding constraints are removed")
            self.aeq = self.aeq[ieq_finite, :]
            self.beq = self.beq[ieq_finite]

        # Remove the variables that are fixed by the bounds.
        tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
        self.ibd_fixed = (self.xl <= self.xu) & ((self.xu - self.xl) <= tol)
        self.ibd_free = ~self.ibd_fixed
        self.x_fixed = np.maximum(xl[self.ibd_fixed], np.minimum(0.5 * (xl[self.ibd_fixed] + xu[self.ibd_fixed]), xu[self.ibd_fixed]))
        self.xl = self.xl[self.ibd_free]
        self.xu = self.xu[self.ibd_free]
        self.bub -= np.dot(self.aub[:, self.ibd_fixed], self.x_fixed)
        self.aub = self.aub[:, self.ibd_free]
        self.beq -= np.dot(self.aeq[:, self.ibd_fixed], self.x_fixed)
        self.aeq = self.aeq[:, self.ibd_free]

        # The following attributes are set when calling the nonlinear inequality
        # and equality constraint functions for the first time, respectively.
        self._m_nonlinear_ub = 0 if self._cub is None else None
        self._m_nonlinear_eq = 0 if self._ceq is None else None

        # Whether to print the function evaluations.
        self._disp = options["disp"]

        # Store the histories of the points at which the objective and
        # constraint functions have been evaluated, if necessary.
        # TODO: A sliding window mechanism should be implemented to store only
        #  the most recent function evaluations.
        self._store_hist = store_hist
        self._x_hist = []
        self._fun_hist = []
        self._cub_hist = []
        self._ceq_hist = []

    @property
    def n(self):
        """
        Dimension of the problem.
        """
        return self.xl.size

    @property
    def n_init(self):
        """
        Dimension of the problem with the fixed variables.
        """
        return self.ibd_fixed.size

    @property
    def m_linear_ub(self):
        return self.bub.size

    @property
    def m_linear_eq(self):
        return self.beq.size

    @property
    def m_nonlinear_ub(self):
        if self._m_nonlinear_ub is not None:
            return self._m_nonlinear_ub
        else:
            raise AttributeError("Call the nonlinear inequality constraint function to set this attribute")

    @property
    def m_nonlinear_eq(self):
        if self._m_nonlinear_eq is not None:
            return self._m_nonlinear_eq
        else:
            raise AttributeError("Call the nonlinear equality constraint function to set this attribute")

    @property
    def type(self):
        if self._m_nonlinear_ub is not None and self._m_nonlinear_ub > 0 or self._m_nonlinear_eq is not None and self._m_nonlinear_eq > 0:
            # The type of the problem is confirmed.
            return "nonlinearly constrained"
        elif self._m_nonlinear_ub is None and self._cub is not None or self._m_nonlinear_eq is None and self._ceq is not None:
            # The type of the problem is assumed.
            return "nonlinearly constrained"
        elif self.m_linear_ub > 0 or self.m_linear_eq > 0:
            return "linearly constrained"
        elif np.any(self.xl > -np.inf) or np.any(self.xu < -np.inf):
            return "bound-constrained"
        else:
            return "unconstrained"

    def fun(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point of evaluation.

        Returns
        -------
        float
            Objective function value.
        """
        x_complete = self.build_x(x)
        f = float(self._fun(x_complete, *self._args))
        max_f = huge(x.dtype)
        if np.isnan(f) or f > max_f:
            f = max_f
        self.n_fev += 1
        if self._disp:
            print(f"{self._fun.__name__}({x_complete}) = {f}")
        if self._store_hist:
            self._x_hist.append(np.copy(x))
            self._fun_hist.append(f)
        return f

    def cub(self, x):
        """
        Evaluate the nonlinear inequality constraint function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point of evaluation.

        Returns
        -------
        numpy.ndarray, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint function value.
        """
        c = self._eval_con(self._cub, x)
        if self._m_nonlinear_ub is None:
            self._m_nonlinear_ub = c.size
        if self._store_hist and c.size > 0:
            self._cub_hist.append(np.copy(c))
        return c

    def ceq(self, x):
        """
        Evaluate the nonlinear equality constraint function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point of evaluation.

        Returns
        -------
        numpy.ndarray, shape (m_nonlinear_eq,)
            Nonlinear equality constraint function value.
        """
        c = self._eval_con(self._ceq, x)
        if self._m_nonlinear_eq is None:
            self._m_nonlinear_eq = c.size
        if self._store_hist and c.size > 0:
            self._ceq_hist.append(np.copy(c))
        return c

    def merit(self, x, fun_x, cub_x, ceq_x, penalty):
        m_val = fun_x
        if penalty > 0.0:
            c = np.r_[np.maximum(0.0, np.dot(self.aub, x) - self.bub), np.maximum(0.0, cub_x), np.dot(self.aeq, x) - self.beq, ceq_x]
            m_val += penalty * np.linalg.norm(c)
        return m_val

    def resid(self, x, cub_x, ceq_x):
        """
        Evaluate the residual.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point of evaluation.
        cub_x : numpy.ndarray, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint function value.
        ceq_x : numpy.ndarray, shape (m_nonlinear_eq,)
            Nonlinear equality constraint function value.

        Returns
        -------
        float
            The value of the
        """
        cub = np.r_[np.dot(self.aub, x) - self.bub, cub_x]
        ceq = np.r_[np.dot(self.aeq, x) - self.beq, ceq_x]
        cbd = np.r_[x - self.xu, self.xl - x]
        return max(map(lambda array: np.max(array, initial=0.0), [cub, np.abs(ceq), cbd]))

    def build_x(self, x):
        """
        Build the complete variables with the fixed components.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point of evaluation.

        Returns
        -------
        numpy.ndarray, shape (n_init,)
            The complete array `x`, with the fixed variables.
        """
        x_complete = np.empty(self.n_init)
        x_complete[self.ibd_free] = np.maximum(self.xl, np.minimum(x, self.xu))
        x_complete[self.ibd_fixed] = self.x_fixed
        return x_complete

    def _eval_con(self, con, x):
        """
        Evaluate the constraint function `con`.
        """
        if con is not None:
            x_complete = self.build_x(x)
            c = np.atleast_1d(np.squeeze(con(x_complete, *self._args))).astype(float)
            max_con = huge(x.dtype)
            c[np.isnan(c) | (c > max_con)] = max_con
            c[c < -max_con] = -max_con
            if self._disp and c.size > 0:
                print(f"{con.__name__}({x_complete}) = {c}")
        else:
            c = np.asarray([])
        return c


class Models:
    """
    Objective and constraint function quadratic models.

    Attributes
    ----------
    manager : ModelManager
    fun_values : numpy.ndarray, shape (npt,)
    cub_values : numpy.ndarray, shape (npt, m_nonlinear_ub)
    ceq_values : numpy.ndarray, shape (npt, m_nonlinear_eq)
    are_built : bool
        Whether the initial models are built.
    """

    def __init__(self, nlp, x0, options):
        self._nlp = nlp
        self._debug = options["debug"]
        self.manager = self.ModelManager(x0, self._nlp.xl, self._nlp.xu, options["npt"], options["rhobeg"])

        # Evaluate the nonlinear constraints are x0.
        x_eval = self._nlp.build_x(self.manager.base + self.manager.xpt[0, :])
        cub_init = self._nlp.cub(x_eval)
        ceq_init = self._nlp.ceq(x_eval)

        # Evaluate the functions at the interpolation points.
        self.fun_values = np.empty(self.npt)
        self.cub_values = np.empty((self.npt, cub_init.size))
        self.ceq_values = np.empty((self.npt, ceq_init.size))
        self.are_built = False
        for k in range(self.npt):
            x_eval = self._nlp.build_x(self.manager.base + self.manager.xpt[k, :])
            self.fun_values[k] = self._nlp.fun(x_eval)
            if k == 0:
                self.cub_values[k, :] = cub_init
                self.ceq_values[k, :] = ceq_init
            else:
                self.cub_values[k, :] = self._nlp.cub(x_eval)
                self.ceq_values[k, :] = self._nlp.ceq(x_eval)
            tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(self._nlp.xl, self._nlp.xu)
            if self.fun_values[k] <= options["target"] and self._nlp.resid(self.manager.base + self.manager.xpt[k, :], self.cub_values[k, :], self.ceq_values[k, :]) <= tol:
                # The computations must be stopped as a (nearly) feasible
                # interpolation point has an objective function value below the
                # target value.
                break
        else:
            # This else statement is reached when the for loop ended normally,
            # i.e., when the target value on the objective function is not
            # reached by any (almost) feasible interpolation point. The code
            # below builds all the quadratic models.
            self._fun_model = self.Quadratic(self.manager, self.fun_values)
            self._fun_model_alt = copy.deepcopy(self._fun_model)
            self._cub_models = np.empty(self.m_nonlinear_ub, dtype=self.Quadratic)
            self._cub_models_alt = np.empty(self.m_nonlinear_ub, dtype=self.Quadratic)
            self._ceq_models = np.empty(self.m_nonlinear_eq, dtype=self.Quadratic)
            self._ceq_models_alt = np.empty(self.m_nonlinear_eq, dtype=self.Quadratic)
            for i in range(self.m_nonlinear_ub):
                self._cub_models[i] = self.Quadratic(self.manager, self.cub_values[:, i])
                self._cub_models_alt[i] = copy.deepcopy(self._cub_models[i])
            for i in range(self.m_nonlinear_eq):
                self._ceq_models[i] = self.Quadratic(self.manager, self.ceq_values[:, i])
                self._ceq_models_alt[i] = copy.deepcopy(self._ceq_models[i])
            self._check_models()
            self.are_built = True

    @property
    def n(self):
        return self._nlp.n

    @property
    def npt(self):
        return self.manager.npt

    @property
    def m_linear_ub(self):
        return self._nlp.m_linear_ub

    @property
    def m_linear_eq(self):
        return self._nlp.m_linear_eq

    @property
    def m_nonlinear_ub(self):
        return self._nlp.m_nonlinear_ub

    @property
    def m_nonlinear_eq(self):
        return self._nlp.m_nonlinear_eq

    @property
    def k(self):
        return self.manager.k

    @k.setter
    def k(self, k_new):
        if self.manager.k != k_new:
            self._fun_model.change_extension_point(k_new, self.manager, self.fun_values)
            self._fun_model_alt.change_extension_point(k_new, self.manager, self.fun_values)
            for i, (model, model_alt) in enumerate(zip(self._cub_models, self._cub_models_alt)):
                model.change_extension_point(k_new, self.manager, self.cub_values[:, i])
                model_alt.change_extension_point(k_new, self.manager, self.cub_values[:, i])
            for i, (model, model_alt) in enumerate(zip(self._ceq_models, self._ceq_models_alt)):
                model.change_extension_point(k_new, self.manager, self.ceq_values[:, i])
                model_alt.change_extension_point(k_new, self.manager, self.ceq_values[:, i])
            self.manager.k = k_new
        self._check_models()

    def fun_model(self, x):
        return self._fun_model(x, self.manager)

    def fun_model_grad(self, x):
        return self._fun_model.grad(x, self.manager)

    def fun_model_hess(self):
        return self._fun_model.hess(self.manager)

    def fun_model_hess_prod(self, x):
        return self._fun_model.hess_prod(x, self.manager)

    def fun_model_curv(self, x):
        return self._fun_model.curv(x, self.manager)

    def fun_model_alt(self, x):
        return self._fun_model_alt(x, self.manager)

    def fun_model_alt_grad(self, x):
        return self._fun_model_alt.grad(x, self.manager)

    def fun_model_alt_hess(self):
        return self._fun_model_alt.hess(self.manager)

    def fun_model_alt_hess_prod(self, x):
        return self._fun_model_alt.hess_prod(x, self.manager)

    def fun_model_alt_curv(self, x):
        return self._fun_model_alt.curv(x, self.manager)

    def cub_model(self, x, mask=None):
        return np.array([model(x, self.manager) for model in self._get_cub_models(mask)])

    def cub_model_grad(self, x, mask=None):
        return np.reshape([model.grad(x, self.manager) for model in self._get_cub_models(mask)], (-1, self.n))

    def cub_model_hess(self, mask=None):
        return np.reshape([model.hess(self.manager) for model in self._get_cub_models(mask)], (-1, self.n, self.n))

    def cub_model_hess_prod(self, x, mask=None):
        return np.reshape([model.hess_prod(x, self.manager) for model in self._get_cub_models(mask)], (-1, self.n))

    def cub_model_curv(self, x, mask=None):
        return np.array([model.curv(x, self.manager) for model in self._get_cub_models(mask)])

    def cub_model_alt(self, x, mask=None):
        return np.array([model(x, self.manager) for model in self._get_cub_models_alt(mask)])

    def cub_model_alt_grad(self, x, mask=None):
        return np.reshape([model.grad(x, self.manager) for model in self._get_cub_models_alt(mask)], (-1, self.n))

    def cub_model_alt_hess(self, mask=None):
        return np.reshape([model.hess(self.manager) for model in self._get_cub_models_alt(mask)], (-1, self.n, self.n))

    def cub_model_alt_hess_prod(self, x, mask=None):
        return np.reshape([model.hess_prod(x, self.manager) for model in self._get_cub_models_alt(mask)], (-1, self.n))

    def cub_model_alt_curv(self, x, mask=None):
        return np.array([model.curv(x, self.manager) for model in self._get_cub_models_alt(mask)])

    def ceq_model(self, x, mask=None):
        return np.array([model(x, self.manager) for model in self._get_ceq_models(mask)])

    def ceq_model_grad(self, x, mask=None):
        return np.reshape([model.grad(x, self.manager) for model in self._get_ceq_models(mask)], (-1, self.n))

    def ceq_model_hess(self, mask=None):
        return np.reshape([model.hess(self.manager) for model in self._get_ceq_models(mask)], (-1, self.n, self.n))

    def ceq_model_hess_prod(self, x, mask=None):
        return np.reshape([model.hess_prod(x, self.manager) for model in self._get_ceq_models(mask)], (-1, self.n))

    def ceq_model_curv(self, x, mask=None):
        return np.array([model.curv(x, self.manager) for model in self._get_ceq_models(mask)])

    def ceq_model_alt(self, x, mask=None):
        return np.array([model(x, self.manager) for model in self._get_ceq_models_alt(mask)])

    def ceq_model_alt_grad(self, x, mask=None):
        return np.reshape([model.grad(x, self.manager) for model in self._get_ceq_models_alt(mask)], (-1, self.n))

    def ceq_model_alt_hess(self, mask=None):
        return np.reshape([model.hess(self.manager) for model in self._get_ceq_models_alt(mask)], (-1, self.n, self.n))

    def ceq_model_alt_hess_prod(self, x, mask=None):
        return np.reshape([model.hess_prod(x, self.manager) for model in self._get_ceq_models_alt(mask)], (-1, self.n))

    def ceq_model_alt_curv(self, x, mask=None):
        return np.array([model.curv(x, self.manager) for model in self._get_ceq_models_alt(mask)])

    def reset_models(self):
        self._fun_model = copy.deepcopy(self._fun_model_alt)
        self._cub_models = copy.deepcopy(self._cub_models_alt)
        self._ceq_models = copy.deepcopy(self._ceq_models_alt)

    def shift_base(self, delta):
        if np.linalg.norm(self.manager.xpt[self.k, :]) >= 3.0 * delta:
            # Modify the interpolation set and the interpolation system.
            _log.debug("Update the shift of the origin.")
            x_prev = np.copy(self.manager.xpt[self.k, :])
            self.manager.shift_base()

            # Update the models.
            self._fun_model.shift_interpolation_points(self.manager, x_prev)
            self._fun_model_alt.shift_interpolation_points(self.manager, x_prev)
            for model, model_alt in zip(self._cub_models, self._cub_models_alt):
                model.shift_interpolation_points(self.manager, x_prev)
                model_alt.shift_interpolation_points(self.manager, x_prev)
            for model, model_alt in zip(self._ceq_models, self._ceq_models_alt):
                model.shift_interpolation_points(self.manager, x_prev)
                model_alt.shift_interpolation_points(self.manager, x_prev)
            self._check_models()

    def update_interpolation_set(self, k_new, step, fun_x, cub_x, ceq_x):
        self.manager.update_interpolation_system(k_new, step)

        x_new = self.manager.xpt[self.k, :] + step
        x_old = np.copy(self.manager.xpt[k_new, :])
        fun_diff = fun_x - self.fun_model(x_new)
        cub_diff = cub_x - self.cub_model(x_new)
        ceq_diff = ceq_x - self.ceq_model(x_new)
        self.fun_values[k_new] = fun_x
        self.cub_values[k_new, :] = cub_x
        self.ceq_values[k_new, :] = ceq_x
        self.manager.xpt[k_new, :] = x_new
        self._fun_model.update_interpolation_set(self.manager, x_old, k_new, fun_diff)
        self._fun_model_alt = self.Quadratic(self.manager, self.fun_values)
        for i, model in enumerate(self._cub_models):
            model.update_interpolation_set(self.manager, x_old, k_new, cub_diff[i])
            self._cub_models_alt[i] = self.Quadratic(self.manager, self.cub_values[:, i])
        for i, model in enumerate(self._ceq_models):
            model.update_interpolation_set(self.manager, x_old, k_new, ceq_diff[i])
            self._ceq_models_alt[i] = self.Quadratic(self.manager, self.ceq_values[:, i])
        self._check_models()

    def get_improving_step(self, k_new, delta):
        # Determine the k_new-th Lagrange polynomial.
        # Note: the constant c_lag should always be zero, since the point to
        #  remove from the interpolation set should not be the best point so
        #  far. However, we include it here if a future modification to this
        #  method modifies this behavior.
        lag = self.Quadratic(self.manager, k_new)
        # c_lag = lag(self.manager.xpt[self.k, :], self.manager)
        c_lag = 1.0 if k_new == self.k else 0.0
        g_lag = lag.grad(self.manager.xpt[self.k, :], self.manager)

        # Compute a simple constrained Cauchy step.
        xl = self._nlp.xl - self.manager.base - self.manager.xpt[self.k, :]
        xu = self._nlp.xu - self.manager.base - self.manager.xpt[self.k, :]
        step = bound_constrained_cauchy_step(c_lag, g_lag, lambda x: lag.hess_prod(x, self.manager), xl, xu, delta, self._debug)
        lag_values, _, beta = self.manager.get_lag_values(step)
        alpha = self.manager.get_alpha(k_new)
        sigma = lag_values[k_new] ** 2.0 + alpha * beta

        # Compute the solution on the straight lines joining the interpolation
        # points to the k-th one, and choose it if it provides a larger value of
        # the denominator of the updating formula.
        xpt = self.manager.xpt - self.manager.xpt[self.k, np.newaxis, :]
        xpt[[0, self.k], :] = xpt[[self.k, 0], :]
        step_alt = bound_constrained_xpt_step(c_lag, g_lag, lambda x: lag.hess_prod(x, self.manager), xpt[1:, :], xl, xu, delta, self._debug)
        lag_values_alt, _, beta_alt = self.manager.get_lag_values(step_alt)
        sigma_alt = lag_values_alt[k_new] ** 2.0 + alpha * beta_alt
        if abs(sigma_alt) >= abs(sigma):
            step = step_alt
            sigma = sigma_alt

        # Compute a Cauchy step on the tangent space of the active constraints.
        if self._nlp.type in "linearly constrained nonlinearly constrained":
            aub, bub, aeq, beq = self.get_constraint_linearizations(self.manager.xpt[self.k, :])
            tol_bd = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
            tol_ub = 10.0 * np.finfo(float).eps * max(aub.shape) * max_abs_arrays(bub)
            free_xl = xl <= -tol_bd
            free_xu = xu >= tol_bd
            free_ub = bub >= tol_ub

            # Compute the Cauchy step.
            n_act, q = get_qr_tangential(aub, aeq, free_xl, free_xu, free_ub)
            g_lag_proj = np.dot(q[:, n_act:], np.dot(q[:, n_act:].T, g_lag))
            norm_g_lag_proj = np.linalg.norm(g_lag_proj)
            if 0 < n_act < self.n and norm_g_lag_proj > np.finfo(float).tiny * delta:
                step_alt = (delta / norm_g_lag_proj) * g_lag_proj
                if lag.curv(step_alt, self.manager) < 0.0:
                    step_alt = -step_alt

                # Evaluate the constraint violation at the Cauchy step.
                cub = np.dot(aub, step_alt) - bub
                ceq = np.dot(aeq, step_alt) - beq
                cbd = np.r_[step_alt - xu, xl - step_alt]
                resid = max(map(lambda array: np.max(array, initial=0.0), [cub, np.abs(ceq), cbd]))

                # Accept the new step if it is nearly feasible and do not
                # drastically worsen the denominator of the updating formula.
                tol = np.max(np.abs(step_alt[~free_xl]), initial=0.0)
                tol = np.max(np.abs(step_alt[~free_xu]), initial=tol)
                tol = np.max(np.abs(np.dot(aub[~free_ub, :], step_alt)), initial=tol)
                tol = min(10.0 * tol, 1e-2 * np.linalg.norm(step_alt))
                if resid <= tol:
                    lag_values_alt, _, beta_alt = self.manager.get_lag_values(step_alt)
                    sigma_alt = lag_values_alt[k_new] ** 2.0 + alpha * beta_alt
                    if abs(sigma_alt) >= 0.1 * abs(sigma):
                        step = np.maximum(xl, np.minimum(step_alt, xu))

        if self._debug:
            tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
            if np.any(step + tol < xl) or np.any(xu < step - tol):
                warnings.warn("the improving step does not respect the bound constraints")
            if np.linalg.norm(step) > 1.1 * delta:
                warnings.warn("the improving step does not respect the trust-region constraint")

        return step

    def get_second_order_correction_step(self, step, **kwargs):
        x_prev = self.manager.xpt[self.k, :] + step
        aub, bub, aeq, beq = self.get_constraint_linearizations(x_prev)
        xl = self._nlp.xl - self.manager.base - x_prev
        xu = self._nlp.xu - self.manager.base - x_prev
        delta = np.inner(step, step)
        soc_step = bound_constrained_normal_step(aub, bub, aeq, beq, xl, xu, delta, self._debug, **kwargs)

        if self._debug:
            tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
            if np.any(soc_step + tol < xl) or np.any(xu < soc_step - tol):
                warnings.warn("the second-order correction step does not respect the bound constraints")
            if np.linalg.norm(soc_step) > 1.1 * delta:
                warnings.warn("the second-order correction step does not respect the trust-region constraint")

        return soc_step

    def get_constraint_linearizations(self, x):
        cub_model_grad = self.cub_model_grad(x)
        aub = np.r_[self._nlp.aub, cub_model_grad]
        bub = np.r_[self._nlp.bub - np.dot(self._nlp.aub, self.manager.base + x), -self.cub_model(x)]
        ceq_model_grad = self.ceq_model_grad(x)
        aeq = np.r_[self._nlp.aeq, ceq_model_grad]
        beq = np.r_[self._nlp.beq - np.dot(self._nlp.aeq, self.manager.base + x), -self.ceq_model(x)]
        return aub, bub, aeq, beq

    def _get_cub_models(self, mask):
        return self._cub_models if mask is None else self._cub_models[mask]

    def _get_cub_models_alt(self, mask):
        return self._cub_models_alt if mask is None else self._cub_models_alt[mask]

    def _get_ceq_models(self, mask):
        return self._ceq_models if mask is None else self._ceq_models[mask]

    def _get_ceq_models_alt(self, mask):
        return self._ceq_models_alt if mask is None else self._ceq_models_alt[mask]

    def _check_models(self):
        if self._debug:
            self._fun_model.check(self.manager, self.fun_values)
            self._fun_model_alt.check(self.manager, self.fun_values)
            for i, (model, model_alt) in enumerate(zip(self._cub_models, self._cub_models_alt)):
                model.check(self.manager, self.cub_values[:, i])
                model_alt.check(self.manager, self.cub_values[:, i])
            for i, (model, model_alt) in enumerate(zip(self._ceq_models, self._ceq_models_alt)):
                model.check(self.manager, self.ceq_values[:, i])
                model_alt.check(self.manager, self.ceq_values[:, i])

    class ModelManager:
        """
        Manager of quadratic models.

        Attributes
        ----------
        base : numpy.ndarray, shape (n,)
            Shift from the origin in the calculations.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points.
        k : int
            Point around which the quadratic models are expanded.
        """

        def __init__(self, x0, xl, xu, npt, rhobeg):
            # Modify the initial guess in order to avoid conflicts between the
            # bounds and the initial interpolation points. The coordinates of
            # the initial guess should either equal the bound components or
            # allow the projection of the initial trust region onto the
            # components to lie entirely inside the bounds.
            dist_xl = x0 - xl
            very_close_xl = (dist_xl <= 0.5 * rhobeg) & (xl < x0)
            if np.any(very_close_xl):
                x0[very_close_xl] = xl[very_close_xl]
            close_xl = (0.5 * rhobeg < dist_xl) & (dist_xl <= rhobeg) & (xl < x0)
            if np.any(close_xl):
                x0[close_xl] = np.minimum(xl[close_xl] + rhobeg, xu[close_xl])
            dist_xu = xu - x0
            very_close_xu = (dist_xu <= 0.5 * rhobeg) & (xl < x0)
            if np.any(very_close_xu):
                x0[very_close_xu] = xu[very_close_xu]
            close_xu = (0.5 * rhobeg < dist_xu) & (dist_xu <= rhobeg) & (x0 < xu)
            if np.any(close_xu):
                x0[close_xu] = np.maximum(xl[close_xu], xu[close_xu] - rhobeg)

            # The interpolation points are stored in the rows of xpt.
            self.base = x0
            self.xpt = np.zeros((npt, x0.size))
            self.k = 0

            # The following hold the inverse of the coefficient matrix of the
            # KKT system of interpolation. We employ the following
            # representation, designed by Prof. M. J. D. Powell in NEWUOA. The
            # matrix _b holds the last n columns of the inverse matrix, and the
            # matrix _z holds a rank factorization of the leading npy-by-npt
            # submatrix of the inverse matrix, this factorization being
            # _z times diag(dz) times _z.T, where the elements of dz are plus or
            # minus one. Namely, we store the index _idz with dz[:_idz] = -1 and
            # dz[_idz:] = 1. In theory, we should always have _idz = 0 (i.e.,
            # the leading npy-by-npt submatrix of the inverse matrix is positive
            # semidefinite), but this may not be true in practice due to
            # computer rounding errors.
            self._b = np.zeros((self.npt + self.n, self.n))
            self._z = np.zeros((self.npt, self.npt - self.n - 1))
            self._idz = 0

            # Set the initial interpolation set and the corresponding inverse of
            # the coefficient matrix of the KKT system of interpolation. Details
            # on the calculations below are provided in [1]. It is assumed in
            # the calculations that there is no conflict between the bounds and
            # x0. Hence, the components of the initial guess should either equal
            # a bound components or allow the projection of the initial trust
            # region onto the components to lie entirely inside the bounds.
            #
            # [1] M. J. D. Powell. The BOBYQA algorithm for bound constrained
            #     optimization without derivatives. Technical report DAMTP
            #     2009/NA06. Cambridge, UK: Department of Applied Mathematics
            #     and Theoretical Physics, University of Cambridge, 2009.
            for k in range(self.npt):
                if 1 <= k <= self.n:
                    if abs(xu[k - 1] - self.base[k - 1]) <= 0.5 * rhobeg:
                        alpha = -rhobeg
                    else:
                        alpha = rhobeg
                    self.xpt[k, k - 1] = max(xl[k - 1] - self.base[k - 1], min(alpha, xu[k - 1] - self.base[k - 1]))
                    if self.npt <= k + self.n:
                        self._b[0, k - 1] = -1.0 / alpha
                        self._b[k, k - 1] = 1.0 / alpha
                        self._b[self.npt + k - 1, k - 1] = -0.5 * rhobeg ** 2.0
                elif self.n < k <= 2 * self.n:
                    alpha = self.xpt[k - self.n, k - self.n - 1]
                    if abs(xl[k - self.n - 1] - self.base[k - self.n - 1]) <= 0.5 * rhobeg:
                        beta = min(2.0 * rhobeg, xu[k - self.n - 1] - self.base[k - self.n - 1])
                    elif abs(xu[k - self.n - 1] - self.base[k - self.n - 1]) <= 0.5 * rhobeg:
                        beta = max(-2.0 * rhobeg, xl[k - self.n - 1] - self.base[k - self.n - 1])
                    else:
                        beta = -rhobeg
                    self.xpt[k, k - self.n - 1] = max(xl[k - self.n - 1] - self.base[k - self.n - 1], min(beta, xu[k - self.n - 1] - self.base[k - self.n - 1]))
                    self._b[0, k - self.n - 1] = -(alpha + beta) / (alpha * beta)
                    self._b[k, k - self.n - 1] = -0.5 / self.xpt[k - self.n, k - self.n - 1]
                    self._b[k - self.n, k - self.n - 1] = -self._b[0, k - self.n - 1] - self._b[k, k - self.n - 1]
                    self._z[0, k - self.n - 1] = np.sqrt(2.0) / (alpha * beta)
                    self._z[k, k - self.n - 1] = np.sqrt(0.5) / rhobeg ** 2.0
                    self._z[k - self.n, k - self.n - 1] = -self._z[0, k - self.n - 1] - self._z[k, k - self.n - 1]
                elif k > 2 * self.n:
                    shift = (k - self.n - 1) // self.n
                    i = k - (1 + shift) * self.n - 1
                    j = (i + shift) % self.n
                    self.xpt[k, i] = self.xpt[i + 1, i]
                    self.xpt[k, j] = self.xpt[j + 1, j]
                    self._z[0, k - self.n - 1] = 1.0 / rhobeg ** 2.0
                    self._z[k, k - self.n - 1] = 1.0 / rhobeg ** 2.0
                    self._z[i + 1, k - self.n - 1] = -1.0 / rhobeg ** 2.0
                    self._z[j + 1, k - self.n - 1] = -1.0 / rhobeg ** 2.0

        @property
        def n(self):
            return self.xpt.shape[1]

        @property
        def npt(self):
            """
            Number of interpolation points.
            """
            return self.xpt.shape[0]

        @property
        def _z_alt(self):
            """
            Matrix ``_z times diag(dz)``, with dz[:_idz] = -1 and dz[_idz:] = 1.
            """
            return np.c_[-self._z[:, :self._idz], self._z[:, self._idz:]]

        def solve(self, rhs):
            """
            Solve the interpolation system.

            Parameters
            ----------
            rhs : {numpy.ndarray, shape (npt,), int}
                First `npt` elements of the right-hand side of the interpolation
                system. If `rhs` is an integer, it is equivalent to
                ``numpy.eye(1, npt, rhs)`` and hence, it builds the rhs-th least
                Frobenius norm Lagrange polynomial.

            Returns
            -------
            numpy.ndarray, shape (n,)
                Gradient of the quadratic model.
            numpy.ndarray, shape (npt,)
                Implicit Hessian of the quadratic model.
            """
            if isinstance(rhs, (int, np.integer)):
                grad = np.copy(self._b[rhs, :])
                impl_hess = np.dot(self._z, self._z_alt[rhs, :])
            else:
                grad = np.dot(self._b[:self.npt, :].T, rhs)
                impl_hess = np.dot(self._z, np.dot(self._z_alt.T, rhs))
            return grad, impl_hess

        def shift_base(self):
            x_prev = np.copy(self.xpt[self.k, :])

            # Make the changes to _b that do not depend on _z.
            length = 0.25 * np.inner(x_prev, x_prev)
            h_update = np.dot(self.xpt, x_prev) - 2.0 * length
            h_xpt = self.xpt - 0.5 * x_prev[np.newaxis, :]
            for k in range(self.npt):
                step = h_update[k] * h_xpt[k, :] + length * x_prev
                update = np.outer(self._b[k, :], step)
                self._b[self.npt:, :] += update + update.T

            # Revise _b to incorporate the changes that depend on _z.
            update = length * np.outer(x_prev, np.sum(self._z, axis=0)) + np.matmul(h_xpt.T, self._z * h_update[:, np.newaxis])
            for k in range(self._idz):
                self._b[:self.npt, :] -= np.outer(self._z[:, k], update[:, k])
                self._b[self.npt:, :] -= np.outer(update[:, k], update[:, k])
            for k in range(self._idz, self.npt - x_prev.size - 1):
                self._b[:self.npt, :] += np.outer(self._z[:, k], update[:, k])
                self._b[self.npt:, :] += np.outer(update[:, k], update[:, k])

            # Finally, update the interpolation set.
            self.base += self.xpt[self.k, :]
            self.xpt -= x_prev[np.newaxis, :]

        def get_index_to_remove(self, step=None):
            if step is not None:
                alpha = self.get_alpha()
                lag_values, _, beta = self.get_lag_values(step)
                sigma = np.square(lag_values) + beta * alpha
            else:
                sigma = 1.0
            dist_sq = np.sum(np.square((self.xpt - self.xpt[np.newaxis, self.k, :])), axis=1)
            i_max = np.argmax(np.sqrt(np.abs(sigma)) * dist_sq)
            return i_max, np.sqrt(dist_sq[i_max])

        def update_interpolation_system(self, k_new, step):
            lag_values, temp_prod, beta = self.get_lag_values(step)

            # Put zeros in the k_new-th row of _z by applying a sequence of
            # Givens rotations. The remaining updates are performed below.
            rotg, = get_blas_funcs(('rotg',), (self._z,))
            rot, = get_blas_funcs(('rot',), (self._z,))
            jdz = 0
            for j in range(1, self.npt - self.n - 1):
                if j == self._idz:
                    jdz = self._idz
                elif abs(self._z[k_new, j]) > 0.0:
                    c = self._z[k_new, jdz]
                    s = self._z[k_new, j]
                    self._z[:, jdz], self._z[:, j] = rot(self._z[:, jdz], self._z[:, j], *rotg(c, s))
                    self._z[k_new, j] = 0.0

            # Evaluate the denominator in Equation (2.12) of Powell (2004).
            scala = self._z[k_new, 0] if self._idz == 0 else -self._z[k_new, 0]
            scalb = 0.0 if jdz == 0 else self._z[k_new, jdz]
            omega = scala * self._z[:, 0] + scalb * self._z[:, jdz]
            alpha = omega[k_new]
            tau = lag_values[k_new]
            sigma = alpha * beta + tau ** 2.0
            lag_values[k_new] -= 1.0
            b_max = np.max(np.abs(self._b), initial=1.0)
            z_max = np.max(np.abs(self._z), initial=1.0)
            if abs(sigma) < np.finfo(float).tiny * max(b_max, z_max):
                # The denominator of the updating formula is too small to safely
                # divide the coefficients of the KKT matrix of interpolation.
                # Theoretically, the value of abs(sigma) is always positive, and
                # becomes small only for ill-conditioned problems.
                raise ZeroDivisionError("The denominator of the updating formula is zero")

            # Complete the update of the matrix _z.
            reduce = False
            h = np.sqrt(abs(sigma))
            if jdz == 0:
                scala = tau / h
                scalb = self._z[k_new, 0] / h
                self._z[:, 0] = scala * self._z[:, 0] - scalb * lag_values
                if sigma < 0.0:
                    if self._idz == 0:
                        self._idz = 1
                    else:
                        reduce = True
            else:
                kdz = jdz if beta >= 0.0 else 0
                jdz -= kdz
                tempa = self._z[k_new, jdz] * beta / sigma
                tempb = self._z[k_new, jdz] * tau / sigma
                temp = self._z[k_new, kdz]
                scala = 1. / np.sqrt(abs(beta) * temp ** 2.0 + tau ** 2.0)
                scalb = scala * h
                self._z[:, kdz] = tau * self._z[:, kdz] - temp * lag_values
                self._z[:, kdz] *= scala
                self._z[:, jdz] -= tempa * omega + tempb * lag_values
                self._z[:, jdz] *= scalb
                if sigma <= 0.0:
                    if beta < 0.0:
                        self._idz += 1
                    else:
                        reduce = True
            if reduce:
                self._idz -= 1
                self._z[:, [0, self._idz]] = self._z[:, [self._idz, 0]]

            # Update accordingly _b. The copy below is crucial, as the slicing
            # would otherwise return a view of the knew-th row of _b only.
            b_sav = np.copy(self._b[k_new, :])
            for j in range(self.n):
                c = (alpha * temp_prod[j] - tau * b_sav[j]) / sigma
                s = (tau * temp_prod[j] + beta * b_sav[j]) / sigma
                self._b[:self.npt, j] += c * lag_values - s * omega
                self._b[self.npt:self.npt + j + 1, j] += c * temp_prod[:j + 1]
                self._b[self.npt:self.npt + j + 1, j] -= s * b_sav[:j + 1]
                self._b[self.npt + j, :j + 1] = self._b[self.npt:self.npt + j + 1, j]

        def get_lag_values(self, step):
            lag_values = np.empty(self.npt + self.n)
            step_sq = np.inner(step, step)
            x_opt_sq = np.inner(self.xpt[self.k, :], self.xpt[self.k, :])
            step_x_opt = np.inner(step, self.xpt[self.k, :])
            xpt_step = np.dot(self.xpt, step)
            xpt_x_opt = np.dot(self.xpt, self.xpt[self.k, :])
            check = xpt_step * (0.5 * xpt_step + xpt_x_opt)
            temp = np.dot(self._z_alt.T, check)
            beta = np.inner(temp[:self._idz], temp[:self._idz]) - np.inner(temp[self._idz:], temp[self._idz:])
            lag_values[:self.npt] = np.dot(self._b[:self.npt, :], step) + np.dot(self._z, temp)
            lag_values[self.k] += 1.0
            lag_values[self.npt:] = np.dot(self._b[:self.npt, :].T, check)
            bsp = np.inner(lag_values[self.npt:], step)
            lag_values[self.npt:] += np.dot(self._b[self.npt:, :], step)
            bsp += np.inner(lag_values[self.npt:], step)
            beta += step_x_opt ** 2.0 + step_sq * (x_opt_sq + 2.0 * step_x_opt + 0.5 * step_sq) - bsp
            return lag_values[:self.npt], lag_values[self.npt:], beta

        def get_alpha(self, k=None):
            if k is None:
                z_sq = self._z ** 2.0
                return np.sum(z_sq[:, self._idz:], axis=1) - np.sum(z_sq[:, :self._idz], axis=1)
            else:
                z_sq = self._z[k, :] ** 2.0
                return np.sum(z_sq[self._idz:]) - np.sum(z_sq[:self._idz])

    class Quadratic:
        """
        Quadratic model.
        """

        def __init__(self, manager, values):
            if isinstance(values, (int, np.integer)):
                self._q0 = 1.0 if manager.k == values else 0.0
            else:
                self._q0 = values[manager.k]
            self._grad, self._impl_hess = manager.solve(values)
            self._expl_hess = None
            self._grad += self.hess_prod(manager.xpt[manager.k, :], manager)

        def __call__(self, x, manager):
            """
            Evaluate the model.

            Parameters
            ----------
            x : numpy.ndarray, shape (n,)
                Point of evaluation.

            Returns
            -------
            float
                Model value.
            """
            x_diff = x - manager.xpt[manager.k, :]
            q_val = self._q0 + np.inner(self._grad, x_diff) + 0.5 * np.inner(self._impl_hess, np.square(np.dot(manager.xpt, x_diff)))
            if self._expl_hess is not None:
                q_val += 0.5 * np.inner(x_diff, np.dot(self._expl_hess, x_diff))
            return q_val

        def grad(self, x, manager):
            """
            Evaluate the gradient of the model.

            Parameters
            ----------
            x : numpy.ndarray, shape (n,)
                Point of evaluation.

            Returns
            -------
            numpy.ndarray, shape(n,)
                Gradient of the model.
            """
            return self._grad + self.hess_prod(x - manager.xpt[manager.k, :], manager)

        def hess(self, manager):
            """
            Evaluate the Hessian of the model.

            Returns
            -------
            numpy.ndarray, shape(n, n)
                Hessian of the model.
            """
            h_val = np.matmul(manager.xpt.T, self._impl_hess[:, np.newaxis] * manager.xpt)
            if self._expl_hess is not None:
                h_val += self._expl_hess
            return h_val

        def hess_prod(self, x, manager):
            """
            Evaluate the product of the Hessian of the model with any vector.

            Parameters
            ----------
            x : numpy.ndarray, shape (n,)
                Vector to be left-multiplied.

            Returns
            -------
            numpy.ndarray, shape(n,)
                Product of the Hessian of the model with `x`.
            """
            h_val = np.dot(manager.xpt.T, self._impl_hess * np.dot(manager.xpt, x))
            if self._expl_hess is not None:
                h_val += np.dot(self._expl_hess, x)
            return h_val

        def curv(self, x, manager):
            """
            Evaluate the curvature of the model.

            Parameters
            ----------
            x : numpy.ndarray, shape (n,)
                Point of evaluation.

            Returns
            -------
            float
                Curvature of the model.
            """
            c_val = np.inner(self._impl_hess, np.square(np.dot(manager.xpt, x)))
            if self._expl_hess is not None:
                c_val += np.inner(x, np.dot(self._expl_hess, x))
            return c_val

        def change_extension_point(self, k_new, manager, values):
            self._q0 = values[k_new]
            self._grad += self.hess_prod(manager.xpt[k_new, :] - manager.xpt[manager.k, :], manager)

        def shift_interpolation_points(self, manager, x_prev):
            h_xpt = manager.xpt + 0.5 * x_prev[np.newaxis, :]
            update = np.outer(np.dot(h_xpt.T, self._impl_hess), x_prev)
            if self._expl_hess is None:
                self._expl_hess = np.zeros((self._grad.size, self._grad.size))
            self._expl_hess = self._expl_hess + update + update.T

        def update_interpolation_set(self, manager, x_old, k_new, diff):
            # Update the Hessian of the quadratic function.
            grad_new, impl_hess_new = manager.solve(k_new)
            if self._expl_hess is None:
                self._expl_hess = np.zeros((self._grad.size, self._grad.size))
            self._expl_hess += self._impl_hess[k_new] * np.outer(x_old, x_old)
            self._impl_hess[k_new] = 0.0
            self._impl_hess += diff * impl_hess_new

            # Update the gradient of the model.
            temp = impl_hess_new * np.dot(manager.xpt, manager.xpt[manager.k, :])
            self._grad += diff * (grad_new + np.dot(manager.xpt.T, temp))

        def check(self, manager, values):
            """
            Check whether the interpolation conditions are met.
            """
            tol = 10.0 * np.sqrt(np.finfo(float).eps) * manager.npt * max_abs_arrays(values)
            diff = max(map(lambda k: abs(self(manager.xpt[k, :], manager) - values[k]), range(manager.npt)))
            if diff > tol:
                warnings.warn(f"the error in the interpolation conditions is {diff}", RuntimeWarning)


class OptimizationManager:

    def __init__(self, nlp, models):
        self.nlp = nlp
        self.models = models

        # Determine the best interpolation point.
        self.penalty = 0.0
        self._set_best_point()

        # Evaluate the initial Lagrange multipliers.
        self.lm_linear_ub = np.empty(self.m_linear_ub)
        self.lm_linear_eq = np.empty(self.m_linear_eq)
        self.lm_nonlinear_ub = np.empty(self.m_nonlinear_ub)
        self.lm_nonlinear_eq = np.empty(self.m_nonlinear_eq)
        self.set_qp_multipliers()

    @property
    def n(self):
        return self.nlp.n

    @property
    def m_linear_ub(self):
        return self.nlp.m_linear_ub

    @property
    def m_linear_eq(self):
        return self.nlp.m_linear_eq

    @property
    def m_nonlinear_ub(self):
        return self.nlp.m_nonlinear_ub

    @property
    def m_nonlinear_eq(self):
        return self.nlp.m_nonlinear_eq

    @property
    def k_opt(self):
        return self.models.k

    @k_opt.setter
    def k_opt(self, k_new):
        self.models.k = k_new

    @property
    def base(self):
        return self.models.manager.base

    @property
    def x_opt(self):
        return self.models.manager.xpt[self.k_opt, :]

    @property
    def fun_opt(self):
        return self.models.fun_values[self.k_opt]

    @property
    def cub_opt(self):
        return self.models.cub_values[self.k_opt, :]

    @property
    def ceq_opt(self):
        return self.models.ceq_values[self.k_opt, :]

    def lag_model(self, x):
        return self.models.fun_model(x) + np.inner(self.lm_nonlinear_ub, self.models.cub_model(x)) + np.inner(self.lm_nonlinear_eq, self.models.ceq_model(x))

    def lag_model_grad(self, x):
        return self.models.fun_model_grad(x) + np.dot(self.lm_nonlinear_ub, self.models.cub_model_grad(x)) + np.dot(self.lm_nonlinear_eq, self.models.ceq_model_grad(x))

    def lag_model_hess(self):
        return self.models.fun_model_hess() + np.dot(self.lm_nonlinear_ub, self.models.cub_model_hess()) + np.dot(self.lm_nonlinear_eq, self.models.ceq_model_hess())

    def lag_model_hess_prod(self, x):
        return self.models.fun_model_hess_prod(x) + np.dot(self.lm_nonlinear_ub, self.models.cub_model_hess_prod(x)) + np.dot(self.lm_nonlinear_eq, self.models.ceq_model_hess_prod(x))

    def lag_model_curv(self, x):
        return self.models.fun_model_curv(x) + np.inner(self.lm_nonlinear_ub, self.models.cub_model_curv(x)) + np.inner(self.lm_nonlinear_eq, self.models.ceq_model_curv(x))

    def get_trust_region_step(self, delta, options, **kwargs):
        debug = options["debug"]

        # Evaluate the linearizations of the constraints.
        aub, bub, aeq, beq = self.models.get_constraint_linearizations(self.x_opt)
        xl = self.nlp.xl - self.base - self.x_opt
        xu = self.nlp.xu - self.base - self.x_opt

        # Evaluate the normal step of the Byrd-Omojokun approach.
        normal_step = np.zeros(self.nlp.n)
        delta_sav = delta
        if self.nlp.type not in "unconstrained bound-constrained":
            delta *= np.sqrt(0.5)
            normal_step = bound_constrained_normal_step(aub, bub, aeq, beq, xl, xu, kwargs["zeta"] * delta, debug)
            if debug:
                tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
                if np.any(normal_step + tol < xl) or np.any(xu < normal_step - tol):
                    warnings.warn("the normal step does not respect the bound constraint")
                if np.linalg.norm(normal_step) > 1.1 * kwargs["zeta"] * delta:
                    warnings.warn("the normal step does not respect the trust-region constraint")

        # Evaluate the tangential step of the Byrd-Omojokun approach.
        delta = np.sqrt(delta ** 2.0 - np.inner(normal_step, normal_step))
        xl -= normal_step
        xu -= normal_step
        bub = np.maximum(bub - np.dot(aub, normal_step), 0.0)
        if self.nlp.type in "unconstrained bound-constrained":
            g_opt = self.models.fun_model_grad(self.x_opt + normal_step)
            tangential_step = bound_constrained_tangential_step(g_opt, self.models.fun_model_hess_prod, xl, xu, delta, debug)
        else:
            g_opt = self.models.fun_model_grad(self.x_opt) + self.lag_model_hess_prod(normal_step)
            tangential_step = linearly_constrained_tangential_step(g_opt, self.lag_model_hess_prod, xl, xu, aub, bub, aeq, delta, debug)
        if debug:
            tol = 10.0 * np.finfo(float).eps * self.n * max_abs_arrays(xl, xu)
            if np.any(tangential_step + tol < xl) or np.any(xu < tangential_step - tol):
                warnings.warn("the tangential step does not respect the bound constraints")
            if np.linalg.norm(normal_step + tangential_step) > 1.1 * delta_sav:
                warnings.warn("the trial step does not respect the trust-region constraint")

        return normal_step, tangential_step

    def increase_penalty(self, step, **kwargs):
        aub, bub, aeq, beq = self.models.get_constraint_linearizations(self.x_opt)
        viol_diff = np.linalg.norm(np.r_[np.maximum(0.0, -bub), beq]) - np.linalg.norm(np.r_[np.maximum(0.0, np.dot(aub, step) - bub), np.dot(aeq, step) - beq])
        sqp_var = np.inner(step, self.models.fun_model_grad(self.x_opt) + 0.5 * self.lag_model_hess_prod(step))

        threshold = np.linalg.norm(np.r_[self.lm_linear_ub, self.lm_linear_eq, self.lm_nonlinear_ub, self.lm_nonlinear_eq])
        if abs(viol_diff) > np.finfo(float).tiny * abs(sqp_var):
            threshold = max(threshold, sqp_var / viol_diff)
        k_sav = self.k_opt
        if self.penalty <= kwargs["upsilon1"] * threshold:
            self.penalty = kwargs["upsilon2"] * threshold
            self._set_best_point()
        return k_sav == self.k_opt

    def decrease_penalty(self):
        if self.penalty > 0.0:
            rub = np.c_[np.matmul(self.base[np.newaxis, :] + self.models.manager.xpt, self.nlp.aub.T) - self.nlp.bub[np.newaxis, :], self.models.cub_values]
            req = np.matmul(self.base[np.newaxis, :] + self.models.manager.xpt, self.nlp.aeq.T) - self.nlp.beq[np.newaxis, :]
            req = np.c_[req, -req, self.models.ceq_values, -self.models.ceq_values]
            resid = np.c_[rub, req]
            c_min = np.min(resid, axis=0)
            c_max = np.max(resid, axis=0)
            indices = c_min < 2.0 * c_max
            if np.any(indices):
                f_min = np.min(self.models.fun_values)
                f_max = np.max(self.models.fun_values)
                c_min_neg = np.minimum(0.0, c_min[indices])
                denom = np.min(c_max[indices] - c_min_neg)
                if denom > np.finfo(float).tiny * (f_max - f_min):
                    self.penalty = min(self.penalty, (f_max - f_min) / denom)
            else:
                self.penalty = 0.0

    def set_qp_multipliers(self):
        incl_linear_ub = np.dot(self.nlp.aub, self.x_opt) >= self.nlp.bub
        incl_nonlinear_ub = self.cub_opt >= 0.0
        incl_xl = self.nlp.xl - self.base >= self.x_opt
        incl_xu = self.nlp.xu - self.base <= self.x_opt
        m_linear_ub = np.count_nonzero(incl_linear_ub)
        m_nonlinear_ub = np.count_nonzero(incl_nonlinear_ub)
        m_xl = np.count_nonzero(incl_xl)
        m_xu = np.count_nonzero(incl_xu)

        identity = np.eye(self.nlp.n)
        c_jac = np.r_[-identity[incl_xl, :], identity[incl_xu, :], self.nlp.aub[incl_linear_ub, :], self.models.cub_model_grad(self.x_opt, incl_nonlinear_ub), self.nlp.aeq, self.models.ceq_model_grad(self.x_opt)]

        if c_jac.size > 0:
            g_opt = self.models.fun_model_grad(self.x_opt)
            xl_lm = np.full(c_jac.shape[0], -np.inf)
            xl_lm[:m_xl + m_xu + m_linear_ub + m_nonlinear_ub] = 0.0
            res = lsq_linear(c_jac.T, -g_opt, bounds=(xl_lm, np.inf), method='bvls')

            self.lm_linear_ub.fill(0.0)
            self.lm_linear_ub[incl_linear_ub] = res.x[m_xl + m_xu:m_xl + m_xu + m_linear_ub]
            self.lm_nonlinear_ub.fill(0.0)
            self.lm_nonlinear_ub[incl_nonlinear_ub] = res.x[m_xl + m_xu + m_linear_ub:m_xl + m_xu + m_linear_ub + m_nonlinear_ub]
            self.lm_linear_eq[:] = res.x[m_xl + m_xu + m_linear_ub + m_nonlinear_ub:m_xl + m_xu + m_linear_ub + m_nonlinear_ub + self.models.m_linear_eq]
            self.lm_nonlinear_eq[:] = res.x[m_xl + m_xu + m_linear_ub + m_nonlinear_ub + self.models.m_linear_eq:]

    def update_interpolation_set(self, k_new, step, fun_x, cub_x, ceq_x):
        self.models.update_interpolation_set(k_new, step, fun_x, cub_x, ceq_x)
        self._set_best_point()

    def _set_best_point(self):
        k_opt = self.k_opt
        m_opt = self.nlp.merit(self.base + self.x_opt, self.fun_opt, self.cub_opt, self.ceq_opt, self.penalty)
        r_opt = self.nlp.resid(self.base + self.x_opt, self.cub_opt, self.ceq_opt)
        tol = 10.0 * np.finfo(float).eps * self.models.manager.npt * max(1.0, abs(m_opt))
        for k in range(self.models.manager.npt):
            if k != self.models.k:
                m_val = self.nlp.merit(self.base + self.models.manager.xpt[k, :], self.models.fun_values[k], self.models.cub_values[k, :], self.models.ceq_values[k, :], self.penalty)
                r_val = self.nlp.resid(self.base + self.models.manager.xpt[k, :], self.models.cub_values[k, :], self.models.ceq_values[k, :])
                if m_val < m_opt or abs(m_val - m_opt) <= tol and r_val < r_opt:
                    k_opt = k
                    m_opt = m_val
        self.k_opt = k_opt
