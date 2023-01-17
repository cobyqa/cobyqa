import logging
import warnings

import numpy as np

from .structs import Models, NonlinearProblem, OptimizationManager
from .utils import OptimizeResult, max_abs_arrays

_log = logging.getLogger(__name__)

EXIT_RHOEND_SUCCESS = 0
EXIT_TARGET_SUCCESS = 1
EXIT_FTOL_ABS_SUCCESS = 2
EXIT_FTOL_REL_SUCCESS = 3
EXIT_XTOL_ABS_SUCCESS = 4
EXIT_XTOL_REL_SUCCESS = 5
EXIT_FIXED_SUCCESS = 9
EXIT_MAXFEV_WARNING = 6
EXIT_MAXITER_WARNING = 7
EXIT_LINALG_ERROR = 8
EXIT_BOUNDS_INFEASIBLE_ERROR = -1


def minimize(fun, x0, args=(), xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, options=None, **kwargs):
    r"""
    Minimize a real-valued function.

    The minimization can be subject to bound, linear-inequality,
    linear-equality, nonlinear-inequality, and nonlinear-equality constraints.
    Although the solver may encounter infeasible points (including the initial
    guess `x0`), the bounds constraints (if any) are always respected.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Parameters of the objective and constraint functions.
    xl : array_like, shape (n,), optional
        Lower-bound constraints on ``x``. Use ``-numpy.inf`` to disable the
        bound constraints on some variables.
    xu : array_like, shape (n,), optional
        Upper-bound constraints on ``x``. Use ``numpy.inf`` to disable the bound
        constraints on some variables.
    aub : array_like, shape (mlub, n), optional
        Jacobian matrix of the linear inequality constraints. Each row of `aub`
        stores the gradient of a linear inequality constraint.
    bub : array_like, shape (mlub,)
        Right-hand side of the linear inequality constraints ``aub @ x <= bub``.
    aeq : array_like, shape (mleq, n), optional
        Jacobian matrix of the linear equality constraints. Each row of `aeq`
        stores the gradient of a linear equality constraint.
    beq : array_like, shape (mleq,)
        Right-hand side of the linear equality constraints ``aeq @ x = beq``.
    cub : callable
        Nonlinear inequality constraint function.

            ``cub(x, *args) -> array_like, shape (mnlub,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    ceq : callable
        Nonlinear equality constraint function.

            ``ceq(x, *args) -> array_like, shape (mnleq,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    options : dict, optional
        Options to forward to the solver. Accepted options are:

            rhobeg : float, optional
                Initial trust-region radius (default is 1.0).
            rhoend : float, optional
                Final trust-region radius (default is 1e-6).
            npt : int, optional
                Number of interpolation points (default is ``2 * n + 1``).
            maxfev : int, optional
                Maximum number of function evaluations (default is ``500 * n``).
            maxiter: int, optional
                Maximum number of iterations (default is ``1000 * n``).
            target : float, optional
                Target value on the objective function (default is
                ``-numpy.inf``). If the solver encounters a (nearly) feasible
                point at which the objective function evaluation is below the
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
                Whether to print pieces of information on the execution of the
                optimizer (default is False).
            debug : bool, optional
                Whether to make debugging tests during the execution, which is
                not recommended in production (default is False).

    Returns
    -------
    OptimizeResult
        Result of the optimizer. Important attributes are: ``x`` the solution
        point, ``success`` a flag indicating whether the optimization terminated
        successfully, and ``message`` a description of the termination status of
        the optimization. See `OptimizeResult` for details.

    Other Parameters
    ----------------
    store_hist : bool, optional
        Whether to store the histories of the points at which the objective and
        constraint functions have been evaluated (default is False).
    eta1 : float, optional
        If the trust-region ratio is smaller than or equal to `eta1`, then the
        trust-region radius is decreased (default is 0.1).
    eta2 : float, optional
        If the trust-region ratio is larger than `eta2`, then the trust-region
        radius is increased (default is 0.7).
    eta3 : float, optional
        The lower bound on the trust-region radius is considered small if it is
        smaller than or equal to ``eta3 * options["rhoend"]`` (default is 16).
    eta4 : float, optional
        The lower bound on the trust-region radius is considered large if it is
        larger than ``eta4 * options["rhoend"]`` (default is 250).
    eta5 : float, optional
        If the trust-region ratio is larger than `eta5`, then it is considered
        too large for restarting the trust-region models (default is 0.01).
    upsilon1 : float, optional
        If the penalty parameter is smaller than or equal to `upsilon1` times
        the smallest theoretical threshold, it is increased (default is 1.5).
    upsilon2 : float, optional
        Factor by which the penalty parameter is increased (default is 2).
    theta1 : float, optional
        Factor by which the trust-region radius is decreased (default is 0.5).
    theta2 : float, optional
        If the trust-region radius is smaller than or equal to `theta2` times
        the lower bound on the trust-region radius, then it is set to the lower
        bound on the trust-region radius (default is 1.4).
    theta3 : float, optional
        Factor by which the trust-region radius is increased (default is
        :math:`\sqrt{2}`).
    theta4 : float, optional
        An empirical factor to increase the trust-region radius (default is 2).
    theta5 : float, optional
        Factor by which the lower bound on the trust-region radius is decreased
        (default is 0.1).
    zeta : float, optional
        Factor by which the trust-region radius is decreased in the normal
        subproblem of the Byrd-Omojokun approach (default is 0.8).

    References
    ----------
    .. [1] J. Nocedal and S. J. Wright. Numerical Optimization. Second. Springer
       Ser. Oper. Res. Financ. Eng. New York, NY, US: Springer, 2006.
    .. [2] M. J. D. Powell. "A direct search optimization method that models the
       objective and constraint functions by linear interpolation." In: Advances
       in Optimization and Numerical Analysis. Edited by S. Gomez and J. P.
       Hennart. Dordrecht, NL: Springer, 1994, pages 51–67.

    Examples
    --------
    We consider the problem of minimizing the Rosenbrock function as implemented
    in the `scipy.optimize` module.

    .. testsetup::

        import numpy as np
        np.set_printoptions(precision=1, suppress=True)

    >>> from scipy.optimize import rosen
    >>> from cobyqa import minimize

    An application of the `minimize` function in the unconstrained case may be:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0)
    >>> res.x
    array([1., 1., 1., 1., 1.])

    We now consider Example 16.4 of [1]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \R^2}   & \quad (x_1 - 1)^2 + (x_2 - 2.5)^2\\
            \text{s.t.}         & \quad -x_1 + 2x_2 \le 2,\\
                                & \quad x_1 + 2x_2 \le 6,\\
                                & \quad x_1 - 2x_2 \le 2,\\
                                & \quad x_1 \ge 0,\\
                                & \quad x_2 \ge 0.
        \end{aligned}

    Its objective function can be implemented as:

    >>> def quadratic(x):
    ...     return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0

    This problem can be solved using `minimize` as:

    >>> x0 = [2.0, 0.0]
    >>> xl = [0.0, 0.0]
    >>> aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    >>> bub = [2.0, 6.0, 2.0]
    >>> res = minimize(quadratic, x0, xl=xl, aub=aub, bub=bub)
    >>> res.x
    array([1.4, 1.7])

    Moreover, although clearly unreasonable in this case, the constraints can
    also be provided as:

    >>> def cub(x):
    ...     c1 = -x[0] + 2.0 * x[1] - 2.0
    ...     c2 = x[0] + 2.0 * x[1] - 6.0
    ...     c3 = x[0] - 2.0 * x[1] - 2.0
    ...     return [c1, c2, c3]

    This problem can be solved using `minimize` as:

    >>> res = minimize(quadratic, x0, xl=xl, cub=cub)
    >>> res.x
    array([1.4, 1.7])

    To conclude, let us consider Problem G of [2]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \R^3}   & \quad f(x) = x_3\\
            \text{s.t.}         & \quad -5x_1 + x_2 - x_3 \le 0,\\
                                & \quad 5x_1 + x_2 - x_3 \le 0,\\
                                & \quad x_1^2 + x_2^2 + 4x_2 - x_3 \le 0.
        \end{aligned}

    Its only nonlinear constraints can be implemented in Python as:

    >>> def cub(x):
    ...     return x[0] ** 2.0 + x[1] ** 2.0 + 4.0 * x[1] - x[2]

    This problem can be solved using `minimize` as:

    >>> x0 = [1.0, 1.0, 1.0]
    >>> aub = [[-5.0, 1.0, -1.0], [5.0, 1.0, -1.0]]
    >>> bub = [0.0, 0.0]
    >>> res = minimize(lambda x: x[2], x0, aub=aub, bub=bub, cub=cub)
    >>> res.x
    array([ 0., -3., -3.])
    """
    # Process the initial guess.
    x0 = _process_1d(x0, "x0")
    n = x0.size

    # Process the arguments of the objective function.
    if not isinstance(args, tuple):
        args = (args,)

    # Process the bound constraints.
    if xl is None:
        xl = np.full_like(x0, -np.inf)
    xl = _process_1d(xl, "xl", x0.size)
    if xu is None:
        xu = np.full_like(x0, np.inf)
    xu = _process_1d(xu, "xu", x0.size)

    # Process the linear inequality constraints.
    if aub is None:
        aub = np.empty((0, n))
    aub = _process_2d(aub, "aub", n)
    if bub is None:
        bub = np.empty(0)
    bub = _process_1d(bub, "bub", aub.shape[0])

    # Process the linear equality constraints.
    if aeq is None:
        aeq = np.empty((0, n))
    aeq = _process_2d(aeq, "aeq", n)
    if beq is None:
        beq = np.empty(0)
    beq = _process_1d(beq, "beq", aeq.shape[0])

    # Process the options.
    if options is None:
        options = {}
    options = dict(options)

    # Set the default constants.
    _set_default_constants(kwargs)

    # Set the default options that must be set before building the nonlinear
    # optimization problem structure below.
    _set_early_default_options(options)

    # Construct the nonlinear optimization problem structure.
    nlp = NonlinearProblem(fun, args, xl, xu, aub, bub, aeq, beq, cub, ceq, options, kwargs["store_hist"])

    # Project the initial guess onto the bound constraints.
    x0 = x0[nlp.ibd_free]
    if np.all(nlp.xl <= nlp.xu):
        x0 = np.maximum(nlp.xl, np.minimum(x0, nlp.xu))

    # Set the default options.
    _set_default_options(options, nlp)

    # Start the computations.
    n_iter = 0
    result = OptimizeResult()
    if np.any(nlp.xu < nlp.xl):
        # The bound constraints are infeasible.
        x_opt = x0
        g_opt = np.full_like(x_opt, np.nan)
        result.status = EXIT_BOUNDS_INFEASIBLE_ERROR
        result.fun = nlp.fun(x_opt)
        result.success = False
        result.maxcv = nlp.resid(x_opt, nlp.cub(x_opt), nlp.ceq(x_opt))
    elif nlp.n == 0:
        # All the variables have been fixed by the bounds.
        x_opt = np.empty(0)
        g_opt = np.empty(0)
        max_cv = nlp.resid(x_opt, nlp.cub(x_opt), nlp.ceq(x_opt))
        result.status = EXIT_FIXED_SUCCESS
        result.fun = nlp.fun(x_opt)
        result.maxcv = max_cv
        result.success = max_cv <= 10.0 * np.finfo(float).eps
    else:
        models = Models(nlp, x0, options)
        if not models.are_built:
            # The target value has been reached by a nearly feasible point.
            x_opt = models.manager.base + models.manager.xpt[nlp.n_fev - 1, :]
            g_opt = np.full_like(x_opt, np.nan)
            result.status = EXIT_TARGET_SUCCESS
            result.fun = models.fun_values[nlp.n_fev - 1]
            result.success = True
            if nlp.type != "unconstrained":
                result.maxcv = nlp.resid(x_opt, models.cub_values[nlp.n_fev - 1, :], models.ceq_values[nlp.n_fev - 1, :])
        else:
            # Set the manager of the optimization process.
            manager = OptimizationManager(nlp, models)

            # Start the main loop.
            _log.debug("Start the main loop")
            rho = options["rhobeg"]
            delta = rho
            k_new = -1
            n_short_steps = 0
            n_very_short_steps = 0
            n_alt_models = 0
            fun_sav = manager.fun_opt
            x_sav = manager.base + manager.x_opt
            while True:
                if n_iter >= options["maxiter"]:
                    x_opt = manager.base + manager.x_opt
                    result.status = EXIT_MAXITER_WARNING
                    result.fun = manager.fun_opt
                    result.success = False
                    if nlp.type != "unconstrained":
                        result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                    break
                n_iter += 1

                # Update the shift of the origin.
                models.shift_base(delta)

                test_short_step = 0.5
                test_very_short_step = 0.1
                if nlp.type not in "unconstrained bound-constrained":
                    test_short_step *= np.sqrt(0.5)
                    test_very_short_step *= np.sqrt(0.5)
                reduce_rho = False
                improve_geometry = False

                # Evaluate the trial step.
                delta_sav = delta
                normal_step, tangential_step = manager.get_trust_region_step(delta, options, **kwargs)
                step = normal_step + tangential_step
                _log.debug(f"Trial point: {manager.base + manager.x_opt + step}")
                s_norm = np.linalg.norm(step)
                attempt_eval_fun = s_norm > test_short_step * delta

                if not attempt_eval_fun:
                    delta *= kwargs["theta1"]
                    if delta <= kwargs["theta2"] * rho:
                        delta = rho
                    if delta_sav > rho:
                        n_short_steps = 0
                        n_very_short_steps = 0
                    else:
                        n_short_steps += 1
                        if s_norm >= test_short_step * rho:
                            n_short_steps = 0
                        n_very_short_steps += 1
                        if s_norm >= test_very_short_step * rho:
                            n_very_short_steps = 0
                    reduce_rho = delta_sav <= rho and (n_short_steps >= 5 or n_very_short_steps >= 3)
                    if reduce_rho:
                        n_short_steps = 0
                        n_very_short_steps = 0
                    else:
                        k_new, dist_new = models.manager.get_index_to_remove()
                        improve_geometry = dist_new > max(delta, 2.0 * rho)
                else:
                    # Increase the penalty parameter if necessary. The method
                    # `OptimizationManager.increase_penalty` returns True if and
                    # only if the index of the optimal point does not change.
                    same_opt_point = manager.increase_penalty(step, **kwargs)
                    _log.debug(f"{manager.penalty=}")
                    if same_opt_point:
                        # Evaluate the objective and constraint functions.
                        if nlp.n_fev >= options["maxfev"]:
                            x_opt = manager.base + manager.x_opt
                            result.status = EXIT_MAXFEV_WARNING
                            result.fun = manager.fun_opt
                            result.success = False
                            if nlp.type != "unconstrained":
                                result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                            break
                        x_new = manager.x_opt + step
                        x_eval = nlp.build_x(manager.base + x_new)
                        fun_x = nlp.fun(x_eval)
                        cub_x = nlp.cub(x_eval)
                        ceq_x = nlp.ceq(x_eval)
                        max_cv = nlp.resid(manager.base + x_new, cub_x, ceq_x)
                        tol = 10.0 * np.finfo(float).eps * nlp.n * max_abs_arrays(nlp.xl, nlp.xu)
                        if fun_x <= options["target"] and max_cv <= tol:
                            # The computations must be stopped as the trial
                            # point is (nearly) feasible and has an objective
                            # function value below the target value.
                            x_opt = manager.base + x_new
                            result.status = EXIT_TARGET_SUCCESS
                            result.fun = fun_x
                            result.success = True
                            if nlp.type != "unconstrained":
                                result.maxcv = max_cv
                            break

                        # Evaluate the merit function values.
                        merit_old = nlp.merit(manager.base + manager.x_opt, manager.fun_opt, manager.cub_opt, manager.ceq_opt, manager.penalty)
                        merit_new = nlp.merit(manager.base + x_new, fun_x, cub_x, ceq_x, manager.penalty)
                        merit_model_old = nlp.merit(manager.base + manager.x_opt, 0.0, models.cub_model(manager.x_opt), models.ceq_model(manager.x_opt), manager.penalty)
                        merit_model_new = nlp.merit(manager.base + x_new, np.inner(step, models.fun_model_grad(manager.x_opt) + 0.5 * manager.lag_model_hess_prod(step)), models.cub_model(manager.x_opt) + np.dot(models.cub_model_grad(manager.x_opt), step), models.ceq_model(manager.x_opt) + np.dot(models.ceq_model_grad(manager.x_opt), step), manager.penalty)

                        # Perform a second-order correction step to prevent the
                        # Maratos effect to occur.
                        if nlp.type == "nonlinearly constrained" and merit_new > merit_old and np.linalg.norm(normal_step) <= kwargs["zeta"] ** 2.0 * delta:
                            soc_step = models.get_second_order_correction_step(step, **kwargs)
                            if np.linalg.norm(soc_step) > 0.0:
                                step += soc_step

                                # Evaluate the objective and constraint
                                # functions.
                                if nlp.n_fev >= options["maxfev"]:
                                    x_opt = manager.base + manager.x_opt
                                    result.status = EXIT_MAXFEV_WARNING
                                    result.fun = manager.fun_opt
                                    result.success = False
                                    if nlp.type != "unconstrained":
                                        result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                                    break
                                x_new = manager.x_opt + step
                                x_eval = nlp.build_x(manager.base + x_new)
                                fun_x = nlp.fun(x_eval)
                                cub_x = nlp.cub(x_eval)
                                ceq_x = nlp.ceq(x_eval)
                                max_cv = nlp.resid(manager.base + x_new, cub_x, ceq_x)
                                tol = 10.0 * np.finfo(float).eps * nlp.n * max_abs_arrays(nlp.xl, nlp.xu)
                                if fun_x <= options["target"] and max_cv <= tol:
                                    # The computations must be stopped as the
                                    # trial point is (nearly) feasible and has
                                    # an objective function value below the
                                    # target value.
                                    x_opt = manager.base + x_new
                                    result.status = EXIT_TARGET_SUCCESS
                                    result.fun = fun_x
                                    result.success = True
                                    if nlp.type != "unconstrained":
                                        result.maxcv = max_cv
                                    break

                                # Evaluate the model merit function values.
                                merit_model_old = nlp.merit(manager.base + manager.x_opt, 0.0, models.cub_model(manager.x_opt), models.ceq_model(manager.x_opt), manager.penalty)
                                merit_model_new = nlp.merit(manager.base + x_new, np.inner(step, models.fun_model_grad(manager.x_opt) + 0.5 * manager.lag_model_hess_prod(step)), models.cub_model(manager.x_opt) + np.dot(models.cub_model_grad(manager.x_opt), step), models.ceq_model(manager.x_opt) + np.dot(models.ceq_model_grad(manager.x_opt), step), manager.penalty)

                        # Calculate the trust-region ratio.
                        if abs(merit_model_old - merit_model_new) > np.finfo(float).tiny * abs(merit_old - merit_new):
                            ratio = (merit_old - merit_new) / abs(merit_model_old - merit_model_new)
                        else:
                            ratio = -1.0

                        # Update the Lagrange multipliers.
                        manager.set_qp_multipliers()

                        # Choose an interpolation point to remove.
                        k_new = models.manager.get_index_to_remove(step)[0]

                        # Update the interpolation set.
                        try:
                            manager.update_interpolation_set(k_new, step, fun_x, cub_x, ceq_x)
                        except ZeroDivisionError:
                            x_opt = manager.base + manager.x_opt
                            result.status = EXIT_LINALG_ERROR
                            result.fun = manager.fun_opt
                            result.success = False
                            if nlp.type != "unconstrained":
                                result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                            break
                        x_diff = np.linalg.norm(x_eval - x_sav)
                        if abs(fun_x - fun_sav) <= min(options["ftol_abs"], options["ftol_rel"] * max(abs(fun_sav), 1.0)) or x_diff <= min(options["xtol_abs"], options["xtol_rel"] * max(np.linalg.norm(x_sav), 1.0)):
                            x_opt = manager.base + x_new
                            result.fun = fun_x
                            if abs(fun_x - fun_sav) <= options["ftol_abs"]:
                                result.status = EXIT_FTOL_ABS_SUCCESS
                            elif abs(fun_x - fun_sav) <= options["ftol_rel"] * max(abs(fun_sav), 1.0):
                                result.status = EXIT_FTOL_REL_SUCCESS
                            elif x_diff <= options["xtol_abs"]:
                                result.status = EXIT_XTOL_ABS_SUCCESS
                            else:
                                result.status = EXIT_XTOL_REL_SUCCESS
                            result.success = True
                            if nlp.type != "unconstrained":
                                result.maxcv = nlp.resid(manager.base + x_new, cub_x, ceq_x)
                            break
                        fun_sav = fun_x
                        x_sav = x_eval

                        # Update the trust-region radius.
                        if ratio <= kwargs["eta1"]:
                            delta *= kwargs["theta1"]
                        elif ratio <= kwargs["eta2"]:
                            delta = max(kwargs["theta1"] * delta, s_norm)
                        else:
                            delta = min(kwargs["theta3"] * delta, max(kwargs["theta1"] * delta, kwargs["theta4"] * s_norm))
                        if delta <= kwargs["theta2"] * rho:
                            delta = rho

                        # Attempt to replace the models by the alternative ones.
                        if delta <= rho:
                            if ratio > kwargs["eta5"]:
                                n_alt_models = 0
                            else:
                                n_alt_models += 1
                                grad = models.fun_model_grad(manager.x_opt)
                                grad_alt = models.fun_model_alt_grad(manager.x_opt)
                                if np.linalg.norm(grad) < 10.0 * np.linalg.norm(grad_alt):
                                    n_alt_models = 0
                                if n_alt_models >= 3:
                                    models.reset_models()
                                    n_alt_models = 0

                        # TODO
                        k_new, dist_new = models.manager.get_index_to_remove()
                        reduce_rho = delta_sav <= rho and ratio <= kwargs["eta1"] and dist_new <= max(delta, 2.0 * rho)
                        improve_geometry = ratio <= kwargs["eta1"] and dist_new > max(delta, 2.0 * rho)
                    else:
                        _log.debug("Increasing the penalty changed the optimal point")

                if reduce_rho:
                    if rho <= options["rhoend"]:
                        x_opt = manager.base + manager.x_opt
                        result.status = EXIT_RHOEND_SUCCESS
                        result.fun = manager.fun_opt
                        result.success = True
                        if nlp.type != "unconstrained":
                            result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                        break
                    delta *= kwargs["theta1"]
                    if kwargs["eta4"] * options["rhoend"] < rho:
                        rho *= kwargs["theta5"]
                    elif kwargs["eta3"] * options["rhoend"] < rho <= kwargs["eta4"] * options["rhoend"]:
                        rho = np.sqrt(rho * options["rhoend"])
                    else:
                        rho = options["rhoend"]
                    delta = max(delta, rho)
                    manager.decrease_penalty()
                    if options["disp"]:
                        print()
                        print(f"New trust-region radius: {rho}.")
                        print(f'Number of function evaluations: {nlp.n_fev}.')
                        print(f'Least value of {fun.__name__}: {manager.fun_opt}.')
                        if nlp.type != "unconstrained":
                            max_cv = nlp.resid(manager.base + manager.x_opt, manager.cub_opt, manager.ceq_opt)
                            print(f'Maximum constraint violation: {max_cv}.')
                        print(f'Corresponding point: {manager.base + manager.x_opt}.')
                        print()

                if improve_geometry:
                    step = models.get_improving_step(k_new, max(kwargs["theta5"] * delta, rho))

                    # Evaluate the objective and constraint functions.
                    if nlp.n_fev >= options["maxfev"]:
                        x_opt = manager.base + manager.x_opt
                        result.status = EXIT_MAXFEV_WARNING
                        result.fun = manager.fun_opt
                        result.success = False
                        if nlp.type != "unconstrained":
                            result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                        break
                    x_new = manager.x_opt + step
                    x_eval = nlp.build_x(manager.base + x_new)
                    fun_x = nlp.fun(x_eval)
                    cub_x = nlp.cub(x_eval)
                    ceq_x = nlp.ceq(x_eval)
                    max_cv = nlp.resid(manager.base + x_new, cub_x, ceq_x)
                    tol = 10.0 * np.finfo(float).eps * nlp.n * max_abs_arrays(nlp.xl, nlp.xu)
                    if fun_x <= options["target"] and max_cv <= tol:
                        # The computations must be stopped as the trial
                        # point is (nearly) feasible and has an objective
                        # function value below the target value.
                        x_opt = manager.base + x_new
                        result.status = EXIT_TARGET_SUCCESS
                        result.fun = fun_x
                        result.success = True
                        if nlp.type != "unconstrained":
                            result.maxcv = max_cv
                        break

                    # Update the interpolation set.
                    try:
                        manager.update_interpolation_set(k_new, step, fun_x, cub_x, ceq_x)
                    except ZeroDivisionError:
                        x_opt = manager.base + manager.x_opt
                        result.status = EXIT_LINALG_ERROR
                        result.fun = manager.fun_opt
                        result.success = False
                        if nlp.type != "unconstrained":
                            result.maxcv = nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
                        break

            g_opt = models.fun_model_grad(x_opt - manager.base)

    # Build the result structure and return.
    g_complete = np.full(nlp.n_init, np.nan)
    g_complete[nlp.ibd_free] = g_opt
    result.jac = g_complete
    result.message = {
        EXIT_RHOEND_SUCCESS: "Lower bound for the trust-region radius has been reached.",
        EXIT_TARGET_SUCCESS: "Target function value has been achieved.",
        EXIT_FTOL_ABS_SUCCESS: "Absolute tolerance on the objective function has been reached.",
        EXIT_FTOL_REL_SUCCESS: "Relative tolerance on the objective function has been reached.",
        EXIT_XTOL_ABS_SUCCESS: "Absolute tolerance on the decision variables has been reached.",
        EXIT_XTOL_REL_SUCCESS: "Relative tolerance on the decision variables has been reached.",
        EXIT_FIXED_SUCCESS: "All variables are fixed by the constraints.",
        EXIT_MAXFEV_WARNING: "Maximum number of function evaluations has been exceeded.",
        EXIT_MAXITER_WARNING: "Maximum number of iterations has been exceeded.",
        EXIT_LINALG_ERROR: "Denominator of the updating formula is zero.",
        EXIT_BOUNDS_INFEASIBLE_ERROR: "Bound constraints are infeasible.",
    }.get(result.status, "Unknown exit status.")
    result.nfev = nlp.n_fev
    result.nit = n_iter
    result.x = nlp.build_x(x_opt)
    if options["disp"]:
        print()
        print(result.message)
        print(f"Number of function evaluations: {result.nfev}.")
        print(f"Least value of {fun.__name__}: {result.fun}.")
        if nlp.type != "unconstrained":
            print(f"Maximum constraint violation: {result.maxcv}.")
        print(f"Corresponding point: {result.x}.")
        print()
    return result


def _process_1d(array, name, size=None):
    """
    Preprocess a one-dimensional array.
    """
    array = np.atleast_1d(np.squeeze(array)).astype(float)
    if array.ndim != 1:
        warnings.warn(f"{name} has {array.ndim} dimensions; it will be flattened", RuntimeWarning)
        array = array.flatten()
    if size is not None and array.size != size:
        raise ValueError(f"{name} has {array.size} elements ({size} expected)")
    return array


def _process_2d(array, name, n_col):
    """
    Preprocess a two-dimensional array.
    """
    array = np.atleast_2d(array).astype(float)
    if array.ndim != 2:
        raise ValueError(f"{name} has {array.ndim} dimensions (2 expected)")
    elif array.shape[1] != n_col:
        raise ValueError(f"{name} has {array.shape[1]} column(s) ({n_col} expected)")
    return array


def _eval_functions(manager, step, options, result):
    x_opt = manager.base + manager.x_opt
    if manager.nlp.n_fev >= options["maxfev"]:
        result.status = EXIT_MAXFEV_WARNING
        result.fun = manager.fun_opt
        result.success = False
        if manager.nlp.type != "unconstrained":
            result.maxcv = manager.nlp.resid(x_opt, manager.cub_opt, manager.ceq_opt)
        raise StopIteration
    x_new = manager.x_opt + step
    x_eval = manager.nlp.build_x(manager.base + x_new)
    fun_x = manager.nlp.fun(x_eval)
    cub_x = manager.nlp.cub(x_eval)
    ceq_x = manager.nlp.ceq(x_eval)
    max_cv = manager.nlp.resid(manager.base + x_new, cub_x, ceq_x)
    tol = 10.0 * np.finfo(float).eps * manager.nlp.n * max_abs_arrays(manager.nlp.xl, manager.nlp.xu)
    if fun_x <= options["target"] and max_cv <= tol:
        # The computations must be stopped as the trial
        # point is (nearly) feasible and has an objective
        # function value below the target value.
        x_opt = manager.base + x_new
        result.status = EXIT_TARGET_SUCCESS
        result.fun = fun_x
        result.success = True
        if manager.nlp.type != "unconstrained":
            result.maxcv = max_cv
        raise StopIteration
    return fun_x, cub_x, ceq_x


def _set_early_default_options(options):
    """
    Set the default options that are needed to build the `NonlinearProblem`.
    """
    options.setdefault("disp", False)


def _set_default_options(options, nlp):
    """
    Set the default options of the optimizer.
    """
    rhoend = options.get("rhoend", 1e-6)
    options.setdefault("rhobeg", max(1.0, rhoend))
    if nlp.n > 0:
        options["rhobeg"] = min(0.5 * np.min(nlp.xu - nlp.xl), options["rhobeg"])
    options.setdefault("rhoend", min(rhoend, options["rhobeg"]))
    options.setdefault("npt", 2 * nlp.n + 1)
    options.setdefault("maxfev", max(500 * nlp.n, options["npt"] + 1))
    options.setdefault("maxiter", 1000 * nlp.n)
    options.setdefault("target", -np.inf)
    options.setdefault("ftol_abs", -1.0)
    options.setdefault("ftol_rel", -1.0)
    options.setdefault("xtol_abs", -1.0)
    options.setdefault("xtol_rel", -1.0)
    options.setdefault("debug", False)


def _set_default_constants(kwargs):
    """
    Set the default constants of the optimizer.
    """
    kwargs.setdefault("store_hist", False)
    kwargs.setdefault("eta1", 0.1)
    kwargs.setdefault("eta2", 0.7)
    kwargs.setdefault("eta3", 1.6e1)
    kwargs.setdefault("eta4", 2.5e2)
    kwargs.setdefault("eta5", 1e-2)
    kwargs.setdefault("theta1", 0.5)
    kwargs.setdefault("theta2", 1.4)
    kwargs.setdefault("theta3", np.sqrt(2.0))
    kwargs.setdefault("theta4", 2.0)
    kwargs.setdefault("theta5", 0.1)
    kwargs.setdefault("upsilon1", 1.5)
    kwargs.setdefault("upsilon2", 2.0)
    kwargs.setdefault("zeta", 0.8)
