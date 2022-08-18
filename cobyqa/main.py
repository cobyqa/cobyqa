import numpy as np

from .optimize import TrustRegion
from .utils import RestartRequiredException, absmax_arrays


def minimize(fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None, Aeq=None,
             beq=None, cub=None, ceq=None, options=None, **kwargs):
    r"""
    Minimize a real-valued function.

    The minimization can be subject to bound, linear inequality, linear
    equality, nonlinear inequality, and nonlinear equality constraints using a
    derivative-free trust-region SQP method. Although the solver may encounter
    infeasible points (including the initial guess), the bounds constraints (if
    any) are always respected.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
        parameters to forward to the objective function.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Parameters to forward to the objective, the nonlinear inequality
        constraint, and the nonlinear equality constraint functions.
    xl : array_like, shape (n,), optional
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,), optional
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    Aub : array_like, shape (mlub, n), optional
        Jacobian matrix of the linear inequality constraints. Each row of `Aub`
        stores the gradient of a linear inequality constraint.
    bub : array_like, shape (mlub,), optional
        Right-hand side vector of the linear inequality constraints
        ``Aub @ x <= bub``, where ``x`` has the same size than `x0`.
    Aeq : array_like, shape (mleq, n), optional
        Jacobian matrix of the linear equality constraints. Each row of `Aeq`
        stores the gradient of a linear equality constraint.
    beq : array_like, shape (mleq,), optional
        Right-hand side vector of the linear equality constraints
        `Aeq @ x = beq`, where ``x`` has the same size than `x0`.
    cub : callable
        Nonlinear inequality constraint function ``ceq(x, *args) <= 0``.

            ``cub(x, *args) -> array_like, shape (mnlub,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
        parameters to forward to the constraint function.
    ceq : callable
        Nonlinear equality constraint function ``ceq(x, *args) = 0``.

            ``ceq(x, *args) -> array_like, shape (mnleq,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
        parameters to forward to the constraint function.
    options : dict, optional
        Options to forward to the solver. Accepted options are:

            rhobeg : float, optional
                Initial trust-region radius (the default is 1).
            rhoend : float, optional
                Final trust-region radius (the default is 1e-6).
            npt : int, optional
                Number of interpolation points for the objective and constraint
                models (the default is ``2 * n + 1``).
            maxfev : int, optional
                Upper bound on the number of objective and constraint function
                evaluations (the default is ``500 * n``).
            maxiter: int, optional
                Upper bound on the number of main loop iterations (the default
                is ``1000 * n``).
            target : float, optional
                Target value on the objective function (the default is
                ``-numpy.inf``). If the solver encounters a feasible point at
                which the objective function evaluations is below the target
                value, then the computations are stopped.
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
                solver (the default is False).
            respect_bounds : bool, optional
                Whether to respect the bounds through the iterations (the
                default is True).
            debug : bool, optional
                Whether to make debugging tests during the execution, which is
                not recommended in production (the default is False).

    Returns
    -------
    OptimizeResult
        Result of the optimization solver. Important attributes are: ``x`` the
        solution point, ``success`` a flag indicating whether the optimization
        terminated successfully, and ``message`` a description of the
        termination status of the optimization. See `OptimizeResult` for a
        description of other attributes.

    Other Parameters
    ----------------
    alternative_models_radius_threshold : float, optional
        Threshold on the trust-region ratio used to decide whether alternative
        models should be used (the default is 1e-2).
    constraint_activation_factor : float, optional
        Factor on the linear models of the constraints used by the trust-region
        subproblem solvers to decide whether a constraint should be considered
        active (the default is 0.2).
    exact_normal_step : bool, optional
        Whether the normal subproblem should be solved exactly using SciPy (the
        default is False).
    improve_tcg : bool, optional
        Whether to improve the truncated conjugate gradient step round the
        trust-region boundary (the default is True). It is currently only
        implemented for bound-constrained problems.
    large_radius_bound_detection_factor : float, optional
        Factor on the trust-region radius lower bound used to decide whether the
        current trust-region radius is large (the default is 250).
    large_radius_reduction_factor : float, optional
        Factor on the trust-region radius lower bound used to reduce large
        trust-region radii (the default is 0.1)
    large_ratio_threshold: float, optional
        Threshold on the trust-region ratio used to decide whether an iteration
        performed well (the default is 0.7).
    low_ratio_threshold: float, optional
        Threshold on the trust-region ratio used to decide whether an iteration
        performed poorly (the default is 0.1).
    normal_step_shrinkage_factor : float, optional
        Shrinkage factor on the trust-region radius for the normal subproblems
        (the default is 0.8).
    penalty_detection_factor : float, optional
        Factor on the penalty coefficient used to decide whether it should be
        increased (the default is 1.5).
    penalty_growth_factor : float, optional
        Increasing factor on the penalty coefficient (the default is 2).
    radius_growth_factor : float, optional
        Increasing factor on the trust-region radius (the default is sqrt(2)).
    radius_reduction_factor : float, optional
        Reduction factor on the trust-region radius (the default is 0.5).
    short_radius_bound_detection_factor : float, optional
        Factor on the trust-region radius lower bound used to decide whether the
        current trust-region radius is short (the default is 16).
    short_radius_detection_factor : float, optional
        Factor on the trust-region radius lower bound used to reduce short
        trust-region radii (the default is 1.4)
    short_step_detection_factor : float, optional
        Factor on the norm of the current trust-region step to decide whether is
        small (the default is 0.5).
    store_history : bool, optional
        Whether the history of the different evaluations should be stored (the
        default is False).

    References
    ----------
    .. [1] J. Nocedal and S. J. Wright. Numerical Optimization. Second. Springer
       Ser. Oper. Res. Financ. Eng. New York, NY, US: Springer, 2006.
    .. [2] M. J. D. Powell. "A direct search optimization method that models the
       objective and constraint functions by linear interpolation." In: Advances
       in Optimization and Numerical Analysis. Ed. by S. Gomez and J. P.
       Hennart. Dordrecht, NL: Springer, 1994, pp. 51--67.

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

        \begin{array}{ll}
            \min        & \quad q(x) = (x_1 - 1)^2 + (x_2 - 2.5)^2\\
            \text{s.t.} & \quad -x_1 + 2x_2 \le 2,\\
                        & \quad x_1 + 2x_2 \le 6,\\
                        & \quad x_1 - 2x_2 \le 2,\\
                        & \quad x_1 \ge 0,\\
                        & \quad x_2 \ge 0,\\
                        & \quad x \in \R^2.
        \end{array}

    Its objective function can be implemented as:

    >>> def quadratic(x):
    ...     return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0

    This problem can be solved using `minimize` as:

    >>> x0 = [2.0, 0.0]
    >>> xl = [0.0, 0.0]
    >>> Aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    >>> bub = [2.0, 6.0, 2.0]
    >>> res = minimize(quadratic, x0, xl=xl, Aub=Aub, bub=bub)
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

        \begin{array}{ll}
            \min        & \quad f(x) = x_3\\
            \text{s.t.} & \quad -5x_1 + x_2 - x_3 \le 0,\\
                        & \quad 5x_1 + x_2 - x_3 \le 0,\\
                        & \quad x_1^2 + x_2^2 + 4x_2 - x_3 \le 0,\\
                        & \quad x \in \R^3.
        \end{array}

    Its only nonlinear constraints can be implemented in Python as:

    >>> def cub(x):
    ...     return x[0] ** 2.0 + x[1] ** 2.0 + 4.0 * x[1] - x[2]

    This problem can be solved using `minimize` as:

    >>> x0 = [1.0, 1.0, 1.0]
    >>> Aub = [[-5.0, 1.0, -1.0], [5.0, 1.0, -1.0]]
    >>> bub = [0.0, 0.0]
    >>> res = minimize(lambda x: x[2], x0, Aub=Aub, bub=bub, cub=cub)
    >>> res.x
    array([ 0., -3., -3.])
    """
    # Build the initial models of the optimization problem. The computations
    # must be stopped immediately if all indices are fixed by the bound
    # constraints or if the target function value has been reached by an initial
    # interpolation point (in which case the initial models are not built).
    if not isinstance(args, tuple):
        args = (args,)
    _set_default_constants(kwargs)
    struct = TrustRegion(fun, x0, xl, xu, Aub, bub, Aeq, beq, cub, ceq, options,
                         *args, **kwargs)
    if np.all(struct.ifix):
        exit_status = 9
        nfev = 1
    elif struct.target_reached:
        exit_status = 1
        nfev = struct.kopt + 1
    else:
        exit_status = 0
        nfev = struct.npt

    # Begin the iterative procedure. The purpose of n_alt_models is to decide
    # whether alternatives models should be used. The purpose of n_short_half
    # and n_short_tenth is to prevent the algorithm from reducing the lower
    # bound on the trust-region radius too fast.
    rho = struct.rhobeg
    delta = rho
    fsav = struct.fopt
    xsav = struct.xbase + struct.xopt
    nit = 0
    n_alt_models = 0
    n_short_half = 0
    n_short_tenth = 0
    while exit_status == 0:
        # Update the shift of the origin to manage computer rounding errors. The
        # computations are stopped beforehand if the maximum number of
        # iterations has been reached.
        if nit >= struct.maxiter:
            exit_status = 7
            break
        nit += 1
        struct.shift_origin(delta)

        # Evaluate the trial step.
        delsav = delta
        fopt = struct.fopt
        kopt = struct.kopt
        coptub = np.copy(struct.coptub)
        copteq = np.copy(struct.copteq)
        xopt = np.copy(struct.xopt)
        test = kwargs.get('short_step_detection_factor')
        is_trust_region_step = not struct.is_model_step
        reduce_rho = False
        evaluate_fun = True
        if struct.type in 'LO':
            test /= np.sqrt(2.0)
        if is_trust_region_step:
            nstep, tstep = struct.trust_region_step(delta, **kwargs)
            step = nstep + tstep
            snorm = np.linalg.norm(step)
            iact = struct.active_set(struct.xopt)
            inew = struct.active_set(struct.xopt + step)
            if not np.array_equal(iact, inew):
                test = kwargs.get('constraint_activation_factor') - 1e-4
                if struct.type in 'LO':
                    test /= np.sqrt(2.0)
            evaluate_fun = snorm > test * delta
            if not evaluate_fun:
                delta *= kwargs.get('radius_reduction_factor')
                if delta <= kwargs.get('short_radius_detection_factor') * rho:
                    delta = rho
                if delsav > rho:
                    n_short_half, n_short_tenth = 0, 0
                else:
                    n_short_half += 1
                    n_short_tenth += 1
                    if snorm >= 0.5 * rho:
                        n_short_half = 0
                    if snorm >= 0.1 * rho:
                        n_short_tenth = 0
                reduce_rho = n_short_half >= 5 or n_short_tenth >= 3
                reduce_rho = reduce_rho and delsav <= rho
                if not reduce_rho:
                    struct.prepare_model_step(max(delta, 2.0 * rho))
                else:
                    n_short_half, n_short_tenth = 0, 0
            else:
                n_short_half, n_short_tenth = 0, 0
        else:
            deltx = kwargs.get('large_radius_reduction_factor') * delta
            deltx = max(deltx, rho)
            nstep = np.zeros_like(struct.xopt)
            tstep = struct.model_step(deltx, **kwargs)
            step = tstep
            snorm = np.linalg.norm(step)

        if evaluate_fun:
            # Evaluate the objective function, include the trial point in the
            # interpolation set, and update accordingly the models. The
            # computations are stopped beforehand if the maximum number of
            # function evaluations has been reached.
            if nfev >= struct.maxfev:
                exit_status = 6
                break
            nfev += 1
            try:
                fx, mopt, ratio = struct.update(nstep, tstep, delta, **kwargs)
            except RestartRequiredException:
                continue
            except ZeroDivisionError:
                exit_status = 8
                break
            if struct.target_reached:
                exit_status = 1
                break
            if abs(fx - fsav) <= struct.ftol_abs:
                exit_status = 2
                break
            if abs(fx - fsav) <= struct.ftol_rel * max(abs(fsav), 1.0):
                exit_status = 3
                break
            xfull = struct.xbase + xopt + step
            xdiff = np.linalg.norm(xfull - xsav)
            if xdiff <= struct.xtol_abs:
                exit_status = 4
                break
            if xdiff <= struct.xtol_rel * max(np.linalg.norm(xsav), 1.0):
                exit_status = 5
                break
            fsav = fx
            xsav = xfull

            # Update the trust-region radius.
            low_ratio = kwargs.get('low_ratio_threshold')
            if is_trust_region_step:
                gamma = kwargs.get('radius_reduction_factor')
                if ratio <= low_ratio:
                    delta *= gamma
                elif ratio <= kwargs.get('large_ratio_threshold'):
                    delta = max(gamma * delta, snorm)
                else:
                    delbd = kwargs.get('radius_growth_factor') * delta
                    delta = max(gamma * delta, snorm / gamma)
                    delta = min(delta, delbd)
                if delta <= kwargs.get('short_radius_detection_factor') * rho:
                    delta = rho

            # Attempt to replace the models by the alternative ones.
            if is_trust_region_step and delta <= rho:
                if ratio > kwargs.get('alternative_models_radius_threshold'):
                    n_alt_models = 0
                else:
                    n_alt_models += 1
                    gd = struct.model_obj_grad(struct.xopt)
                    gd_alt = struct.model_obj_alt_grad(struct.xopt)
                    if np.linalg.norm(gd) < 10.0 * np.linalg.norm(gd_alt):
                        n_alt_models = 0
                    if n_alt_models >= 3:
                        struct.reset_models()
                        n_alt_models = 0

            # If a trust-region step has provided a sufficient decrease or if a
            # model-improvement step has just been computed, then the next
            # iteration is a trust-region step. If an interpolation point is
            # substantially far from the trust-region center, a
            # model-improvement step is entertained.
            reduce_rho = is_trust_region_step and ratio < low_ratio
            if not reduce_rho:
                struct.prepare_trust_region_step()
            else:
                struct.prepare_model_step(max(delta, 2.0 * rho))
                reduce_rho = not struct.is_model_step and delsav <= rho
                if reduce_rho:
                    ropt = struct.rval[struct.kopt]
                    msav = struct(xopt, fopt, coptub, copteq)
                    rsav = struct.rval[kopt]
                    reduce_rho = not struct.less_merit(mopt, ropt, msav, rsav)

        # Update the lower bound on the trust-region radius.
        if reduce_rho:
            if rho <= struct.rhoend:
                break
            large_radius = kwargs.get('large_radius_bound_detection_factor')
            short_radius = kwargs.get('short_radius_bound_detection_factor')
            delta = kwargs.get('radius_reduction_factor') * rho
            if rho > large_radius * struct.rhoend:
                rho *= kwargs.get('large_radius_reduction_factor')
            elif rho <= short_radius * struct.rhoend:
                rho = struct.rhoend
            else:
                rho = np.sqrt(rho * struct.rhoend)
            delta = max(delta, rho)
            struct.prepare_trust_region_step()
            struct.reduce_penalty()
            if struct.disp:
                x_full = struct.get_x(struct.xbase + struct.xopt)
                maxcv = struct.maxcv if struct.type not in 'UB' else None
                message = f'New trust-region radius: {rho}.'
                _print(fun.__name__, x_full, struct.fopt, maxcv, nfev, message)

    # Get the success flag.
    if exit_status == 9:
        bdtol = 10.0 * np.finfo(float).eps * struct.xopt.size
        bdtol *= absmax_arrays(xl, xu, initial=1.0)
        success = struct.type == 'U' or struct.maxcv <= bdtol
    else:
        success = exit_status in [0, 1, 2, 3, 4, 5]

    # Build the result structure and return.
    try:
        penalty = struct.penalty
    except AttributeError:
        penalty = 0.0
    result = struct.build_result(penalty, **kwargs)
    result.nfev = nfev
    result.nit = nit
    result.status = exit_status
    result.success = success
    result.message = {
        0: 'Lower bound for the trust-region radius has been reached.',
        1: 'Target function value has been achieved.',
        2: 'Absolute tolerance on the objective function has been reached.',
        3: 'Relative tolerance on the objective function has been reached.',
        4: 'Absolute tolerance on the decision variables has been reached.',
        5: 'Relative tolerance on the decision variables has been reached.',
        6: 'Maximum number of function evaluations has been exceeded.',
        7: 'Maximum number of iterations has been exceeded.',
        8: 'Denominator of the updating formula is zero.',
        9: 'All variables are fixed by the constraints.',
        -1: 'Bound constraints are infeasible.',
    }.get(exit_status, 'Unknown exit status.')
    if struct.disp:
        _print(fun.__name__, result.x, result.fun, result.get('maxcv'),
               result.nfev, result.message)
    return result


def _set_default_constants(kwargs):
    kwargs.setdefault('alternative_models_radius_threshold', 1e-2)
    kwargs.setdefault('constraint_activation_factor', 0.2)
    kwargs.setdefault('exact_normal_step', False)
    kwargs.setdefault('improve_tcg', True)
    kwargs.setdefault('large_radius_bound_detection_factor', 250.0)
    kwargs.setdefault('large_radius_reduction_factor', 0.1)
    kwargs.setdefault('large_ratio_threshold', 0.7)
    kwargs.setdefault('low_ratio_threshold', 0.1)
    kwargs.setdefault('normal_step_shrinkage_factor', 0.8)
    kwargs.setdefault('penalty_detection_factor', 1.5)
    kwargs.setdefault('penalty_growth_factor', 2.0)
    kwargs.setdefault('radius_growth_factor', np.sqrt(2.0))
    kwargs.setdefault('radius_reduction_factor', 0.5)
    kwargs.setdefault('short_radius_bound_detection_factor', 16.0)
    kwargs.setdefault('short_radius_detection_factor', 1.4)
    kwargs.setdefault('short_step_detection_factor', 0.5)
    kwargs.setdefault('store_history', False)


def _print(fun, x, fopt, maxcv, nf, message):
    print()
    print(message)
    print(f'Number of function evaluations: {nf}.')
    print(f'Least value of {fun}: {fopt}.')
    if maxcv is not None:
        print(f'Maximum constraint violation: {maxcv}.')
    print(f'Corresponding point: {x}.')
    print()
