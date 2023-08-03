import numpy as np
from scipy.optimize import OptimizeResult

from .framework import TrustRegion
from .problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem
from .utils import MaxEvalError, get_arrays_tol

# Default options.
DEFAULT_OPTIONS = {
    'debug': False,
    'max_eval_f': lambda n: 500 * n,
    'max_iter_f': lambda n: 1000 * n,
    'npt_f': lambda n: 2 * n + 1,
    'radius_init': 1.0,
    'radius_final': 1e-6,
    'store_hist': False,
    'target': -np.inf,
    'verbose': False,
}

# Exit status.
EXIT_RADIUS_SUCCESS = 0
EXIT_TARGET_SUCCESS = 1
EXIT_FIXED_SUCCESS = 2
EXIT_MAX_EVAL_WARNING = 3
EXIT_MAX_ITER_WARNING = 4
EXIT_INFEASIBLE_ERROR = -1


def minimize(fun, x0, args=(), xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, options=None):
    r"""
    Minimize a scalar function using the COBYQA method.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective and constraints function.
    xl : array_like, shape (n,), optional
        Lower bounds on the variables ``xl <= x``.
    xu : array_like, shape (n,), optional
        Upper bounds on the variables ``x <= xu``.
    aub : array_like, shape (m_linear_ub, n), optional
        Left-hand side matrix of the linear inequality constraints
        ``aub @ x <= bub``.
    bub : array_like, shape (m_linear_ub,), optional
        Right-hand side vector of the linear inequality constraints
        ``aub @ x <= bub``.
    aeq : array_like, shape (m_linear_eq, n), optional
        Left-hand side matrix of the linear equality constraints
        ``aeq @ x == beq``.
    beq : array_like, shape (m_linear_eq,), optional
        Right-hand side vector of the linear equality constraints
        ``aeq @ x == beq``.
    cub : callable, optional
        Nonlinear inequality constraints function ``cub(x, *args) <= 0``.

            ``cub(x, *args) -> array_like, shape (m_nonlinear_ub,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    ceq : callable, optional
        Nonlinear equality constraints function ``ceq(x, *args) == 0``.

            ``ceq(x, *args) -> array_like, shape (m_nonlinear_eq,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    options : dict, optional
        Options passed to the solver. Accepted keys are:

            verbose : bool, optional
                Whether to print information about the optimization procedure.
            max_eval : int, optional
                Maximum number of function evaluations.
            max_iter : int, optional
                Maximum number of iterations.
            target : float, optional
                Target on the objective function value. The optimization
                procedure is terminated when the objective function value of a
                nearly feasible point is less than or equal to this target.
            store_hist : bool, optional
                Whether to store the history of the function evaluations.
            radius_init : float, optional
                Initial trust-region radius.
            radius_final : float, optional
                Final trust-region radius.
            npt : int, optional
                Number of interpolation points.
            debug : bool, optional
                Whether to perform additional checks. This option should be
                used only for debugging purposes and is highly discouraged.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result of the optimization procedure, which has the following fields:

            message : str
                Description of the cause of the termination.
            success : bool
                Whether the optimization procedure terminated successfully.
            status : int
                Termination status of the optimization procedure.
            x : ndarray, shape (n,)
                Solution point.
            fun : float
                Objective function value at the solution point.
            maxcv : float
                Maximum constraint violation at the solution point.
            nit : int
                Number of iterations.
            nfev : int
                Number of function evaluations.

    References
    ----------
    .. [1] J. Nocedal and S. J. Wright. *Numerical Optimization*. Springer
       Series in Operations Research and Financial Engineering. Springer, New
       York, NY, USA, second edition, 2006.
    .. [2] M. J. D. Powell. A direct search optimization method that models the
       objective and constraint functions by linear interpolation. In S. Gomez
       and J. P. Hennart, editors, *Advances in Optimization and Numerical
       Analysis*, volume 275 of *Mathematics and Its Applications*, pages 51–67.
       Springer, Dordrecht, The Netherlands, 1994.

    Examples
    --------
    We first minimize the Rosenbrock function implemented in `scipy.optimize`.

    .. testsetup::

        import numpy as np
        np.set_printoptions(precision=3, suppress=True)

    >>> from scipy.optimize import rosen
    >>> from cobyqa import minimize

    To solve the problem using COBYQA, run:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0)
    >>> res.x
    array([1., 1., 1., 1., 1.])

    To see how bound and linear constraints are handled using `minimize`, we
    solve Example 16.4 of [1]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \mathbb{R}^2}   & \quad (x_1 - 1)^2 + (x_2 - 2.5)^2\\
            \text{s.t.}                 & \quad -x_1 + 2x_2 \le 2,\\
                                        & \quad x_1 + 2x_2 \le 6,\\
                                        & \quad x_1 - 2x_2 \le 2,\\
                                        & \quad x_1 \ge 0,\\
                                        & \quad x_2 \ge 0.
        \end{aligned}

    Its objective function can be implemented as:

    >>> def fun(x):
    ...     return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0

    This problem can be solved using `minimize` as:

    >>> x0 = [2.0, 0.0]
    >>> xl = [0.0, 0.0]
    >>> aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    >>> bub = [2.0, 6.0, 2.0]
    >>> res = minimize(fun, x0, xl=xl, aub=aub, bub=bub)
    >>> res.x
    array([1.4, 1.7])

    Finally, to see how nonlinear constraints are handled, we solve Problem (F)
    of [2]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \mathbb{R}^2}   & \quad -x_1 - x_2\\
            \text{s.t.}                 & \quad x_1^2 - x_2 \le 0,\\
                                        & \quad x_1^2 + x_2^2 \le 1.
        \end{aligned}

    Its objective and constraint functions can be implemented as:

    >>> def fun(x):
    ...     return -x[0] - x[1]
    >>>
    >>> def cub(x):
    ...     return [x[0] ** 2.0 - x[1], x[0] ** 2.0 + x[1] ** 2.0 - 1.0]

    This problem can be solved using `minimize` as:

    >>> x0 = [1.0, 1.0]
    >>> res = minimize(fun, x0, cub=cub)
    >>> res.x
    array([0.707, 0.707])
    """
    # Get basic options that are needed for the initialization.
    if options is None:
        options = {}
    verbose = options.get('verbose', DEFAULT_OPTIONS['verbose'])
    store_hist = options.get('store_hist', DEFAULT_OPTIONS['store_hist'])

    # Initialize the objective function.
    obj = ObjectiveFunction(fun, verbose, store_hist, *args)

    # Initialize the bound constraints.
    n_orig = len(x0)
    if xl is None:
        xl = np.full(n_orig, -np.inf, dtype=float)
    if xu is None:
        xu = np.full(n_orig, np.inf, dtype=float)
    bounds = BoundConstraints(xl, xu)

    # Initialize the linear constraints.
    if aub is None:
        aub = np.empty((0, n_orig))
    if bub is None:
        bub = np.empty(0)
    linear_ub = LinearConstraints(aub, bub, False)
    if aeq is None:
        aeq = np.empty((0, n_orig))
    if beq is None:
        beq = np.empty(0)
    linear_eq = LinearConstraints(aeq, beq, True)

    # Initialize the nonlinear constraints.
    nonlinear_ub = NonlinearConstraints(cub, False, verbose, store_hist, *args)
    nonlinear_eq = NonlinearConstraints(ceq, True, verbose, store_hist, *args)

    # Initialize the problem (remove the fixed variables).
    pb = Problem(obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq)

    # Set the default options.
    _set_default_options(options, pb.n)

    # Initialize the models and skip the computations whenever possible.
    if not pb.bounds.is_feasible:
        # The bound constraints are infeasible.
        return _build_result(pb, pb.x0, False, EXIT_INFEASIBLE_ERROR, 0, verbose)
    elif pb.n == 0:
        # All variables are fixed by the bound constraints.
        return _build_result(pb, pb.x0, True, EXIT_FIXED_SUCCESS, 0, verbose)
    if verbose:
        print('Starting the optimization procedure.')
        print(f'Initial trust-region radius: {options["radius_init"]}.')
        print(f'Final trust-region radius: {options["radius_final"]}.')
        print(f'Maximum number of function evaluations: {options["max_eval"]}.')
        print(f'Maximum number of iterations: {options["max_iter"]}.')
        print()
    framework = TrustRegion(pb, options)
    if framework.models.target_init:
        # The target on the objective function value has been reached
        return _build_result(pb, framework.x_best, True, EXIT_TARGET_SUCCESS, 0, verbose, framework.fun_best, framework.cub_best, framework.ceq_best)
    elif pb.n_eval >= options['max_eval']:
        # The maximum number of function evaluations has been exceeded.
        return _build_result(pb, framework.x_best, False, EXIT_MAX_ITER_WARNING, 0, verbose, framework.fun_best, framework.cub_best, framework.ceq_best)

    # Start the optimization procedure.
    k_new = None
    success = False
    n_iter = 0
    n_short_steps = 0
    n_very_short_steps = 0
    n_alt_models = 0
    while True:
        if n_iter >= options['max_iter']:
            status = EXIT_MAX_ITER_WARNING
            break
        n_iter += 1

        # Update the point around which the quadratic models are built.
        if np.linalg.norm(framework.x_best - framework.models.interpolation.x_base) >= 0.0 * framework.radius:
            framework.shift_x_base(options)

        # Evaluate the trial step.
        radius_save = framework.radius
        normal_step, tangential_step = framework.get_trust_region_step(options)
        step = normal_step + tangential_step
        s_norm = np.linalg.norm(step)

        reduce_resolution = False
        improve_geometry = False
        eval_functions = s_norm > 0.5 * framework.radius

        if not eval_functions:
            framework.radius *= 0.5
            if radius_save > framework.resolution:
                n_short_steps = 0
                n_very_short_steps = 0
            else:
                n_short_steps += 1
                if s_norm >= 0.5 * framework.resolution:
                    n_short_steps = 0
                n_very_short_steps += 1
                if s_norm >= 0.1 * framework.resolution:
                    n_very_short_steps = 0
            reduce_resolution = radius_save <= framework.resolution and (n_short_steps >= 5 or n_very_short_steps >= 3)
            if reduce_resolution:
                n_short_steps = 0
                n_very_short_steps = 0
            else:
                k_new, dist_new = framework.get_index_to_remove()
                improve_geometry = dist_new > max(framework.radius, 2.0 * framework.resolution)
        else:
            # Increase the penalty parameter if necessary.
            same_best_point = framework.increase_penalty(step)
            if same_best_point:
                # Evaluate the objective and constraint functions.
                try:
                    fun_val, cub_val, ceq_val, target = _eval(pb, framework, step, options)
                except MaxEvalError:
                    status = EXIT_MAX_EVAL_WARNING
                    break
                if target:
                    return _build_result(pb, framework.x_best + step, True, EXIT_TARGET_SUCCESS, n_iter, verbose, fun_val, cub_val, ceq_val)

                # Perform a second-order correction step if necessary.
                merit_old = framework.merit(framework.x_best, framework.fun_best, framework.cub_best, framework.ceq_best)
                merit_new = framework.merit(framework.x_best + step, fun_val, cub_val, ceq_val)
                if pb.type == 'nonlinearly constrained' and merit_new > merit_old and np.linalg.norm(normal_step) > 0.8 ** 2.0 * framework.radius:
                    soc_step = framework.get_second_order_correction_step(step, options)
                    if np.linalg.norm(soc_step) > 0.0:
                        step += soc_step

                        # Evaluate the objective and constraint functions.
                        try:
                            fun_val, cub_val, ceq_val, target = _eval(pb, framework, step, options)
                        except MaxEvalError:
                            status = EXIT_MAX_EVAL_WARNING
                            break
                        if target:
                            return _build_result(pb, framework.x_best + step, True, EXIT_TARGET_SUCCESS, n_iter, verbose, fun_val, cub_val, ceq_val)

                # Calculate the reduction ratio.
                ratio = framework.get_reduction_ratio(step, fun_val, cub_val, ceq_val)

                # Update the Lagrange multipliers.
                framework.set_multipliers()

                # Choose an interpolation point to remove.
                k_new = framework.get_index_to_remove(framework.x_best + step)[0]

                # Update the interpolation set.
                framework.models.update_interpolation(k_new, framework.x_best + step, fun_val, cub_val, ceq_val)
                framework.set_best_index()

                # Update the trust-region radius.
                framework.update_radius(step, ratio)

                # Attempt to replace the models by the alternative ones.
                if framework.radius <= framework.resolution:
                    if ratio >= 0.01:
                        n_alt_models = 0
                    else:
                        n_alt_models += 1
                        grad = framework.models.fun_grad(framework.x_best)
                        grad_alt = framework.models.fun_alt_grad(framework.x_best)
                        if np.linalg.norm(grad) < 10.0 * np.linalg.norm(grad_alt):
                            n_alt_models = 0
                        if n_alt_models >= 3:
                            framework.models.reset_models()
                            n_alt_models = 0

                # Update reduce_resolution and improve_geometry.
                k_new, dist_new = framework.get_index_to_remove()
                reduce_resolution = radius_save <= framework.resolution and ratio <= 0.1 and dist_new <= max(framework.radius, 2.0 * framework.resolution)
                improve_geometry = ratio <= 0.1 and dist_new > max(framework.radius, 2.0 * framework.resolution)

        if reduce_resolution:
            if framework.resolution <= options['radius_final']:
                success = True
                status = EXIT_RADIUS_SUCCESS
                break
            framework.reduce_resolution(options)
            framework.decrease_penalty()

            if verbose:
                resid = pb.resid(framework.x_best, framework.cub_best, framework.ceq_best)
                _print_step(f'New trust-region radius: {framework.resolution}', pb, pb.build_x(framework.x_best), framework.fun_best, resid, pb.n_eval, n_iter)
                print()

        if improve_geometry:
            step = framework.get_geometry_step(k_new, options)

            # Evaluate the objective and constraint functions.
            try:
                fun_val, cub_val, ceq_val, target = _eval(pb, framework, step, options)
            except MaxEvalError:
                status = EXIT_MAX_EVAL_WARNING
                break
            if target:
                return _build_result(pb, framework.x_best + step, True, EXIT_TARGET_SUCCESS, n_iter, verbose, fun_val, cub_val, ceq_val)

            # Update the interpolation set.
            framework.models.update_interpolation(k_new, framework.x_best + step, fun_val, cub_val, ceq_val)
            framework.set_best_index()

    return _build_result(pb, framework.x_best, success, status, n_iter, verbose, framework.fun_best, framework.cub_best, framework.ceq_best)


def _set_default_options(options, n):
    """
    Set the default options.
    """
    if 'radius_init' in options and options['radius_init'] <= 0.0:
        raise ValueError('The initial trust-region radius must be positive.')
    if 'radius_final' in options and options['radius_final'] < 0.0:
        raise ValueError('The final trust-region radius must be nonnegative.')
    if 'radius_init' in options and 'radius_final' in options:
        if options['radius_init'] < options['radius_final']:
            raise ValueError('The initial trust-region radius must be greater than or equal to the final trust-region radius.')
    elif 'radius_init' in options:
        options['radius_final'] = min(DEFAULT_OPTIONS['radius_final'], options['radius_init'])
    elif 'radius_final' in options:
        options['radius_init'] = max(DEFAULT_OPTIONS['radius_init'], options['radius_final'])
    else:
        options['radius_init'] = DEFAULT_OPTIONS['radius_init']
        options['radius_final'] = DEFAULT_OPTIONS['radius_final']
    options['radius_init'] = float(options['radius_init'])
    options['radius_final'] = float(options['radius_final'])
    if 'npt' in options and options['npt'] > (n + 1) * (n + 2) // 2:
        raise ValueError(f'The number of interpolation points must be at most {(n + 1) * (n + 2) // 2}.')
    options.setdefault('npt', DEFAULT_OPTIONS['npt_f'](n))
    options['npt'] = int(options['npt'])
    if 'max_eval' in options and options['max_eval'] <= 0:
        raise ValueError('The maximum number of function evaluations must be positive.')
    options.setdefault('max_eval', max(DEFAULT_OPTIONS['max_eval_f'](n), options['npt'] + 1))
    options['max_eval'] = int(options['max_eval'])
    if 'max_iter' in options and options['max_iter'] <= 0:
        raise ValueError('The maximum number of iterations must be positive.')
    options.setdefault('max_iter', DEFAULT_OPTIONS['max_iter_f'](n))
    options['max_iter'] = int(options['max_iter'])
    options.setdefault('target', DEFAULT_OPTIONS['target'])
    options['target'] = float(options['target'])
    options.setdefault('verbose', DEFAULT_OPTIONS['verbose'])
    options['verbose'] = bool(options['verbose'])
    options.setdefault('store_hist', DEFAULT_OPTIONS['store_hist'])
    options['store_hist'] = bool(options['store_hist'])
    options.setdefault('debug', DEFAULT_OPTIONS['debug'])
    options['debug'] = bool(options['debug'])


def _eval(pb, framework, step, options):
    """
    Evaluate the objective and constraint functions.
    """
    if pb.n_eval >= options['max_eval']:
        raise MaxEvalError
    x_eval = framework.x_best + step
    fun_val = pb.fun(x_eval)
    cub_val = pb.cub(x_eval)
    ceq_val = pb.ceq(x_eval)
    r_val = pb.resid(x_eval, cub_val, ceq_val)
    tol_bounds = get_arrays_tol(pb.bounds.xl, pb.bounds.xu)
    return fun_val, cub_val, ceq_val, fun_val <= options['target'] and r_val < tol_bounds


def _build_result(pb, x, success, status, n_iter, verbose, fun_val=None, cub_val=None, ceq_val=None):
    """
    Build the result of the optimization process.
    """
    # Build the result.
    result = OptimizeResult()
    result.x = pb.build_x(x)
    result.fun = fun_val if fun_val is not None else pb.fun(x)
    result.maxcv = pb.resid(x, cub_val, ceq_val)
    result.nfev = pb.n_eval
    result.nit = n_iter
    result.success = success and (result.maxcv < 10.0 * np.finfo(float).eps * np.max(np.abs(result.x), initial=1.0))
    result.status = status
    result.message = {
        EXIT_RADIUS_SUCCESS: 'The lower bound for the trust-region radius has been reached',
        EXIT_TARGET_SUCCESS: 'The target objective function value has been reached',
        EXIT_FIXED_SUCCESS: 'All variables are fixed by the bound constraints',
        EXIT_MAX_EVAL_WARNING: 'The maximum number of function evaluations has been exceeded',
        EXIT_MAX_ITER_WARNING: 'The maximum number of iterations has been exceeded',
        EXIT_INFEASIBLE_ERROR: 'The bound constraints are infeasible',
    }.get(status, 'Unknown exit status')

    # Print the result if requested.
    if verbose:
        _print_step(result.message, pb, result.x, result.fun, result.maxcv, result.nfev, result.nit)
    return result


def _print_step(message, pb, x, fun_val, r_val, n_eval, n_iter):
    """
    Print information about the current state of the optimization process.
    """
    print()
    print(f'{message}.')
    print(f'Number of function evaluations: {n_eval}.')
    print(f'Number of iterations: {n_iter}.')
    print(f'Least value of {pb.fun_name}: {fun_val}.')
    print(f'Maximum constraint violation: {r_val}.')
    print(f'Corresponding point: {x}.')
