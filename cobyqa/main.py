import warnings

import numpy as np
from scipy.optimize import Bounds, OptimizeResult

from .framework import TrustRegion
from .problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem
from .utils import MaxEvalError
from .settings import ExitStatus, Options, DEFAULT_OPTIONS, PRINT_OPTIONS


def minimize(fun, x0, args=(), bounds=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, callback=None, options=None):
    r"""
    Minimize a scalar function using the COBYQA method.

    The COBYQA method is a derivative-free optimization method designed to solve
    general nonlinear optimization problems. A complete description of the
    method is given in [3]_.

    Parameters
    ----------
    fun : {callable, None}
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective and constraints function.
    bounds : {`scipy.optimize.Bounds`, array_like, shape (n, 2)}, optional
        Bound constraints of the problem. It can be one of the cases below.

        #. An instance of `scipy.optimize.Bounds`.
        #. An array with shape (n, 2). The bound constraints for ``x[i]`` are
           ``bounds[i][0] <= x[i] <= bounds[i][1]``. Set ``bounds[i][0]`` to
           :math:`-\infty` if there is no lower bound, and set ``bounds[i][1]``
           to :math:`\infty` if there is no upper bound.

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
    callback : callable, optional

        A callback executed at each objective function evaluation. The method
        terminates if a ``StopIteration`` exception is raised by the callback.
        It should have the signature:

            ``callback(intermediate_result)``

        where ``intermediate_result`` is an instance of
        `scipy.optimize.OptimizeResult`, with attributes ``x`` and ``fun``,
        being the point at which the objective function is evaluated and the
        value of the objective function, respectively.
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
            feasibility_tol : float, optional
                Tolerance on the constraint violation.
            radius_init : float, optional
                Initial trust-region radius.
            radius_final : float, optional
                Final trust-region radius.
            nb_points : int, optional
                Number of interpolation points.
            scale : bool, optional
                Whether to scale the variables according to the bounds.
            filter_size : int, optional
                Maximum number of points in the filter. The filter is used to
                select the best point returned by the optimization procedure.
            store_history : bool, optional
                Whether to store the history of the function evaluations.
            history_size : int, optional
                Maximum number of function evaluations to store in the history.
            debug : bool, optional
                Whether to perform additional checks. This option should be
                used only for debugging purposes and is highly discouraged.

    Returns
    -------
    `scipy.optimize.OptimizeResult`
        Result of the optimization procedure, with the following fields:

            message : str
                Description of the cause of the termination.
            success : bool
                Whether the optimization procedure terminated successfully.
            status : int
                Termination status of the optimization procedure.
            x : `numpy.ndarray`, shape (n,)
                Solution point.
            fun : float
                Objective function value at the solution point.
            cub : `numpy.ndarray`, shape (m_nonlinear_ub,)
                Nonlinear inequality constraint values at the solution point.
            ceq : `numpy.ndarray`, shape (m_nonlinear_eq,)
                Nonlinear equality constraint values at the solution point.
            maxcv : float
                Maximum constraint violation at the solution point.
            nit : int
                Number of iterations.
            nfev : int
                Number of function evaluations.

        If the ``store_history`` option is True, the result also has the
        following fields:

            fun_history : `numpy.ndarray`, shape (nfev,)
                History of the objective function values.
            cub_history : `numpy.ndarray`, shape (nfev, m_nonlinear_ub)
                History of the nonlinear inequality constraint values.
            ceq_history : `numpy.ndarray`, shape (nfev, m_nonlinear_eq)
                History of the nonlinear equality constraint values.

        A description of the termination statuses is given below.

        .. list-table::
            :widths: 25 75
            :header-rows: 1

            * - Exit status
              - Description
            * - 0
              - The lower bound for the trust-region radius has been reached.
            * - 1
              - The target objective function value has been reached.
            * - 2
              - All variables are fixed by the bound constraints.
            * - 3
              - The callback requested to stop the optimization procedure.
            * - 4
              - The maximum number of function evaluations has been exceeded.
            * - 5
              - The maximum number of iterations has been exceeded.
            * - -1
              - The bound constraints are infeasible.
            * - -2
              - A linear algebra error occurred.

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
    .. [3] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, The Hong Kong Polytechnic University, Hong
       Kong, China, 2022.

    Examples
    --------
    Let us first minimize the Rosenbrock function implemented in
    `scipy.optimize` in an unconstrained setting.

    .. testsetup::

        import numpy as np
        np.set_printoptions(precision=3, suppress=True)

    >>> import numpy as np
    >>> from cobyqa import minimize
    >>> from scipy.optimize import Bounds, rosen

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
    >>> bounds = Bounds([0.0, 0.0], np.inf)
    >>> aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    >>> bub = [2.0, 6.0, 2.0]
    >>> res = minimize(fun, x0, bounds=bounds, aub=aub, bub=bub)
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
    else:
        options = dict(options)
    verbose = options.get(Options.VERBOSE, DEFAULT_OPTIONS[Options.VERBOSE])
    verbose = bool(verbose)
    feasibility_tol = options.get(Options.FEASIBILITY_TOL, DEFAULT_OPTIONS[Options.FEASIBILITY_TOL])
    feasibility_tol = float(feasibility_tol)
    scale = options.get(Options.SCALE, DEFAULT_OPTIONS[Options.SCALE])
    scale = bool(scale)
    store_history = options.get(Options.STORE_HISTORY, DEFAULT_OPTIONS[Options.STORE_HISTORY])
    store_history = bool(store_history)
    if Options.HISTORY_SIZE in options and options[Options.HISTORY_SIZE] <= 0:
        raise ValueError('The size of the history must be positive.')
    history_size = options.get(Options.HISTORY_SIZE, DEFAULT_OPTIONS[Options.HISTORY_SIZE])
    history_size = int(history_size)
    if Options.FILTER_SIZE in options and options[Options.FILTER_SIZE] <= 0:
        raise ValueError('The size of the filter must be positive.')
    filter_size = options.get(Options.FILTER_SIZE, DEFAULT_OPTIONS[Options.FILTER_SIZE])
    filter_size = int(filter_size)
    debug = options.get(Options.DEBUG, DEFAULT_OPTIONS[Options.DEBUG])
    debug = bool(debug)

    # Initialize the objective function.
    if not isinstance(args, tuple):
        args = (args,)
    obj = ObjectiveFunction(fun, callback, verbose, store_history, history_size, debug, *args)

    # Initialize the bound constraints.
    if not hasattr(x0, '__len__'):
        x0 = [x0]
    n_orig = len(x0)
    xl, xu = _get_bounds(bounds, n_orig)
    bounds = BoundConstraints(xl, xu)

    # Initialize the linear constraints.
    if aub is None:
        aub = np.empty((0, n_orig))
    if bub is None:
        bub = np.empty(0)
    linear_ub = LinearConstraints(aub, bub, False, debug)
    if aeq is None:
        aeq = np.empty((0, n_orig))
    if beq is None:
        beq = np.empty(0)
    linear_eq = LinearConstraints(aeq, beq, True, debug)

    # Initialize the nonlinear constraints.
    nonlinear_ub = NonlinearConstraints(cub, False, verbose, store_history, history_size, debug, *args)
    nonlinear_eq = NonlinearConstraints(ceq, True, verbose, store_history, history_size, debug, *args)

    # Initialize the problem (and remove the fixed variables).
    pb = Problem(obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq, feasibility_tol, scale, filter_size, debug)

    # Set the default options.
    _set_default_options(options, pb.n)

    # Initialize the models and skip the computations whenever possible.
    if not pb.bounds.is_feasible:
        # The bound constraints are infeasible.
        return _build_result(pb, 0.0, False, ExitStatus.INFEASIBLE_ERROR, 0, options)
    elif pb.n == 0:
        # All variables are fixed by the bound constraints.
        return _build_result(pb, 0.0, True, ExitStatus.FIXED_SUCCESS, 0, options)
    if verbose:
        print('Starting the optimization procedure.')
        print(f'Initial trust-region radius: {options[Options.RHOBEG]}.')
        print(f'Final trust-region radius: {options[Options.RHOEND]}.')
        print(f'Maximum number of function evaluations: {options[Options.MAX_EVAL]}.')
        print(f'Maximum number of iterations: {options[Options.MAX_ITER]}.')
        print()
    try:
        framework = TrustRegion(pb, options)
    except np.linalg.LinAlgError:
        # The construction of the initial interpolation set failed.
        return _build_result(pb, 0.0, False, ExitStatus.LINALG_ERROR, 0, options)
    except StopIteration:
        # The callback raised a StopIteration exception.
        return _build_result(pb, 0.0, True, ExitStatus.CALLBACK_SUCCESS, 0, options)
    if framework.models.target_init:
        # The target on the objective function value has been reached
        return _build_result(pb, framework.penalty, True, ExitStatus.TARGET_SUCCESS, 0, options)
    elif pb.n_eval >= options[Options.MAX_EVAL]:
        # The maximum number of function evaluations has been exceeded.
        return _build_result(pb, framework.penalty, False, ExitStatus.MAX_ITER_WARNING, 0, options)

    # Start the optimization procedure.
    success = False
    n_iter = 0
    k_new = None
    n_short_steps = 0
    n_very_short_steps = 0
    n_alt_models = 0
    while True:
        # Stop the optimization procedure if the maximum number of iterations
        # has been exceeded. We do not write the main loop as a for loop because
        # we want to access the number of iterations outside the loop.
        if n_iter >= options[Options.MAX_ITER]:
            status = ExitStatus.MAX_ITER_WARNING
            break
        n_iter += 1

        # Update the point around which the quadratic models are built.
        if np.linalg.norm(framework.x_best - framework.models.interpolation.x_base) >= 10.0 * framework.radius:
            framework.shift_x_base(options)

        # Evaluate the trial step.
        radius_save = framework.radius
        normal_step, tangential_step = framework.get_trust_region_step(options)
        step = normal_step + tangential_step
        s_norm = np.linalg.norm(step)

        # If the trial step is too short, we do not attempt to evaluate the
        # objective and constraint functions. Instead, we reduce the
        # trust-region radius and check whether the resolution should be
        # reduced and whether the geometry of the interpolation set should be
        # improved. Otherwise, we entertain a classical iteration. The criterion
        # for performing an exceptional jump is taken from NEWUOA.
        if s_norm <= 0.5 * framework.resolution:
            framework.radius *= 0.1
            if radius_save > framework.resolution:
                n_short_steps = 0
                n_very_short_steps = 0
            else:
                n_short_steps += 1
                n_very_short_steps += 1
                if s_norm > 0.1 * framework.resolution:
                    n_very_short_steps = 0
            reduce_resolution = n_short_steps >= 5 or n_very_short_steps >= 3
            if reduce_resolution:
                n_short_steps = 0
                n_very_short_steps = 0
                improve_geometry = False
            else:
                try:
                    k_new, dist_new = framework.get_index_to_remove()
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
                improve_geometry = dist_new > max(framework.radius, 2.0 * framework.resolution)
        else:
            # Increase the penalty parameter if necessary.
            same_best_point = framework.increase_penalty(step)
            if same_best_point:
                # Evaluate the objective and constraint functions.
                try:
                    fun_val, cub_val, ceq_val, target = _eval(pb, framework, step, options)
                except MaxEvalError:
                    status = ExitStatus.MAX_EVAL_WARNING
                    break
                except StopIteration:
                    status = ExitStatus.CALLBACK_SUCCESS
                    success = True
                    break
                if target:
                    status = ExitStatus.TARGET_SUCCESS
                    success = True
                    break

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
                            status = ExitStatus.MAX_EVAL_WARNING
                            break
                        except StopIteration:
                            status = ExitStatus.CALLBACK_SUCCESS
                            success = True
                            break
                        if target:
                            status = ExitStatus.TARGET_SUCCESS
                            success = True
                            break

                # Calculate the reduction ratio.
                ratio = framework.get_reduction_ratio(step, fun_val, cub_val, ceq_val)

                # Choose an interpolation point to remove.
                try:
                    k_new = framework.get_index_to_remove(framework.x_best + step)[0]
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break

                # Update the interpolation set.
                try:
                    ill_conditioned = framework.models.update_interpolation(k_new, framework.x_best + step, fun_val, cub_val, ceq_val)
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
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
                        try:
                            grad_alt = framework.models.fun_alt_grad(framework.x_best)
                        except np.linalg.LinAlgError:
                            status = ExitStatus.LINALG_ERROR
                            break
                        if np.linalg.norm(grad) < 10.0 * np.linalg.norm(grad_alt):
                            n_alt_models = 0
                        if n_alt_models >= 3:
                            try:
                                framework.models.reset_models()
                            except np.linalg.LinAlgError:
                                status = ExitStatus.LINALG_ERROR
                                break
                            n_alt_models = 0

                # Update the Lagrange multipliers.
                framework.set_multipliers(framework.x_best + step)

                # Check whether the resolution should be reduced.
                try:
                    k_new, dist_new = framework.get_index_to_remove()
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
                improve_geometry = ill_conditioned or ratio <= 0.1 and dist_new > max(framework.radius, 2.0 * framework.resolution)
                reduce_resolution = radius_save <= framework.resolution and ratio <= 0.1 and not improve_geometry
            else:
                # When increasing the penalty parameter, the best point so far
                # may change. In this case, we restart the iteration.
                reduce_resolution = False
                improve_geometry = False

        # Reduce the resolution if necessary.
        if reduce_resolution:
            if framework.resolution <= options[Options.RHOEND]:
                success = True
                status = ExitStatus.RADIUS_SUCCESS
                break
            framework.reduce_resolution(options)
            framework.decrease_penalty()

            if verbose:
                maxcv_val = pb.maxcv(framework.x_best, framework.cub_best, framework.ceq_best)
                _print_step(f'New trust-region radius: {framework.resolution}', pb, pb.build_x(framework.x_best), framework.fun_best, maxcv_val, pb.n_eval, n_iter)
                print()

        # Improve the geometry of the interpolation set if necessary.
        if improve_geometry:
            try:
                step = framework.get_geometry_step(k_new, options)
            except np.linalg.LinAlgError:
                status = ExitStatus.LINALG_ERROR
                break

            # Evaluate the objective and constraint functions.
            try:
                fun_val, cub_val, ceq_val, target = _eval(pb, framework, step, options)
            except MaxEvalError:
                status = ExitStatus.MAX_EVAL_WARNING
                break
            except StopIteration:
                status = ExitStatus.CALLBACK_SUCCESS
                success = True
                break
            if target:
                status = ExitStatus.TARGET_SUCCESS
                success = True
                break

            # Update the interpolation set.
            try:
                framework.models.update_interpolation(k_new, framework.x_best + step, fun_val, cub_val, ceq_val)
            except np.linalg.LinAlgError:
                status = ExitStatus.LINALG_ERROR
                break
            framework.set_best_index()

    return _build_result(pb, framework.penalty, success, status, n_iter, options)


def _get_bounds(bounds, n):
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    elif isinstance(bounds, Bounds):
        return bounds.lb, bounds.ub
    elif hasattr(bounds, '__len__'):
        bounds = np.asarray(bounds)
        if bounds.shape != (n, 2):
            raise ValueError('The shape of the bounds is not compatible with the number of variables.')
        return bounds[:, 0], bounds[:, 1]
    else:
        raise TypeError('The bounds must be an instance of scipy.optimize.Bounds or an array-like object.')


def _set_default_options(options, n):
    """
    Set the default options.
    """
    if Options.RHOBEG in options and options[Options.RHOBEG] <= 0.0:
        raise ValueError('The initial trust-region radius must be positive.')
    if Options.RHOEND in options and options[Options.RHOEND] < 0.0:
        raise ValueError('The final trust-region radius must be nonnegative.')
    if Options.RHOBEG in options and Options.RHOEND in options:
        if options[Options.RHOBEG] < options[Options.RHOEND]:
            raise ValueError('The initial trust-region radius must be greater than or equal to the final trust-region radius.')
    elif Options.RHOBEG in options:
        options[Options.RHOEND.value] = min(DEFAULT_OPTIONS[Options.RHOEND], options[Options.RHOBEG])
    elif Options.RHOEND in options:
        options[Options.RHOBEG.value] = max(DEFAULT_OPTIONS[Options.RHOBEG], options[Options.RHOEND])
    else:
        options[Options.RHOBEG.value] = DEFAULT_OPTIONS[Options.RHOBEG]
        options[Options.RHOEND.value] = DEFAULT_OPTIONS[Options.RHOEND]
    options[Options.RHOBEG.value] = float(options[Options.RHOBEG])
    options[Options.RHOEND.value] = float(options[Options.RHOEND])
    if Options.NPT in options and options[Options.NPT] <= 0:
        raise ValueError('The number of interpolation points must be positive.')
    if Options.NPT in options and options[Options.NPT] > ((n + 1) * (n + 2)) // 2:
        raise ValueError(f'The number of interpolation points must be at most {((n + 1) * (n + 2)) // 2}.')
    options.setdefault(Options.NPT.value, DEFAULT_OPTIONS[Options.NPT](n))
    options[Options.NPT.value] = int(options[Options.NPT])
    if Options.MAX_EVAL in options and options[Options.MAX_EVAL] <= 0:
        raise ValueError('The maximum number of function evaluations must be positive.')
    options.setdefault(Options.MAX_EVAL.value, max(DEFAULT_OPTIONS[Options.MAX_EVAL](n), options[Options.NPT] + 1))
    options[Options.MAX_EVAL.value] = int(options[Options.MAX_EVAL])
    if Options.MAX_ITER in options and options[Options.MAX_ITER] <= 0:
        raise ValueError('The maximum number of iterations must be positive.')
    options.setdefault(Options.MAX_ITER.value, DEFAULT_OPTIONS[Options.MAX_ITER](n))
    options[Options.MAX_ITER.value] = int(options[Options.MAX_ITER])
    options.setdefault(Options.TARGET.value, DEFAULT_OPTIONS[Options.TARGET])
    options[Options.TARGET.value] = float(options[Options.TARGET])
    options.setdefault(Options.FEASIBILITY_TOL.value, DEFAULT_OPTIONS[Options.FEASIBILITY_TOL])
    options[Options.FEASIBILITY_TOL.value] = float(options[Options.FEASIBILITY_TOL])
    options.setdefault(Options.VERBOSE.value, DEFAULT_OPTIONS[Options.VERBOSE])
    options[Options.VERBOSE.value] = bool(options[Options.VERBOSE])
    options.setdefault(Options.SCALE.value, DEFAULT_OPTIONS[Options.SCALE])
    options[Options.SCALE.value] = bool(options[Options.SCALE])
    options.setdefault(Options.FILTER_SIZE.value, DEFAULT_OPTIONS[Options.FILTER_SIZE])
    options[Options.FILTER_SIZE.value] = int(options[Options.FILTER_SIZE])
    options.setdefault(Options.STORE_HISTORY.value, DEFAULT_OPTIONS[Options.STORE_HISTORY])
    options[Options.STORE_HISTORY.value] = bool(options[Options.STORE_HISTORY])
    options.setdefault(Options.HISTORY_SIZE.value, DEFAULT_OPTIONS[Options.HISTORY_SIZE])
    options[Options.HISTORY_SIZE.value] = int(options[Options.HISTORY_SIZE])
    options.setdefault(Options.DEBUG.value, DEFAULT_OPTIONS[Options.DEBUG])
    options[Options.DEBUG.value] = bool(options[Options.DEBUG])

    # Check whether they are any unknown options.
    for key in options:
        if key not in Options.__members__.values():
            warnings.warn(f'Unknown option: {key}.', RuntimeWarning, 3)


def _eval(pb, framework, step, options):
    """
    Evaluate the objective and constraint functions.
    """
    if pb.n_eval >= options[Options.MAX_EVAL]:
        raise MaxEvalError
    x_eval = framework.x_best + step
    fun_val, cub_val, ceq_val = pb(x_eval)
    r_val = pb.maxcv(x_eval, cub_val, ceq_val)
    return fun_val, cub_val, ceq_val, fun_val <= options[Options.TARGET] and r_val <= options[Options.FEASIBILITY_TOL]


def _build_result(pb, penalty, success, status, n_iter, options):
    """
    Build the result of the optimization process.
    """
    # Build the result.
    x, fun, cub, ceq = pb.best_eval(penalty)
    maxcv = pb.maxcv(x, cub, ceq)
    if status != ExitStatus.TARGET_SUCCESS:
        success = success and maxcv <= options[Options.FEASIBILITY_TOL]
    result = OptimizeResult()
    result.message = {
        ExitStatus.RADIUS_SUCCESS: 'The lower bound for the trust-region radius has been reached',
        ExitStatus.TARGET_SUCCESS: 'The target objective function value has been reached',
        ExitStatus.FIXED_SUCCESS: 'All variables are fixed by the bound constraints',
        ExitStatus.CALLBACK_SUCCESS: 'The callback requested to stop the optimization procedure',
        ExitStatus.MAX_EVAL_WARNING: 'The maximum number of function evaluations has been exceeded',
        ExitStatus.MAX_ITER_WARNING: 'The maximum number of iterations has been exceeded',
        ExitStatus.INFEASIBLE_ERROR: 'The bound constraints are infeasible',
        ExitStatus.LINALG_ERROR: 'A linear algebra error occurred',
    }.get(status, 'Unknown exit status')
    result.success = success
    result.status = status.value
    result.x = pb.build_x(x)
    result.fun = fun
    result.cub = cub
    result.ceq = ceq
    result.maxcv = maxcv
    result.nfev = pb.n_eval
    result.nit = n_iter
    if options[Options.STORE_HISTORY]:
        result.fun_history = pb.fun_history
        result.cub_history = pb.cub_history
        result.ceq_history = pb.ceq_history

    # Print the result if requested.
    if options[Options.VERBOSE]:
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
    if not pb.is_feasibility:
        print(f'Least value of {pb.fun_name}: {fun_val}.')
    print(f'Maximum constraint violation: {r_val}.')
    with np.printoptions(**PRINT_OPTIONS):
        print(f'Corresponding point: {x}.')
