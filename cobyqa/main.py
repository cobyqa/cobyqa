from contextlib import suppress

import numpy as np

from .linalg.utils import get_bdtol
from .optimize import TrustRegion
from .utils import RestartRequiredException


class OptimizeResult(dict):
    """
    Structure for the result of an optimization algorithm.

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


def minimize(fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None, Aeq=None,
             beq=None, cub=None, ceq=None, options=None, **kwargs):
    r"""
    Minimize a real-valued function.

    The minimization can be subject to bound, linear inequality, linear
    equality, nonlinear inequality, and nonlinear equality constraints using a
    derivative-free trust-region SQP method. Although the solver may tackle
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
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints (the default is
        ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``, where the values
        of ``xl`` and ``xu`` evolve to include the shift of the origin).
    lctol : float, optional
        Tolerance for comparisons on the linear constraints (the default is
        ``10 * eps * max(mlub, n) * max(1, max(abs(bub)))``, where the values
        of ``bub`` evolve to include the shift of the origin).
    lstol : float, optional
        Tolerance on the approximate KKT conditions for the calculations of the
        least-squares Lagrange multipliers (the default is
        ``10 * eps * max(n, m) * max(1, max(abs(g)))``, where ``g`` is the
        gradient of the current model of the objective function).

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
    struct = TrustRegion(fun, x0, xl, xu, Aub, bub, Aeq, beq, cub, ceq, options,
                         *args, **kwargs)
    nfev = struct.kopt + 1
    if np.all(struct.ifix):
        exit_status = 8
    elif struct.target_reached:
        exit_status = 1
    else:
        exit_status = 0
        nfev = struct.npt

    # Begin the iterative procedure.
    rho = struct.rhobeg
    delta = rho
    fsav = struct.fopt
    xsav = struct.get_x(struct.xbase + struct.xopt)
    iact = struct.active_set(struct.xopt, **kwargs)
    nit = 0
    itest = 0
    while exit_status == 0:
        # Update the shift of the origin to manage computer rounding errors.
        struct.shift_origin(delta)

        # Evaluate the trial step.
        delsav = delta
        fopt = struct.fopt
        kopt = struct.kopt
        coptub = np.copy(struct.coptub)
        copteq = np.copy(struct.copteq)
        xopt = np.copy(struct.xopt)
        is_trust_region_step = not struct.is_model_step
        nit += 1
        test = 0.5 / np.sqrt(2.0)
        if is_trust_region_step:
            step = struct.trust_region_step(delta, **kwargs)
            snorm = np.linalg.norm(step)
            inew = struct.active_set(struct.xopt + step, **kwargs)
            if not np.array_equal(iact, inew):
                test = 0.1999 / np.sqrt(2.0)
                iact = inew
            if snorm <= test * delta:
                delta = 0.5 * delta
                if delta <= 1.4 * rho:
                    delta = rho
                if delsav > rho:
                    struct.prepare_model_step(delta)
                    continue
        else:
            step = struct.model_step(max(0.1 * delta, rho), **kwargs)
            snorm = np.linalg.norm(step)

        if not is_trust_region_step or snorm > test * delta:
            # Evaluate the objective function, include the trial point in the
            # interpolation set, and update accordingly the models.
            if nfev >= struct.maxfev:
                exit_status = 6
                break
            nfev += 1
            try:
                fx, mopt, ratio = struct.update(step, **kwargs)
            except RestartRequiredException:
                continue
            except ZeroDivisionError:
                exit_status = 7
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
            if is_trust_region_step:
                if ratio <= 0.1:
                    delta *= 0.5
                elif ratio <= 0.7:
                    delta = max(0.5 * delta, snorm)
                else:
                    delbd = np.sqrt(2.0) * delta
                    delta = min(delbd, max(0.2 * delta, 2.0 * snorm))
                if delta <= 1.5 * rho:
                    delta = rho

            # Attempt to replace the models by the alternative ones.
            if is_trust_region_step and delta <= rho:
                if ratio > 1e-2:
                    itest = 0
                else:
                    itest += 1
                    gd = struct.model_obj_grad(struct.xopt)
                    gd_alt = struct.model_obj_alt_grad(struct.xopt)
                    if np.linalg.norm(gd) < 10.0 * np.linalg.norm(gd_alt):
                        itest = 0
                    if itest >= 3:
                        struct.reset_models()
                        itest = 0

            # If a trust-region step has provided a sufficient decrease or if a
            # model-improvement step has just been computed, then the next
            # iteration is a trust-region step.
            if not is_trust_region_step or ratio >= 0.1:
                struct.prepare_trust_region_step()
                continue

            # If an interpolation point is substantially far from the
            # trust-region center, a model-improvement step is entertained.
            struct.prepare_model_step(max(delta, 2.0 * rho))
            if struct.is_model_step or delsav > rho:
                continue
            ropt = struct.rval[struct.kopt]
            msav = struct(xopt, fopt, coptub, copteq)
            rsav = struct.rval[kopt]
            if struct.less_merit(mopt, ropt, msav, rsav):
                continue

        # Update the lower bound on the trust-region radius.
        if rho > struct.rhoend:
            delta = 0.5 * rho
            if rho > 250.0 * struct.rhoend:
                rho *= 0.1
            elif rho <= 16.0:
                rho = struct.rhoend
            else:
                rho = np.sqrt(rho * struct.rhoend)
            delta = max(delta, rho)
            struct.prepare_trust_region_step()
            struct.reduce_penalty_coefficients()
            if struct.disp:
                message = f'New trust-region radius: {rho}.'
                _print(struct, fun.__name__, nfev, message)
            continue
        break

    # Get the success flag.
    if exit_status == 8:
        bdtol = get_bdtol(struct.xl, struct.xu, **kwargs)
        success = struct.type == 'U' or struct.maxcv <= bdtol
    else:
        success = exit_status in [0, 1, 2, 3, 4, 5]

    # Build the result structure and return.
    result = OptimizeResult()
    result.x = struct.get_x(struct.xbase + struct.xopt)
    result.fun = struct.fopt
    result.jac = np.full_like(result.x, np.nan)
    with suppress(AttributeError):
        free_indices = np.logical_not(struct.ifix)
        result.jac[free_indices] = struct.model_obj_grad(struct.xopt)
    result.nfev = nfev
    result.nit = nit
    if struct.type != 'U':
        result.maxcv = struct.maxcv
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
        7: 'Denominator of the updating formula is zero.',
        8: 'All variables are fixed by the constraints.',
        -1: 'Bound constraints are infeasible',
    }.get(exit_status, 'Unknown exit status.')
    if struct.disp:
        _print(struct, fun.__name__, nfev, result.message)
    return result


def _print(problem, fun, nf, message):
    x_full = problem.get_x(problem.xbase + problem.xopt)
    print()
    print(message)
    print(f'Number of function evaluations: {nf}.')
    print(f'Least value of {fun}: {problem.fopt}.')
    if problem.type not in 'UB':
        print(f'Maximum constraint violation: {problem.maxcv}.')
    print(f'Corresponding point: {x_full}.')
    print()
