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
        except KeyError as e:
            raise AttributeError(name) from e

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
        else:
            return f'{self.__class__.__name__}()'


def minimize(fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None, Aeq=None,
             beq=None, cub=None, ceq=None, options=None, **kwargs):
    """
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
    """
    # Build the initial models of the optimization problem. The computations
    # must be stopped immediately if all indices are fixed by the bound
    # constraints or if the target function value has been reached by an initial
    # interpolation point (in which case the initial models are not built).
    framework = TrustRegion(fun, x0, args, xl, xu, Aub, bub, Aeq, beq, cub, ceq,
                            options, **kwargs)
    nf = framework.kopt + 1
    if np.all(framework.ifix):
        exit_status = 13
    elif framework.target_reached:
        exit_status = 1
    else:
        exit_status = 0
        nf = framework.npt

    # Begin the iterative procedure.
    rho = framework.rhobeg
    delta = rho
    nit = 0
    itest = 0
    while exit_status == 0:
        # Update the shift of the origin to manage computer rounding errors.
        framework.shift_origin(delta)

        # Evaluate the trial step.
        delsav = delta
        fopt = framework.fopt
        kopt = framework.kopt
        coptub = np.copy(framework.coptub)
        copteq = np.copy(framework.copteq)
        xopt = np.copy(framework.xopt)
        is_trust_region_step = not framework.is_model_step
        nit += 1
        if is_trust_region_step:
            step = framework.trust_region_step(delta, **kwargs)
            snorm = np.linalg.norm(step)
            if snorm <= .5 * delta:
                delta = rho if delta <= 1.4 * rho else .5 * delta
                if delsav > rho:
                    framework.prepare_model_step(delta)
                    continue
        else:
            step = framework.model_step(max(.1 * delta, rho), **kwargs)
            snorm = np.linalg.norm(step)

        if not is_trust_region_step or snorm > .5 * delta:
            # Evaluate the objective function, include the trial point in the
            # interpolation set, and update accordingly the models.
            if nf >= framework.maxfev:
                exit_status = 3
                break
            nf += 1
            try:
                mopt, ratio = framework.update(step, **kwargs)
            except RestartRequiredException:
                continue
            except ZeroDivisionError:
                exit_status = 9
                break
            if framework.target_reached:
                exit_status = 1
                break

            # Update the trust-region radius.
            if is_trust_region_step:
                if ratio <= .1:
                    delta *= .5
                elif ratio <= .7:
                    delta = max(.5 * delta, snorm)
                else:
                    delbd = np.sqrt(2.) * delta
                    delta = min(delbd, max(.2 * delta, 2. * snorm))
                if delta <= 1.5 * rho:
                    delta = rho

            # Attempt to replace the models by the alternative ones.
            if is_trust_region_step and delta <= rho:
                if ratio > 1e-2:
                    itest = 0
                else:
                    itest += 1
                    gd = framework.model_obj_grad(framework.xopt)
                    gd_alt = framework.model_obj_alt_grad(framework.xopt)
                    if np.linalg.norm(gd) < 10. * np.linalg.norm(gd_alt):
                        itest = 0
                    if itest >= 3:
                        framework.reset_models()
                        itest = 0

            # If a trust-region step has provided a sufficient decrease or if a
            # model-improvement step has just been computed, then the next
            # iteration is a trust-region step.
            if not is_trust_region_step or ratio >= .1:
                framework.prepare_trust_region_step()
                continue

            # If an interpolation point is substantially far from the
            # trust-region center, a model-improvement step is entertained.
            framework.prepare_model_step(max(delta, 2. * rho))
            if framework.is_model_step or delsav > rho:
                continue
            ropt = framework.rval[framework.kopt]
            msav = framework(xopt, fopt, coptub, copteq)
            rsav = framework.rval[kopt]
            if framework.less_merit(mopt, ropt, msav, rsav):
                continue

        # Update the lower bound on the trust-region radius.
        if rho > framework.rhoend:
            delta = .5 * rho
            if rho > 2.5e2 * framework.rhoend:
                rho *= .1
            elif rho <= 1.6e1:
                rho = framework.rhoend
            else:
                rho = np.sqrt(rho * framework.rhoend)
            delta = max(delta, rho)
            framework.prepare_trust_region_step()
            framework.reduce_penalty_coefficients()
            if framework.disp:
                message = f'New trust-region radius: {rho}.'
                _print(framework, fun.__name__, nf, message)
            continue
        break

    # Get the success flag.
    if exit_status == 13:
        bdtol = get_bdtol(framework.xl, framework.xu, **kwargs)
        success = framework.type == 'U' or framework.maxcv <= bdtol
    else:
        success = exit_status in [0, 1]

    # Build the result structure and return.
    result = OptimizeResult()
    result.x = framework.get_x(framework.xbase + framework.xopt)
    result.fun = framework.fopt
    result.jac = np.full_like(result.x, np.nan)
    with suppress(AttributeError):
        free_indices = np.logical_not(framework.ifix)
        result.jac[free_indices] = framework.model_obj_grad(framework.xopt)
    result.nfev = nf
    result.nit = nit
    if framework.type != 'U':
        result.maxcv = framework.maxcv
    result.status = exit_status
    result.success = success
    result.message = {
        0: 'Lower bound for the trust-region radius has been reached.',
        1: 'Target function value has been achieved.',
        3: 'Maximum number of function evaluations has been exceeded.',
        9: 'Denominator of the updating formula is zero.',
        13: 'All variables are fixed by the constraints.',
    }.get(exit_status, 'Unknown exit status.')
    if framework.disp:
        _print(framework, fun.__name__, nf, result.message)
    return result


def _print(problem, fun, nf, message):
    x_full = problem.get_x(problem.xbase + problem.xopt)
    print()
    print(message)
    print(f'Number of function evaluations: {nf}.')
    print(f'Least value of {fun}: {problem.fopt}.')
    print(f'Corresponding point: {x_full}.')
    print()
